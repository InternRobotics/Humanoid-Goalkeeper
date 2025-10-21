import os
import math
import torch
import random
import pickle
from tqdm import tqdm
from legged_gym.utils.math import (
    euler_xyz_to_quat,
    quat_apply_yaw,
    quat_apply_yaw_inverse,
    quat_mul, quat_conjugate,
    quat_mul_yaw_inverse,
    quat_mul_yaw,
    quat_mul,
    quat_apply,
)
# from isaacgym.torch_utils import quat_apply, normalize
import copy
import torch


def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return torch.cat([roll_x.view(-1, 1), pitch_y.view(-1, 1), yaw_z.view(-1, 1)], dim=1)


def load_imitation_dataset(folder, mapping="joint_id.txt", suffix=".pt"):
    # List all files with the given suffix (e.g., .pt)
    filenames = [name for name in os.listdir(folder) if name.endswith(suffix)]
    
    multidataset = {}
    for filename in tqdm(filenames):
        try:
            # Load the tensor data from each file
            
            dataset = {}
            data = torch.load(os.path.join(folder, filename))
            dataset[filename[:-len(suffix)]] = data
            # Use the filename without the suffix as the key in dataset
            dataset_list = list(dataset.values())
            random.shuffle(dataset_list)

            multidataset[filename[:-len(suffix)]] = dataset_list

        except Exception as e:
            print(f"{filename} load failed!!! Error: {e}")
            continue


    # Read and process the joint_id mapping
    with open(mapping, "r") as file:
        lines = file.readlines()

    # Process the joint ID mapping into a dictionary
    lines = [line.strip().split(" ") for line in lines]
    joint_id_dict = {k: int(v) for v, k in lines}

    return multidataset, joint_id_dict


class MotionLib:
    def __init__(self, datasets, mapping, dof_names, keyframe_names, fps=30, min_dt=0.1, device="cpu", amp_obs_type='keyframe', num_steps=2):
        self.device, self.fps = device, fps
        self.env_fps = 50
        self.num_steps = num_steps
        get_len = lambda x: list(x.values())[0].shape[0]

        datasets = [data for data in datasets if get_len(data) > max(math.ceil(min_dt * fps), 3)]

        self.motion_len = torch.tensor([get_len(data) for data in datasets], dtype=torch.long, device=device)
        self.num_motion, self.tot_len = self.motion_len.shape[0], self.motion_len.sum()
        self.motion_sampling_prob = torch.ones(self.num_motion, dtype=torch.float, device=device)

        # import ipdb; ipdb.set_trace()
        self.motion_end_ids = torch.cumsum(self.motion_len, dim=0)
        self.motion_start_ids = torch.nn.functional.pad(self.motion_end_ids, (1, -1), "constant", 0)
        
        self.motion_base_rpy = torch.zeros(self.tot_len, 3, dtype=torch.float, device=device)
        self.motion_base_pos = torch.zeros(self.tot_len, 3, dtype=torch.float, device=device)
        self.motion_base_lin_vel = torch.zeros(self.tot_len, 3, dtype=torch.float, device=device)
        self.motion_base_ang_vel = torch.zeros(self.tot_len, 3, dtype=torch.float, device=device)
        self.motion_dof_pos = torch.zeros(self.tot_len, len(dof_names), dtype=torch.float, device=device)
        self.motion_dof_vel = torch.zeros(self.tot_len, len(dof_names), dtype=torch.float, device=device)
        self.motion_keyframe_pos = torch.zeros(self.tot_len, len(keyframe_names), 3, dtype=torch.float, device=device)
        self.motion_keyframe_rpy = torch.zeros(self.tot_len, len(keyframe_names), 3, dtype=torch.float, device=device)
        self.motion_keyframe_lin_vel = torch.zeros(self.tot_len, len(keyframe_names), 3, dtype=torch.float, device=device)
        self.motion_keyframe_ang_vel = torch.zeros(self.tot_len, len(keyframe_names), 3, dtype=torch.float, device=device)
        
        self.motion_keyframe_pos_local = torch.zeros(self.tot_len, len(keyframe_names), 3, dtype=torch.float, device=device)
        self.motion_keyframe_quat_local = torch.zeros(self.tot_len, len(keyframe_names), 4, dtype=torch.float, device=device)

        for i, traj in enumerate(tqdm(datasets)):
            start, end = self.motion_start_ids[i], self.motion_end_ids[i]

            self.motion_base_pos[start:end] = torch.tensor(traj["base_position"], dtype=torch.float, device=device)
            #! Note: Quat to RPY, not sure the correctness
            self.motion_base_rpy[start:end] = torch.tensor(euler_from_quaternion(traj["base_pose"]), dtype=torch.float, device=device)   
            self.motion_base_lin_vel[start:end-1] = (self.motion_base_pos[start+1:end] - self.motion_base_pos[start:end-1]) * self.fps
            self.motion_base_ang_vel[start:end-1] = (self.motion_base_rpy[start+1:end] - self.motion_base_rpy[start:end-1]) * self.fps
            self.motion_base_lin_vel[end-1:end] = self.motion_base_lin_vel[end-2:end-1]
            self.motion_base_ang_vel[end-1:end] = self.motion_base_ang_vel[end-2:end-1]
            
            dof_pos = torch.tensor(traj["joint_position"], dtype=torch.float, device=device)
            dof_vel = torch.tensor(traj["joint_velocity"], dtype=torch.float, device=device)
            for j, name in enumerate(dof_names):
                if name in mapping.keys():
                    self.motion_dof_pos[start:end, j] = dof_pos[:, mapping[name]]
                    self.motion_dof_vel[start:end, j] = dof_vel[:, mapping[name]]
                    if j == 14:
                        self.motion_dof_pos[start:end, j] = 0.

            for k, name in enumerate(keyframe_names):
                # import ipdb; ipdb.set_trace()
                self.motion_keyframe_pos[start:end, k] = torch.tensor(traj["link_position"][:, k], dtype=torch.float, device=device)
                self.motion_keyframe_rpy[start:end, k] = torch.tensor(euler_from_quaternion(traj["link_oritentation"][:, k]), dtype=torch.float, device=device)
                self.motion_keyframe_lin_vel[start:end, k] = torch.tensor(traj["lin_velocity"][:, k], dtype=torch.float, device=device)
                self.motion_keyframe_ang_vel[start:end, k] = torch.tensor(traj["link_angular_velocity"][:, k], dtype=torch.float, device=device)
            
            self.motion_keyframe_pos[start:end, :, 0:2] -= self.motion_base_pos[start:start+1, None, 0:2]
            self.motion_base_pos[start:end, 0:2] -= self.motion_base_pos[start:start+1, 0:2].clone()

            self.motion_keyframe_pos[start:end, :, 2] -= -0.2
            self.motion_base_pos[start:end, 2] -= -0.2

            # !note: the yaw maybe inaccurate
            local_rotation = euler_xyz_to_quat(self.motion_base_rpy[start:end])[:, None]
            self.motion_keyframe_pos_local[start:end] = quat_apply_yaw_inverse(local_rotation.clone(), self.motion_keyframe_pos[start:end] - self.motion_base_pos[start:end][:, None]) 
            # import ipdb; ipdb.set_trace()
            self.motion_keyframe_quat_local[start:end] = quat_mul_yaw_inverse(local_rotation.clone(), euler_xyz_to_quat(self.motion_keyframe_rpy[start:end]))

        self.amp_obs_type = amp_obs_type



    @staticmethod    
    def calc_blend(motion, time0, time1, w0, w1):
        motion0, motion1 = motion[time0], motion[time1]
        new_w0 = w0.reshape(w0.shape + (1,) * (motion0.dim() - w0.dim()))
        new_w1 = w1.reshape(w1.shape + (1,) * (motion1.dim() - w1.dim()))
        return new_w0 * motion0 + new_w1 * motion1
        
    def get_expert_obs(self, batch_size):
        motion_ids = torch.randint(0, self.num_motion, (batch_size,), device=self.device)
        start_ids = self.motion_start_ids[motion_ids]
        end_ids = self.motion_end_ids[motion_ids]
        motion_len = self.motion_len[motion_ids]

        time_in_proportion = torch.rand(batch_size, device=self.device)
        clip_tail_proportion = (self.num_steps / motion_len)
        
        # Fix: Convert 0 to a tensor with the same device/dtype as clip_tail_proportion
        min_val = torch.zeros_like(clip_tail_proportion)
        time_in_proportion = time_in_proportion.clamp(min_val, 1 - clip_tail_proportion)

        motion_ids = start_ids + torch.floor(time_in_proportion * (end_ids - start_ids)).long()
        motion_dof = self.motion_dof_pos[motion_ids].view(batch_size, -1)

        ratio = self.fps / self.env_fps
        ratio *= torch.rand(batch_size, device=self.device) * 1.0 + 0.25  # Random ratio per sample

        for i in range(1, self.num_steps):
            next_pos = motion_ids + i * ratio
            floor = torch.floor(next_pos).long()
            ceil = floor + 1
            
            max_idx = self.motion_dof_pos.shape[0] - 1
            floor = torch.clamp(floor, 0, max_idx)
            ceil = torch.clamp(ceil, 0, max_idx)

            linear_ratio = (next_pos - floor).unsqueeze(-1)
            motion_dof_next = self.motion_dof_pos[floor] * (1 - linear_ratio) + self.motion_dof_pos[ceil] * linear_ratio
            motion_dof = torch.cat([motion_dof, motion_dof_next], dim=-1).view(batch_size, -1)

        return motion_dof
