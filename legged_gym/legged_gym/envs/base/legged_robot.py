
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import copy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import random
import torch
from torch import Tensor
import torchvision
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg


from legged_gym.envs.g1.g1_utils import (
    MotionLib, 
    load_imitation_dataset
)



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
    
    return roll_x, pitch_y, yaw_z

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg

        

        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.amp_obs_type = self.cfg.amp.obs_type
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_privileged_obs = self.cfg.env.num_privileged_obs
        self.actor_history_length = self.cfg.env.num_actor_history

        self.actor_obs_length = self.cfg.env.num_observations

        self._init_buffers()
        self._prepare_reward_function()
        self.num_amp_obs = cfg.amp.num_obs
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
    
        self.delayed_actions = self.actions.clone().view(1, self.num_envs, self.num_actions).repeat(self.cfg.control.decimation, 1, 1)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
                
        # Randomize Joint Injections
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
            self.joint_injection[:, self.curriculum_dof_indices] = 0.
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.delayed_actions[_]).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def get_amp_observations(self):
        """ with keys
        """

        return self.dof_pos.clone()



    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.catchstep -= 1
        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        
        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_lin_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,7:10])
        self.base_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,10:13])

        self.torso_pos = self.rigid_body_states[:, self.torso_index, 0:3]

        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_gravity[:] = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index, 3:7], self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt
        
        self.feet_pos[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 3:7]
        self.feet_vel[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self.left_feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.left_feet_indices, 0:3]
        self.right_feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.right_feet_indices, 0:3]
        
        # compute contact related quantities
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        self.first_contacts = (self.feet_air_time >= self.dt) * self.contact_filt
        self.feet_air_time += self.dt
        
        # compute joint powers
        joint_powers = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((joint_powers, self.joint_powers[:, :-1]), dim=1)
        
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.compute_reward()

        self.check_termination()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        termination_privileged_obs = self.compute_termination_observations(env_ids)

        self.reset_idx(env_ids)

            
        balllocal =  self.ball_states[:, 0] - self.env_origins[:, 0]
        approachidx = ((balllocal < 0.5) & (balllocal > 0.1) & (self.ball_states[:,7]  - self.ball_vel < 2.0)).nonzero(as_tuple=False).flatten()
        self.end_target[approachidx, :] = self.ball_states[approachidx, :3].clone()
        self.end_target[:, 0] = torch.clip(self.end_target[:, 0], min = self.env_origins[:, 0] + 0.1, max = self.env_origins[:, 0] + 1.0)
        

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        
        # reset contact related quantities
        self.feet_air_time *= ~self.contact_filt

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, termination_privileged_obs

    def check_termination(self):
        """ Check if environments need to be reset
        """

        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 10., dim=1)
        fall_buf = torch.min(self.rigid_body_states[:, self.knee_indices, 2], dim = -1).values < 0.10
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.gravity_termination_buf = torch.any(torch.norm(self.projected_gravity[:, 0:2], dim=-1, keepdim=True) > 0.8, dim=1)
        
        sharpforce_buf = torch.mean(torch.norm(self.contact_forces[:, self.contact_feet_indices, :], dim=-1), dim = -1) > 1.5 * self.cfg.rewards.max_contact_force
        large_landvel_buf = torch.min(self.rigid_body_states[:, self.contact_feet_indices, 9], dim = -1).values < -3.0
        
        
        yaw_buf = torch.abs(self.yaw) > 0.7

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.gravity_termination_buf
        self.reset_buf |= fall_buf
        self.reset_buf |= yaw_buf
        self.reset_buf |= sharpforce_buf
        self.reset_buf |= large_landvel_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        # print(torch.mean(self.episode_sums["handreach"]) / self.max_episode_length, 0.5 * self.reward_scales["handreach"])
        if len(env_ids) == 0:
            return


        self.refresh_actor_rigid_shape_props(env_ids)

        self._reset_dofs(env_ids)
            
        self._reset_root_states(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.joint_powers[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        
         #reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (len(env_ids), self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
            self.actuation_offset[:, self.curriculum_dof_indices] = 0.
        self.reach_goal_timer[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.


        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

            
        if (self.common_step_counter - self.last_step_counter) >500:

            self.curriculumupdate = int(torch.mean(self.episode_length_buf[env_ids].float()) / 50.)
            self.startstep = 50 - random.randint(3, 10)

            self.command_ranges[:, 0] =  torch.clip(self.command_ranges[:, 0] - 0.2 * self.curriculumupdate, self.command_bound[:,0], self.command_bound[:,1])
            self.command_ranges[:, 1] =  torch.clip(self.command_ranges[:, 1] + 0.2 * self.curriculumupdate,  self.command_bound[:,0], self.command_bound[:,1])
            self.command_ranges[:, 2] =  torch.clip(self.command_ranges[:, 2] - 0.2 * self.curriculumupdate, self.command_bound[:,2], self.command_bound[:,3])
            self.command_ranges[:, 3] =  torch.clip(self.command_ranges[:, 3] + 0.2 * self.curriculumupdate, self.command_bound[:,2], self.command_bound[:,3])

            # for i in range(self.num_dof):

            #     m = (self.hard_dof_pos_limits[i, 0] + self.hard_dof_pos_limits[i, 1]) / 2
            #     r = self.hard_dof_pos_limits[i, 1] - self.hard_dof_pos_limits[i, 0]
            #     self.curriculum_dof_pos_limits[i, 0] = m - 0.5 * r * torch.clip((1.1 - 0.2 * self.curriculumupdate), 0.9, 1.1)
            #     self.curriculum_dof_pos_limits[i, 1] = m + 0.5 * r * torch.clip((1.1 - 0.2 * self.curriculumupdate), 0.9, 1.1)

            self.last_step_counter = self.common_step_counter


        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.



        if "escapetask" in self.reward_scales:
            self.reward_scales["escapetask"] = self.escapetask_init * (1 + 0.2 * self.curriculumupdate)
        if "success" in self.reward_scales:
            self.reward_scales["success"] = self.success_init * (1 + 0.2 * self.curriculumupdate)
        if "hasescaped" in self.reward_scales:
            self.reward_scales["hasescaped"] = self.escape_init * (1 + 0.5 * self.curriculumupdate)
        if "bodyreach" in self.reward_scales:
            self.reward_scales["bodyreach"] = self.bodyreach_init * (1 + 0.5 * self.curriculumupdate)            
        if "handreach" in self.reward_scales:
            self.reward_scales["handreach"] = self.hand_init * (1 + 0.5 * self.curriculumupdate)


        if self.curriculumupdate > 1.0:
             self.reward_scales["dof_pos_limits"] = self.dof_pos_init * 2
             self.reward_scales["torque_limits"]  = self.torque_init  * 2

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew


        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """

        ball_pos_in_camera_frame = self.ball_states[:, 0:3] - self.rigid_body_states[:, self.camera_index, 0:3]
        ball_pos_in_camera_frame = quat_rotate_inverse(self.rigid_body_states[:, self.camera_index, 3:7], ball_pos_in_camera_frame)
        horizontal_angle = torch.atan2(ball_pos_in_camera_frame[:, 1], 
                                    ball_pos_in_camera_frame[:, 0]) 
        vertical_angle = torch.atan2(ball_pos_in_camera_frame[:, 2], 
                                    ball_pos_in_camera_frame[:, 0])  
        
        half_h_fov = 80.2 * torch.pi / 180 / 2 
        half_v_fov = 64.6 * torch.pi / 180 / 2

        ball_in_camera_fov = torch.logical_and(
            torch.abs(horizontal_angle) < half_h_fov,
            torch.abs(vertical_angle) < half_v_fov
        ).view(-1, 1)
        ball_in_front = (ball_pos_in_camera_frame[:, 0] > 0).view(-1, 1)


        ball_visible = torch.logical_and(ball_in_camera_fov, ball_in_front)

        initial_vanish = (self.catchstep < self.startstep).view(-1, 1)
        end_target_local = quat_rotate_inverse(self.base_quat, self.ball_states[:,:3] - self.torso_pos) * initial_vanish

        flying = ((end_target_local[:,0] > 0.05) & (end_target_local[:,0] < 3.4) & (end_target_local[:,1] > -2.0) & (end_target_local[:,1] < 2.0) & (end_target_local[:,2] < 1.8) & (self.catchstep > 0.) & ((end_target_local[:,0] < self.ball_last[:,0]) | (self.ball_last[:,0] == 0.))).view(-1, 1)
        random_vanish = (self.catchstep > self.vanish_step).view(-1, 1)
        self.ball_last = end_target_local

        current_obs = torch.cat((   
                                    end_target_local,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.end_regions.unsqueeze(-1),
                                    quat_rotate_inverse(self.base_quat, self.end_target - self.torso_pos),
                                    quat_rotate_inverse(self.base_quat, self.ball_states[:,7:10]) * self.obs_scales.ball_vel,
                                    ),dim=-1)

        # add noise if needed
        current_actor_obs = torch.clone(current_obs[:, :self.num_one_step_obs])

        if self.add_noise:
            current_actor_obs = current_actor_obs + (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec[:(self.num_ballobs + 6 + 2 * self.num_dof + self.num_actions)]

            ## add ball noises ##
            current_actor_obs[:, :self.num_ballobs] =  current_actor_obs[:, :self.num_ballobs] * flying * random_vanish * ball_visible
        else:
            current_actor_obs[:, :self.num_ballobs] =  current_actor_obs[:, :self.num_ballobs] * flying

        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_obs:self.actor_obs_length], current_actor_obs), dim=-1)

        self.privileged_obs_buf = current_obs
        
        
    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """


        initial_vanish = (self.catchstep < self.startstep).view(-1, 1)
        end_target_local = quat_rotate_inverse(self.base_quat, self.ball_states[:,:3] - self.torso_pos) * initial_vanish

        current_obs = torch.cat((   
                                    end_target_local,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.end_regions.unsqueeze(-1),
                                    quat_rotate_inverse(self.base_quat, self.end_target - self.torso_pos),
                                    quat_rotate_inverse(self.base_quat, self.ball_states[:,7:10]) * self.obs_scales.ball_vel,
                                    ),dim=-1)



        return current_obs[env_ids]
    
        
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        start = time()
        print("*"*80)
        print("Start creating ground...")
        self._create_ground_plane()
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*"*80)
        self._create_envs()


    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.actor_handles[env_id]
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)

            for i in range(len(rigid_shape_props)):
                if self.cfg.domain_rand.randomize_friction:
                    rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_restitution:
                    rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]
                
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.curriculum_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)

            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

                m = (self.hard_dof_pos_limits[i, 0] + self.hard_dof_pos_limits[i, 1]) / 2
                r = self.hard_dof_pos_limits[i, 1] - self.hard_dof_pos_limits[i, 0]
                self.curriculum_dof_pos_limits[i, 0] = m - 0.5 * r * 1.1
                self.curriculum_dof_pos_limits[i, 1] = m + 0.5 * r * 1.1


        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id==0:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass
                print(f"Mass of body {i}: {p.mass} (before randomization)")
            print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
            
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = self.default_com + gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def refresh_actor_rigid_body_props(self, env_ids):
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload[env_ids] = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (len(env_ids), 1), device=self.device)
            
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement[env_ids] = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (len(env_ids), 3), device=self.device)
            
        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.actor_handles[env_id]
            rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            rigid_body_props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
            rigid_body_props[0].com = gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])
            
            if self.cfg.domain_rand.randomize_link_mass:
                rng = self.cfg.domain_rand.link_mass_range
                for i in range(1, len(rigid_body_props)):
                    scale = np.random.uniform(rng[0], rng[1])
                    rigid_body_props[i].mass = scale * self.default_rigid_body_mass[i]
            
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, rigid_body_props, recomputeInertia=True)



    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """        
        self._randomize_balls()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller


        actions_scaled = actions * self.cfg.control.action_scale
        self.joint_pos_target = self.default_dof_poses + actions_scaled
        
        self.joint_pos_target[self.catchstep > self.startstep] = self.init_dof_pos[self.catchstep > self.startstep]
    
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        torques = torques + self.actuation_offset + self.joint_injection
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)

        if self.cfg.domain_rand.continue_keep and torch.rand(1).item() > 0.2:
            self.dof_pos[env_ids] = self.dof_pos[torch.randint(0, self.num_envs, (len(env_ids),), device=self.dof_pos.device)]
        else:    
            if self.cfg.domain_rand.randomize_initial_joint_pos:
                init_dos_pos = self.standpos * torch_rand_float(self.cfg.domain_rand.initial_joint_pos_scale[0], self.cfg.domain_rand.initial_joint_pos_scale[1], (len(env_ids), self.num_dof), device=self.device)
                init_dos_pos += torch_rand_float(self.cfg.domain_rand.initial_joint_pos_offset[0], self.cfg.domain_rand.initial_joint_pos_offset[1], (len(env_ids), self.num_dof), device=self.device)
                self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
            else:
                self.dof_pos[env_ids] = self.standpos * torch.ones((len(env_ids), self.num_dof), device=self.device)

        self.init_dof_pos[env_ids] = self.dof_pos[env_ids].clone()


        self.dof_vel[env_ids] = 0.

        env_ids_int32 = torch.cat((2 * env_ids, 2 * env_ids + 1)).to(dtype=torch.int32)

        env_ids_int32 = 2 * env_ids.clone().to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


        
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # self.root_states[env_ids, 1:2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 1), device=self.device) # xy position within 1m of the center
        # self.root_states[env_ids, 2:3] += torch_rand_float(-0.1, 0.1, (len(env_ids), 1), device=self.device) # z position within 0.1m of the ground
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.3, 0.3, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        ball_ids = env_ids
        self.ball_states[ball_ids] = self.base_init_state

        ball_vel = self.assign_ball_states(ball_ids)

        self.ball_states[ball_ids, :3] = self.ball_start[ball_ids, :]

        self.ball_vel[ball_ids] = ball_vel[:,0]
        self.has_in_air[ball_ids] = False
        self.static_obs[ball_ids] *= 0. 
        self.stop_flag[ball_ids] = 0.
        self.success_flag[ball_ids] = 0.
    
        self.ball_last[ball_ids] = 0.
        self.vanish_step[ball_ids] = torch.randint(low=0, high=30, size=(len(ball_ids),), device=self.device)
        self.max_feetht[ball_ids] = 0.
        # move towards the robot
        self.ball_states[ball_ids, 7:13] *= 0. 
        self.ball_states[ball_ids, 7:10] = ball_vel
        
        all_states = torch.cat((self.root_states.unsqueeze(1),self.ball_states.unsqueeze(1)),dim = 1).view(-1, 13)
        env_ids_int32 = torch.cat((2 * env_ids, 2 * env_ids + 1)).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(all_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y

        all_states = torch.cat((self.root_states.unsqueeze(1),self.ball_states.unsqueeze(1)),dim = 1).view(-1, 13)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(all_states))

    def _randomize_balls(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        
        airforce = self.compute_drag_force()
        airforce += torch.empty_like(airforce).uniform_(-0.1, 0.1)
        force_tensor = torch.zeros([self.num_envs, self.num_bodies + 1, 3], device=self.device)
        force_tensor[:, -1, :3] = airforce
        force_tensor = gymtorch.unwrap_tensor(force_tensor)
        self.gym.apply_rigid_body_force_tensors(self.sim, force_tensor)

        if (self.common_step_counter % self.cfg.domain_rand.ball_interval == 0):
            max_vel = self.cfg.domain_rand.max_ball_vel
            self.ball_states[:, 7:10] += torch_rand_float(-max_vel, max_vel, (self.num_envs, 3), device=self.device) # lin vel x/y
            all_states = torch.cat((self.root_states.unsqueeze(1),self.ball_states.unsqueeze(1)),dim = 1).view(-1, 13)
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(all_states))


    def compute_drag_force(self, rho=1.225, drag_coeff=0.47, radius=0.1):
        velocity = self.ball_states[:, 7:10]
        cross_area = torch.pi * (radius ** 2)
        speed = torch.norm(velocity, dim=1, keepdim=True)  # Magnitude of velocity
        return -0.5 * rho * drag_coeff * cross_area * speed * velocity




    def assign_ball_states(self, ball_ids, g=9.81):

        dtype = torch.float
        device = self.ball_start.device

        self.catchstep[ball_ids] = 50 * torch.ones(len(ball_ids), dtype = torch.int, device = self.device)
                
        ball_start_local = torch.stack([
            2.0 * torch.rand(len(ball_ids), dtype=dtype, device=device) + 3.0,
            torch.rand(len(ball_ids), dtype=dtype, device=device) * (self.init_ranges[1] - self.init_ranges[0]) + self.init_ranges[0],
            torch.rand(len(ball_ids), dtype=dtype, device=device) * (self.init_ranges[3] - self.init_ranges[2]) + self.init_ranges[2]
        ], dim=1)

        ball_end_local = torch.stack([
            -0.1 * torch.rand(len(ball_ids), dtype=dtype, device=device) - 0.1,
            torch.rand(len(ball_ids), dtype=dtype, device=device) * (self.command_ranges[ball_ids, 1] - self.command_ranges[ball_ids, 0]) + self.command_ranges[ball_ids, 0],
            torch.rand(len(ball_ids), dtype=dtype, device=device) * (self.command_ranges[ball_ids, 3] - self.command_ranges[ball_ids, 2]) + self.command_ranges[ball_ids, 2]
        ], dim=1)

        # in world frame
        self.ball_start[ball_ids,:] = ball_start_local + self.env_origins[ball_ids]  # convert the local target cord in world frame 
        self.ball_start[ball_ids, 2] = ball_start_local[:, 2]

        self.ball_end[ball_ids,:] = ball_end_local + self.env_origins[ball_ids] # convert the local target cord in world frame 
        self.ball_end[ball_ids, 2] = ball_end_local[:, 2]

        hit_prop = (0.0 - ball_start_local[:,0:1]) / (ball_end_local[:,0:1] - ball_start_local[:,0:1])

        hit_foot_ids = ball_ids[self.end_regions[ball_ids] == 0]
        rand_mask = torch.rand(len(hit_foot_ids), device=self.rigid_body_states.device) < 0.5

        left_pos = self.rigid_body_states[hit_foot_ids, self.contact_feet_indices[0], :3].clone()
        right_pos = self.rigid_body_states[hit_foot_ids, self.contact_feet_indices[1], :3].clone()

        chosen_pos = torch.empty_like(left_pos)
        chosen_pos[rand_mask] = left_pos[rand_mask]
        chosen_pos[~rand_mask] = right_pos[~rand_mask]
        self.ball_end[hit_foot_ids, 1] = chosen_pos[:, 1]

        # Compute velocity
        delta_pos = self.ball_end[ball_ids,:] - self.ball_start[ball_ids,:]

        t_flight = 0.3 + 0.3 * torch.rand((1), dtype=dtype, device=device)

        if self.play:
            t_flight = 0.4 + 0.2 * torch.rand((1), dtype=dtype, device=device)

        self.end_target[ball_ids,:] = self.ball_start[ball_ids,:] + delta_pos * hit_prop

        ball_vel = torch.empty_like(delta_pos)

        ball_vel[:, 0:2] = delta_pos[:, 0:2] / t_flight
        ball_vel[:, 2] = (delta_pos[:, 2] + 0.5 * g * t_flight**2) / t_flight



        return ball_vel


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])\

        noise_vec = torch.zeros( self.num_ballobs + 6 + 2 * self.num_dof + self.num_actions, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:self.num_ballobs] = noise_scales.ball * noise_level
        noise_vec[self.num_ballobs:self.num_ballobs + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[self.num_ballobs + 3: self.num_ballobs + 6] = noise_scales.gravity * noise_level
        noise_vec[self.num_ballobs + 6:(self.num_ballobs + 6 + self.num_dof)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(self.num_ballobs + 6 + self.num_dof):(self.num_ballobs + 6 + 2 * self.num_dof)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(self.num_ballobs + 6 + 2 * self.num_dof):(self.num_ballobs + 6 + 2 * self.num_dof + self.num_actions)] = 0. # previous actions
        return noise_vec




    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        all_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, 2,13)
        self.root_states, self.ball_states = all_states[:, 0, :], all_states[:,1, :]

    

        all_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies + 1, 13)
        self.rigid_body_states = all_body_states[:, :-1, :]

        all_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, self.num_bodies + 1, 3) # shape: num_envs, num_bodies, xyz axis
        self.contact_forces = all_contact_forces[:, :-1, :]

        self.ball_contact_forces = all_contact_forces[:, -1:, :]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        
        self.left_feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.left_feet_indices, 0:3]
        self.right_feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.right_feet_indices, 0:3]


        # initialize some data used later on
        self.common_step_counter = 0
        self.last_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.first_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        # self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_lin_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,7:10])
        # self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,10:13])
        
        self.torso_pos = self.rigid_body_states[:, self.torso_index, 0:3]

        # self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.gravity_vec)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        

        self.ball_traj = [[], [], [], []]
        self.parabola = [[], [], [], []]

        self.end_target = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)

        six = self.num_envs // 2
        self.end_regions = torch.cat([
            torch.zeros(six, dtype=torch.long, device = self.device),
            torch.ones(six, dtype=torch.long, device = self.device),
        ])
    
        command_dict = class_to_dict(self.cfg.commands)

        # Initialize an empty tensor for command ranges
        num_envs = len(self.end_regions)
        self.command_ranges = torch.zeros((num_envs, 4), dtype=torch.float32, device=self.device)
        self.command_bound  = torch.zeros((num_envs, 4), dtype=torch.float32, device=self.device)
        self.init_ranges  = torch.zeros((4), dtype=torch.float32, device=self.device)
        # For each environment, set the appropriate ranges based on end_region
        for env_idx in range(num_envs):
            region = self.end_regions[env_idx].item()  # Get the region (0, 1, 2, or 3)
            region_key = f"ranges_{region}"
            
            # Get the ranges for this region
            region_ranges = command_dict[region_key]
            
            # Assign height and width ranges
            self.command_ranges[env_idx, 0] = region_ranges["width"][0]   # width_0
            self.command_ranges[env_idx, 1] = region_ranges["width"][1]   # width_1
            self.command_ranges[env_idx, 2] = region_ranges["height"][0]  # height_0
            self.command_ranges[env_idx, 3] = region_ranges["height"][1]  # height_1

            if self.play:
                self.command_ranges[env_idx, 0] = region_ranges["maxw"][0]   # width_0
                self.command_ranges[env_idx, 1] = region_ranges["maxw"][1]   # width_1
                self.command_ranges[env_idx, 2] = region_ranges["maxh"][0]  # height_0
                self.command_ranges[env_idx, 3] = region_ranges["maxh"][1]  # height_1



            self.command_bound[env_idx, 0] = region_ranges["maxw"][0]   # width_0
            self.command_bound[env_idx, 1] = region_ranges["maxw"][1]   # width_1
            self.command_bound[env_idx, 2] = region_ranges["maxh"][0]  # height_0
            self.command_bound[env_idx, 3] = region_ranges["maxh"][1]  # height_1

        self.init_ranges[0] = 3 * command_dict["ranges_0"]["maxw"][0]
        self.init_ranges[1] = 3 * command_dict["ranges_0"]["maxw"][1]
        self.init_ranges[2] = command_dict["ranges_0"]["maxh"][0]
        self.init_ranges[3] = command_dict["ranges_1"]["maxh"][1]



        self.catchstep = 50 * torch.ones(self.num_envs, dtype = torch.int, device= self.device)
        self.dist = 5 * torch.ones(self.num_envs, dtype = torch.float, device= self.device)
        self.max_feetht = torch.zeros(self.num_envs, dtype = torch.float, device= self.device)
        self.ball_vel = torch.zeros(self.num_envs, dtype = torch.float, device= self.device)
        self.has_in_air = torch.zeros(self.num_envs, dtype = torch.bool, device= self.device)
        self.static_obs = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)
        self.stop_flag = torch.zeros(self.num_envs, dtype = torch.float, device= self.device)
        self.success_flag = torch.zeros(self.num_envs, dtype = torch.float, device= self.device)
        self.ball_start = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)
        self.ball_end = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)    

        self.ball_last = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)   
        self.vanish_step = torch.randint(low=0, high=30, size=(self.num_envs,), device=self.device)
        
        self.sr = torch.zeros(self.num_envs, 3, dtype = torch.float, device= self.device)  
        self.startstep = 50 - random.randint(3, 10)

        for i in range(self.num_dof):
            name = self.dof_names[i]
            print(f"Joint {self.gym.find_actor_dof_index(self.envs[0], self.actor_handles[0], name, gymapi.IndexDomain.DOMAIN_ACTOR)}: {name}")
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.standpos = torch.tensor([self.cfg.init_state.init_pos], dtype=torch.float32, device=self.default_dof_pos.device)

        self.default_dof_poses = self.default_dof_pos.repeat(self.num_envs,1)
        self.init_dof_pos = self.default_dof_poses.clone()

        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_injection = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuation_offset = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
            self.joint_injection[:, self.curriculum_dof_indices] = 0.0
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
            self.actuation_offset[:, self.curriculum_dof_indices] = 0.0
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        
        #joint powers
        self.joint_powers = torch.zeros(self.num_envs, 100, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)


        # create mocap dataset
        self.init_base_pos_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.init_base_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.init_base_pos_xy[:] = self.base_init_state[:2] + self.env_origins[:, 0:2]
        self.init_base_quat[:] = self.base_init_state[3:7]



        multidataset, mapping = load_imitation_dataset(self.cfg.dataset.folder.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),self.cfg.dataset.joint_mapping.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
        self.motions = {}
        self.motion_ids = {}
        self.motion_time = {}
        self.motion_dict = {}
        for key in multidataset.keys():
            # Here 'key' will be the dataset's key name, and 'dataset' is the actual data

            # Initialize the MotionLib class for the given dataset
            self.motions[key] = MotionLib(multidataset[key], mapping, self.dof_names, self.keyframe_names,
                                        self.cfg.dataset.frame_rate, self.cfg.dataset.min_time, self.device,
                                        self.amp_obs_type)
            
    


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue

            if name == "escapetask":
                self.escapetask_init =  self.reward_scales["escapetask"]
            if name == "success":
                self.success_init = self.reward_scales["success"]
            if name == "hasescaped":
                self.escape_init = self.reward_scales["hasescaped"]
            if name == "handreach":
                self.hand_init = self.reward_scales["handreach"]
            if name == "bodyreach":
                self.bodyreach_init = self.reward_scales["bodyreach"]                

            if name == "torque_limits":
                self.torque_init = self.reward_scales["torque_limits"]
            if name == "dof_pos_limits":
                self.dof_pos_init = self.reward_scales["dof_pos_limits"]

            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.ball_gravity = self.cfg.env.ball_gravity
        self.num_ballobs = self.cfg.env.num_ballobs
        self.play = self.cfg.env.play
        ### load ball ###
        asset_options.disable_gravity = False
        ball_path = self.cfg.asset.ballfile.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        ball_root = os.path.dirname(ball_path)
        ball_file = os.path.basename(ball_path)
        ball_asset = self.gym.load_asset(self.sim, ball_root, ball_file, asset_options)

    

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dof = len(self.dof_names)

    
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        hand_names  = [s for s in body_names if self.cfg.asset.hand_name in s]

        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.ball_handles = []
        self.envs = []
        self.curriculumupdate = 0.
        self.task_scale = 0.
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-0.3, 0.3, (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            
            if i == 0:
                self.default_com = copy.deepcopy(body_props[0].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                    
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.actor_handles.append(actor_handle)

            ballpos = pos
            ballpos[:0] += torch_rand_float(0.5, 1.0, (1,1), device=self.device).squeeze(1)
            ballpos[:1] += torch_rand_float(-0.5, 0.5, (1,1), device=self.device).squeeze(1)
            ballpos[2] = 1.5
            
            start_pose.p = gymapi.Vec3(*ballpos)
            ball_handle = self.gym.create_actor(env_handle, ball_asset, start_pose, "ball", i, 0, 1)
            c = 0.5 + 0.5 * np.random.random(3)
            color = gymapi.Vec3(c[0], c[1], c[2])
            self.gym.set_rigid_body_color(env_handle, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            self.ball_handles.append(ball_handle)

            self.envs.append(env_handle)



        self.left_hip_joint_indices = torch.zeros(len(self.cfg.control.left_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.left_hip_joints)):
            self.left_hip_joint_indices[i] = self.dof_names.index(self.cfg.control.left_hip_joints[i])
            
        self.right_hip_joint_indices = torch.zeros(len(self.cfg.control.right_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.right_hip_joints)):
            self.right_hip_joint_indices[i] = self.dof_names.index(self.cfg.control.right_hip_joints[i])
            
        self.hip_joint_indices = torch.cat((self.left_hip_joint_indices, self.right_hip_joint_indices))
            
        knee_names = self.cfg.asset.knee_names
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.hand_indices = torch.zeros(len(hand_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hand_names)):
            self.hand_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], hand_names[i])

        self.camera_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "d435_link")

        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        left_feet_names = [s for s in body_names if self.cfg.asset.left_foot_name in s]
        right_feet_names = [s for s in body_names if self.cfg.asset.right_foot_name in s]
        
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        
        
        contact_feet_names = [s for s in body_names if self.cfg.asset.contact_foot_names in s]
        
        self.contact_feet_indices = torch.zeros(len(contact_feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(contact_feet_names)):
            self.contact_feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], contact_feet_names[i])


        head_names = [s for s in body_names if self.cfg.asset.head_name in s]
        self.head_indices = torch.zeros(len(head_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(head_names)):
            self.head_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], head_names[i])




        self.left_feet_indices = torch.zeros(len(left_feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_feet_names)):
            self.left_feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_feet_names[i])

        self.right_feet_indices = torch.zeros(len(right_feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_feet_names)):
            self.right_feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
            
        self.curriculum_dof_indices = torch.zeros(len(self.cfg.control.curriculum_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.curriculum_joints)):
            self.curriculum_dof_indices[i] = self.dof_names.index(self.cfg.control.curriculum_joints[i])
            
        self.left_leg_joint_indices = torch.zeros(len(self.cfg.control.left_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.left_leg_joints)):
            self.left_leg_joint_indices[i] = self.dof_names.index(self.cfg.control.left_leg_joints[i])
            
        self.right_leg_joint_indices = torch.zeros(len(self.cfg.control.right_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.right_leg_joints)):
            self.right_leg_joint_indices[i] = self.dof_names.index(self.cfg.control.right_leg_joints[i])
            
        self.leg_joint_indices = torch.cat((self.left_leg_joint_indices, self.right_leg_joint_indices))
            
        self.left_arm_joint_indices = torch.zeros(len(self.cfg.control.left_arm_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.left_arm_joints)):
            self.left_arm_joint_indices[i] = self.dof_names.index(self.cfg.control.left_arm_joints[i])
            
        self.right_arm_joint_indices = torch.zeros(len(self.cfg.control.right_arm_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.right_arm_joints)):
            self.right_arm_joint_indices[i] = self.dof_names.index(self.cfg.control.right_arm_joints[i])
            
        self.elbow_joint_indices = torch.zeros(len(self.cfg.control.elbow_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.elbow_joints)):
            self.elbow_joint_indices[i] = self.dof_names.index(self.cfg.control.elbow_joints[i])

        self.wrist_joint_indices = torch.zeros(len(self.cfg.control.wrist_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.wrist_joints)):
            self.wrist_joint_indices[i] = self.dof_names.index(self.cfg.control.wrist_joints[i])



        self.arm_joint_indices = torch.cat((self.left_arm_joint_indices, self.right_arm_joint_indices))
            
        self.waist_joint_indices = torch.zeros(len(self.cfg.asset.waist_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.waist_joints)):
            self.waist_joint_indices[i] = self.dof_names.index(self.cfg.asset.waist_joints[i])
            
        self.ankle_joint_indices = torch.zeros(len(self.cfg.asset.ankle_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.ankle_joints)):
            self.ankle_joint_indices[i] = self.dof_names.index(self.cfg.asset.ankle_joints[i])


        self.upper_body_joint_indices = torch.cat([self.elbow_joint_indices, self.wrist_joint_indices])

        self.upper_body_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.control.upper_body_link)
        self.torso_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.control.torso_link)
            
        self.keyframe_names = [s for s in body_names if self.cfg.asset.keyframe_name in s]
        self.keyframe_indices = torch.zeros(len(self.keyframe_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(self.keyframe_names):
            self.keyframe_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)



    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """

        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(-num_rows//2, num_rows//2), torch.arange(-num_cols//2, num_cols//2))
    
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.ball_interval = np.ceil(self.cfg.domain_rand.ball_interval_s / self.dt)

        
    def _get_base_heights(self, env_ids=None):

        return self.root_states[:, 2].clone()


    #------------ reward functions----------------



    def _reward_escapetask(self):


        jump_dist = torch.min(self.rigid_body_states[:,self.contact_feet_indices, 2], dim = -1).values  - self.end_target[:,2]
        squat_dist = self.end_target[:, 2] - self.rigid_body_states[:, self.head_indices[0], 2]

        self.dist[self.end_regions == 0] = jump_dist[self.end_regions == 0]
        self.dist[self.end_regions == 1] = squat_dist[self.end_regions == 1]

        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)

        vel_sigma = torch.zeros(self.num_envs, dtype = torch.float, device = self.device)
        vel_sigma[self.end_regions == 0] = 1 + 10.0 * torch.clip(torch.min(self.rigid_body_states[:, self.knee_indices, 9], dim = -1).values, 0.0, 5.0)[self.end_regions == 0]
        vel_sigma[self.end_regions == 1] = 1 - 10.0 * torch.clip(self.rigid_body_states[:, self.head_indices[0], 9], -2.0, 0.0)[self.end_regions == 1]

        vel_sigma[behind] = 2.0

        taskrew = torch.exp(self.cfg.rewards.escape_sigma * (torch.clip(self.dist, min = -0.3, max = 0.3) - self.cfg.rewards.escape_th))
        taskrew = torch.clip(taskrew, min = 0.0, max = 1.0) * vel_sigma

        taskrew[behind] = torch.exp(torch.abs(self.root_states[behind, 2] - self.cfg.rewards.base_height_target) * - 20) * vel_sigma[behind]
        
        left_quat = self.rigid_body_states[:,  self.contact_feet_indices[0], 3:7]  
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.rigid_body_states[:,  self.contact_feet_indices[1], 3:7]  
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        feet_orientation = torch.sum(torch.square(left_gravity[:, :2]), dim=1) + torch.sum(torch.square(right_gravity[:, :2]), dim=1)

        return taskrew * (1 - 3 * torch.clip(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1), 0., 1.0)) * (1 - 3 * torch.clip(feet_orientation, 0., 1.0))

    def _reward_success(self):
        return self.success_flag
    

    def _reward_hasescaped(self):

        escaped = (self.ball_states[:,7] - self.ball_vel < 2.0) & ((self.ball_states[:, 0] - self.env_origins[:, 0]) < 0.0)
        stop = (self.ball_states[:,7] - self.ball_vel > 2.0)
        behind = ((self.ball_states[:, 0] - self.env_origins[:, 0]) < 0.0)
        dist = self.dist > 0.1
        stopped_ids = (escaped | behind| stop ).nonzero(as_tuple=False).flatten()
        success_ids = ((self.stop_flag == 0) & escaped & dist) .nonzero(as_tuple=False).flatten()
        rew_stop = 1.0 * (self.stop_flag == 0) * escaped
        self.success_flag[success_ids] = 1.0
        self.stop_flag[stopped_ids] = 1.0

        return rew_stop

    def _reward_feetorientaion(self):

        left_quat = self.rigid_body_states[:,  self.contact_feet_indices[0], 3:7]  
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.rigid_body_states[:,  self.contact_feet_indices[1], 3:7]  
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        feet_orientation = torch.sum(torch.square(left_gravity[:, :2]), dim=1) + torch.sum(torch.square(right_gravity[:, :2]), dim=1)
        return torch.exp(feet_orientation * -10)



    def _reward_successland(self):

        foot_contact_forces_z = self.contact_forces[:, self.contact_feet_indices, 2]
        
        jump = torch.min(self.rigid_body_states[:, self.contact_feet_indices, 2], dim = -1).values > 0.3
        self.has_in_air = torch.logical_or(self.has_in_air, jump)
  
        has_contact = (foot_contact_forces_z[:, 0] > 1.) & (foot_contact_forces_z[:, 1] > 1.)

        one_feet_contact = (((foot_contact_forces_z[:, 0] >  1.) & (foot_contact_forces_z[:, 1] < 1.)) | ((foot_contact_forces_z[:, 0] <  1.) & (foot_contact_forces_z[:, 1] > 1.))) & (self.has_in_air)

        successful_landings = torch.logical_and(has_contact, self.has_in_air)
    
        air_reward = self.has_in_air.float()  
        landing_reward = successful_landings.float() * 5.0  

        one_feet_punish = one_feet_contact.float() * -2.0

        jump_ids = (self.end_regions == 0)
        
        return (air_reward + landing_reward + one_feet_punish) * jump_ids # * torch.exp(torch.norm(self.projected_gravity[:, :1], dim=1) * -3)  

    def _reward_feet_slippage(self):
        
        # behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)
        foot_vel = self.rigid_body_states[:,  self.contact_feet_indices, 7:10]  
        contactvel = torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.contact_forces[:, self.contact_feet_indices, :], dim=-1) > 1.), dim=1)
        return torch.exp(contactvel * -10)

    def _reward_penalize_sharpcontact(self):

        return (torch.mean(torch.norm(self.contact_forces[:, self.contact_feet_indices, :], dim=-1), dim = -1) >  self.cfg.rewards.max_contact_force) * 1.0 

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)


    def _reward_postorientation(self):
        
        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)

        return torch.exp(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * -3) * behind

    def _reward_postheight(self):
        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)
        base_height = self.root_states[:, 2]
        return torch.exp(torch.abs(base_height - self.cfg.rewards.base_height_target) * - 20) * behind

    def _reward_postangvel(self):
        
        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)

        return torch.exp(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -3)* behind

    def _reward_postlinvel(self):
        
        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)

        return torch.exp(torch.sum(torch.square(self.base_lin_vel[:, 0:1]), dim=1) * -3)* behind

    def _reward_postupperdofpos(self):

        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)
        mse = torch.sum(torch.square(self.dof_pos[:, self.upper_body_joint_indices] - self.default_dof_pos[:, self.upper_body_joint_indices]), dim=-1)
        reward = torch.exp(mse * -1) 
        return reward * behind

    def _reward_postwaistdofpos(self):

        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)
        mse = torch.sum(torch.square(self.dof_pos[:, self.waist_joint_indices] - self.default_dof_pos[:, self.waist_joint_indices]), dim=-1)
        reward = torch.exp(-3 * mse) 
        return reward * behind

    def _reward_stayonline(self):
        distance = torch.clip(torch.abs(self.torso_pos[:,0] - self.env_origins[:, 0]), 0.2, 1.2) - 0.2   
        return distance

    def _reward_noretreat(self):
        return -1 * torch.clip(self.base_lin_vel[:, 0], -1.0, 0.0)  
    

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_yaw(self):
        # Penalize yaw
        return torch.abs(self.yaw) > 0.5

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)


    
    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques / self.p_gains.unsqueeze(0)), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _reward_kneeheight(self):
        return torch.min(self.rigid_body_states[:, self.knee_indices, 2], dim = -1).values < 0.15

    def _reward_landvel(self):
        return  torch.min(self.rigid_body_states[:, self.contact_feet_indices, 9], dim = -1).values < -2.0

