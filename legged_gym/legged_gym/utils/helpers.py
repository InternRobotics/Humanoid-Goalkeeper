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

import os
import re
import copy
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
import torch
import torch.nn.functional as F
import onnxruntime as ort


from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    """Return path to a checkpoint under the given root.

    Behavior:
    - If root is not a directory, try to resolve it by matching the first 4 chars
      of a directory name in the parent folder (kept for backward compatibility).
    - If checkpoint == -1, pick the latest available model file under root.
      Preference order:
        1) Files that contain `model_name_include` if provided
        2) Any .pt/.pth files
      Sorting tries to use numeric checkpoint in the filename if present, then mtime.
    - If checkpoint >= 0, expect a file named `model_{checkpoint}.pt` unless a full
      filename (ending with .pt/.pth/.onnx) is provided.
    """

    # Resolve root by prefix match if a path-like prefix was provided instead of a dir
    if not os.path.isdir(root):  # use first 4 chars to match the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        if os.path.isdir(model_parent):
            model_names = [
                name for name in os.listdir(model_parent)
                if os.path.isdir(os.path.join(model_parent, name))
            ]
            prefix = model_name_cand[:4]
            for name in model_names:
                if name.startswith(prefix):
                    root = os.path.join(model_parent, name)
                    break

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Checkpoint root directory not found or invalid: {root}")

    def is_checkpoint_file(fname: str) -> bool:
        return fname.lower().endswith((".pt", ".pth")) and os.path.isfile(os.path.join(root, fname))

    def extract_ckpt_num(fname: str) -> int:
        # Try to extract the last integer in the filename for sensible sorting
        m = re.findall(r"(\d+)", fname)
        return int(m[-1]) if m else -1

    if checkpoint == -1:
        files = os.listdir(root)
        # Primary filter: match model_name_include if provided
        models = []
        if model_name_include:
            models = [f for f in files if model_name_include in f and is_checkpoint_file(f)]
        # Fallback: any .pt/.pth file
        if not models:
            models = [f for f in files if is_checkpoint_file(f)]

        if not models:
            available = ", ".join(sorted(files))
            hint = f"containing '{model_name_include}' " if model_name_include else ""
            raise FileNotFoundError(
                f"No checkpoint files {hint}found in '{root}'. Available: [{available}]"
            )

        # Sort by (numeric checkpoint, mtime) and take the last as the latest
        models.sort(key=lambda f: (extract_ckpt_num(f), os.path.getmtime(os.path.join(root, f))))
        model = models[-1]
    else:
        # Allow passing a full filename as checkpoint
        if isinstance(checkpoint, str) and checkpoint.lower().endswith((".pt", ".pth", ".onnx")):
            model = checkpoint
        else:
            model = f"model_{checkpoint}.pt"

        if not os.path.isfile(os.path.join(root, model)):
            raise FileNotFoundError(f"Specified checkpoint not found: {os.path.join(root, model)}")

    load_path = os.path.join(root, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "29", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        {"name": "--exptid", "type": str, "help": "exptid"},
        {"name": "--resumeid", "type": str, "help": "exptid"},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    # args.sim_device_id = args.compute_device_id
    args.sim_device = args.rl_device
    # if args.sim_device=='cuda':
    #     args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path, policy_name):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f'{policy_name}.pt')
    model = PolicyOnnx(actor_critic).to('cpu')
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)
        
def load_onnx_policy(path):
    model = ort.InferenceSession(path)
    def run_inference(input_tensor):
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device="cuda:0")
    return run_inference

def export_jit_to_onnx(jit_model, path, dummy_input):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.onnx.export(
        jit_model,                  # JIT 模型
        dummy_input,                # 示例输入张量
        path,                       # 输出文件路径
        export_params=True,         # 导出训练好的参数
        opset_version=11,           # ONNX opset version（推荐 11 或更高版本）
        do_constant_folding=True,   # 是否启用常量折叠优化
        input_names=['input'],      # 输入节点名称
        output_names=['output'],    # 输出节点名称
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 动态轴支持
    )
    print(f"Exported JIT model to ONNX at: {path}")

    
class PolicyOnnx(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.history_encoder = copy.deepcopy(actor_critic.history_encoder)
        self.ball_estimator = copy.deepcopy(actor_critic.ball_estimator)
        self.region_estimator = copy.deepcopy(actor_critic.region_estimator)
        self.history_length = actor_critic.actor_history_length
        self.num_one_step_obs = actor_critic.num_one_step_obs
        self.num_proprioceptive_obs = self.history_length * self.num_one_step_obs
        self.new_obs_start = self.num_one_step_obs * (self.history_length - 1)

    def forward(self, x):

        history_latent = self.history_encoder(x)
        estimate_ball = self.ball_estimator(x)
        estimate_region = self.region_estimator(x)
        estimate_region = torch.argmax(estimate_region, dim=-1, keepdim=True)
        actor_input = torch.cat((x[:,-self.num_one_step_obs:], history_latent, estimate_ball, estimate_region), dim=-1)

        return self.actor(actor_input)

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.num_proprioceptive_obs)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=True,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )    


    