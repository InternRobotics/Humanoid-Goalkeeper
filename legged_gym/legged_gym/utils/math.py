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

import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower


import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple
import math

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower


# @torch.jit.script
def normalize(x, eps=1e-9):
    x_norm =  x.norm(p=2, dim=-1).clip(min=eps, max=None)
    return x / x_norm.unsqueeze(-1)

# @torch.jit.script
def copysign(a, b):
    a = torch.zeros_like(b) + a
    return torch.abs(a) * torch.sign(b)

# @torch.jit.script
def quat_conjugate(x):
    return torch.cat([-x[..., :3], x[..., 3:]], dim=-1)

# @torch.jit.script
def quat_apply(a, b):
    if a.shape[1] == 1 and len(a.shape) == 3:
        a = a.clone().repeat(1, b.shape[1], 1)
    xyz = a[..., :3]
    t = xyz.cross(b, dim=-1) * 2.0
    return (b + a[..., 3:] * t + xyz.cross(t, dim=-1))

# @torch.jit.script
def quat_mul(a, b):
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([x, y, z, w], dim=-1)

# @torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

# @torch.jit.script
def quat_to_euler_xyz(q):
    q = normalize(q)
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * \
                q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(math.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * \
                q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack((roll, pitch, yaw), dim=-1)

# @torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone()
    quat_yaw[..., 0:2] = 0.0
    quat_yaw = normalize(quat_yaw)
    if quat.shape[1] ==1 and len(quat.shape) == 3:
        quat_yaw = quat_yaw.repeat(1, vec.shape[1], 1)

    return quat_apply(quat_yaw, vec)

# @torch.jit.script
def quat_apply_yaw_inverse(quat, vec):
    quat_yaw = quat_conjugate(quat.clone())
    quat_yaw[..., 0:2] = 0.0
    quat_yaw = normalize(quat_yaw)
    if quat.shape[1] ==1 and len(quat.shape) == 3:
        quat_yaw = quat_yaw.repeat(1, vec.shape[1], 1)
    return quat_apply(quat_yaw, vec)

# @torch.jit.script
def quat_mul_yaw(a, b):
    xyz = quat_to_euler_xyz(a) + quat_to_euler_xyz(b)
    xyz_yaw = torch.cat((b[..., 0:2], xyz[..., 2:3]), dim=-1)
    return euler_xyz_to_quat(xyz_yaw)

# @torch.jit.script
def quat_mul_yaw_inverse(a, b):
    xyz = quat_to_euler_xyz(b) - quat_to_euler_xyz(a)
    xyz_yaw = torch.cat((b[..., 0:2], xyz[..., 2:3]), dim=-1)
    return euler_xyz_to_quat(xyz_yaw)

# @torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*math.pi
    angles -= 2*math.pi * (angles > math.pi)
    return angles

# @torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

# @torch.jit.script
def quat_rotate(q, v):
    q_expand = q + torch.zeros_like(v[..., 0:1])
    q_w = q_expand[..., 3:4]
    q_vec = q_expand[..., 0:3]
    a = v * (2.0 * q_w.pow(2) - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    dot = torch.sum(q_vec * v, dim=-1, keepdim=True)
    c = q_vec * dot * 2.0
    return a + b + c

# @torch.jit.script
def quat_rotate_inverse(q, v):
    q_expand = q + torch.zeros_like(v[..., 0:1])
    q_w = q_expand[..., 3:4]
    q_vec = q_expand[..., 0:3]
    a = v * (2.0 * q_w.pow(2) - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    dot = torch.sum(q_vec * v, dim=-1, keepdim=True)
    c = q_vec * dot * 2.0
    return a - b + c

# @torch.jit.script
def quat_to_angle_axis(q):
    # computes axis-angle representation from quaternion q
    # q must be normalized
    q = normalize(q)
    sin_theta = torch.sqrt(1 - q[..., 3] * q[..., 3])
    angle = 2 * torch.acos(q[..., 3])
    angle = normalize_angle(angle)
    axis = q[..., 0:3] / sin_theta[..., None]

    mask = torch.abs(sin_theta) > 1e-5
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    axis = torch.where(mask[..., None], axis, default_axis)
    return wrap_to_pi(angle), axis

# @torch.jit.script
def angle_axis_to_quat(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([xyz, w], dim=-1))

# @torch.jit.script
def euler_xyz_to_quat(xyz):
    roll = xyz[..., 0]
    pitch = xyz[..., 1]
    yaw = xyz[..., 2]
    
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    q = torch.stack([qx, qy, qz, qw], dim=-1)
    return normalize(q)

# @torch.jit.script
def quat_to_rot6d(q):
    q = normalize(q)
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan

# @torch.jit.script
def quat_error(a, b):
    r = quat_mul(a, quat_conjugate(b))
    return r[..., 0:3] * torch.sign(r[..., 3]).unsqueeze(-1)

# @torch.jit.script
def heading(q):
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)
    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading

# @torch.jit.script
def heading_quat(q):
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1
    return angle_axis_to_quat(heading(q), axis)

# @torch.jit.script
def heading_quat_conjugate(q):
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1
    return angle_axis_to_quat(-heading(q), axis)

# @torch.jit.script
def remove_heading_quat(q):
    heading_q = heading_quat_conjugate(q)
    return quat_mul(heading_q, q)

# @torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower