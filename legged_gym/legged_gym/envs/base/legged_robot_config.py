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

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 6144

        num_actor_history = 10
        
        
        num_actions = 29 # number of actuators on robot
        num_dofs = 29
        num_ballobs = 3
        num_one_step_observations = 6 + num_ballobs + num_dofs * 2 + num_actions
        num_privileged_obs = 6 + num_ballobs + num_dofs * 2 + num_actions  + 3 + 1 + 6 + 6 + 1

        num_observations = num_actor_history * num_one_step_observations

        env_spacing = 5.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 3 # episode length in seconds
        ball_gravity = True
        play = False


    class terrain:
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.


    
    class commands:
        class ranges_0:
            height = [0.4, 1.2] 
            width =  [0.2, 1.2]

            maxh = [0.3, 1.5]
            maxw = [0.2, 1.8]

        class ranges_1:
            height = [0.4, 1.2] 
            width = [-1.2, -0.2]

            maxh = [0.3, 1.5] 
            maxw = [-1.8, -0.2]

        class ranges_2:
            height = [1.2, 1.6] 
            width = [0, 1.0]

            maxh = [1.2, 1.8] 
            maxw = [0, 1.5]
        
        class ranges_3:
            height = [1.2, 1.6] 
            width = [-1.0, 0.0]

            maxh = [1.2, 1.8] 
            maxw = [-1.5, 0.0]

        class ranges_4:
            height = [0.1, 0.3] 
            width = [0.2, 1.2]

            maxh = [0.1, 0.3]
            maxw = [0.2, 1.8]
        
        class ranges_5:
            height = [0.1, 0.3] 
            width = [-1.2, -0.2]

            maxh = [0.1, 0.3]
            maxw = [-1.8, -0.2]



    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0
        curriculum_joints = []

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        
        randomize_joint_injection = False
        joint_injection_range = [-0.1, 0.1]
        
        randomize_actuation_offset = False
        actuation_offset_range = [-0.1, 0.1]

        randomize_payload_mass = False
        payload_mass_range = [-5, 10]

        randomize_com_displacement = False
        com_displacement_range = [-0.1, 0.1]

        randomize_link_mass = False
        link_mass_range = [0.7, 1.3]
        
        randomize_friction = False
        friction_range = [0.1, 1.25]
        
        randomize_restitution = False
        restitution_range = [0.1, 1.0]
        
        randomize_kp = False
        kp_range = [0.8, 1.2]
        
        randomize_kd = False
        kd_range = [0.8, 1.2]
        
        randomize_initial_joint_pos = False
        initial_joint_pos_scale = [0.5, 1.5]
        initial_joint_pos_offset = [-0.1, 0.1]
        
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 1.

        delay = False

    class rewards:
        class scales:
            
            # task rewards
            eereach = 10.0
            success = 5.0
            stopball = 100.0

            # move rewards
            stayonline = -2.0
            noretreat = -2.0

            # feet rewards
            successland = 4.0
            feetorientaion = 3.0
            penalize_sharpcontact = -100.
            penalize_kneeheight = -100.
            feet_slippage = 3.0

            # post rewards
            postorientation = 3.0
            postangvel = 3.0
            postupperdofpos = 1.0
            postwaistdofpos = 1.0
            postlinvel = 1.0


            # reg rewards
            ang_vel_xy = -0.1
            dof_acc = -2.5e-7
            smoothness = -0.1

            torques = -1e-5
            dof_vel = -5e-4

            dof_pos_limits = -3.0
            dof_vel_limits = -2.0
            torque_limits = -3.0

            deviation_waist_pitch_joint = -0.001


        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)

        catch_th = 0.5
        handheight_th = 1.0
        reach_th = 0.2
        strict_th = 0.15

        target_dof_pos_sigma = -20
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        catch_sigma = 5.0

        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.95
        max_contact_force = 1000. # forces above this value are penalized


    class dataset:
        folder = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper"
        joint_mapping = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper/joint_id.txt"
        frame_rate = 30
        min_time = 0.1 # sec

    class amp:

        obs_type = 'dof'
        num_obs = 29 * 2  # (old and new)
        amp_coef = 0.4
        num_steps = 2

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            ball_vel = 0.2
            ball_pos = 0.3
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            ball = 0.08
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'HIMOnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 256]
        critic_hidden_dims = [512, 256, 256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100 # per iteration
        max_iterations = 200000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt