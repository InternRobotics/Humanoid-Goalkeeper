from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G129Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 8192
        num_actor_history = 10
        
        
        num_actions = 29 # number of actuators on robot
        num_dofs = 29
        num_ballobs = 3
        num_one_step_observations = 6 + num_ballobs + num_dofs * 2 + num_actions
        num_privileged_obs = 6 + num_ballobs + num_dofs * 2 + num_actions  + 3 + 1 + 6

        num_observations = num_actor_history * num_one_step_observations

        env_spacing = 5.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 3 # episode length in seconds
        ball_gravity = True
        play = False
        
    class commands:

        class ranges_0:
            height = [0.05, 0.07] 
            width = [-0.05, 0.05]

            maxh = [0.05, 0.07] 
            maxw = [-0.05, 0.05]
        
        class ranges_1:
            height = [0.7, 1.2] 
            width = [-0.05, 0.05]

            maxh = [0.4, 1.2] 
            maxw = [-0.05, 0.05]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.73] # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            
            'left_hip_pitch_joint': -0.1, # in use
            'left_hip_roll_joint': 0.2, # in use
            'left_hip_yaw_joint': 0.0, # in use
            'left_knee_joint': 0.3, # in use
            'left_ankle_pitch_joint': -0.2, # in use
            'left_ankle_roll_joint': -0.2,

            'right_hip_pitch_joint': -0.1, # in use
            'right_hip_roll_joint': -0.2, # in use
            'right_hip_yaw_joint': 0.0, # in use
            'right_knee_joint': 0.3, # in use
            'right_ankle_pitch_joint': -0.2, # in use
            'right_ankle_roll_joint': 0.2,

            'waist_yaw_joint': 0.0, # in use
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,


            'left_shoulder_pitch_joint': 0.0, # in use
            'left_shoulder_roll_joint': 0.5, # in use
            'left_shoulder_yaw_joint': 0.0, # in use
            'left_elbow_joint': 1.2, # in use
            'left_wrist_roll_joint': 0.0, # in use
            'left_wrist_pitch_joint':0.0, 
            'left_wrist_yaw_joint':0.0,


            'right_shoulder_pitch_joint': 0.0, # in use
            'right_shoulder_roll_joint': -0.5,  # in use
            'right_shoulder_yaw_joint': 0.0, # in use
            'right_elbow_joint': 1.2,  # in use
            'right_wrist_roll_joint': 0.0, # in use
            'right_wrist_pitch_joint':0.0, 
            'right_wrist_yaw_joint':0.0,
            }

        init_pos = [-0.34930936, -0.03763366, -0.22198406,  0.93093884, -0.50943524, -0.08583859,
            0.13749947, -0.44516975, -0.06791031,  0.11570476, -0.17351833,  0.34241587,
            -0.00869134,  0.00670955,  0.01293622,  0.00395479,  0.49003497, -0.00168978,
            1.2062242,  -0.01060604,  0.00490874, -0.00869134,  0.00319979, -0.4975251,
            -0.00450607,  1.20307243,  0.00536893,  0.0053766,   0.00324437]
        

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 150,
                     'knee': 300,
                     'ankle': 40,
                     'shoulder': 150,
                     'elbow': 150,
                     'waist': 150,
                     'wrist': 20,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'shoulder': 2,
                     'elbow': 2,
                     'waist': 2,
                     'wrist': 0.5,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        curriculum_joints = ['waist_yaw_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint']
        left_leg_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint']
        right_leg_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']
        knee_joints = ['left_knee_joint', 'right_knee_joint']
        left_arm_joints = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint']
        right_arm_joints = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']

        elbow_joints = ['left_elbow_joint', 'right_elbow_joint']

        wrist_joints = ['left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']


        upper_body_link = "pelvis"  # "torso_link"
        torso_link = "torso_link"

        left_hip_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint']
        right_hip_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint']


    class terrain:
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

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

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/g1_29.urdf'
        ballfile = '{LEGGED_GYM_ROOT_DIR}/resources/gymassets/urdf/ball.urdf'
        name = "g1"
        foot_name = "ankle_pitch"
        left_foot_name = "left_foot"
        right_foot_name = "right_foot"

        contact_foot_names = "ankle_roll_link"

        hand_name = "hand"
        penalize_contacts_on = ["hip","pelvis", "hand"]
        terminate_after_contacts_on = []

        waist_joints = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
        ankle_joints = [ "left_ankle_pitch_joint", "right_ankle_pitch_joint"]
        imu_link = "imu_link"
        knee_names = ["left_knee_link", "right_knee_link"]
        head_name = "mid360_link"
        keyframe_name = "keyframe"

        disable_gravity = False
        collapse_fixed_joints = False # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False

        density = 0.001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01
        thickness = 0.01
    class domain_rand(LeggedRobotCfg.domain_rand):
        
        randomize_joint_injection = True
        joint_injection_range = [-0.01, 0.01]
        
        randomize_actuation_offset = True
        actuation_offset_range = [-0.01, 0.01]

        randomize_payload_mass = True
        payload_mass_range = [-5, 10]

        randomize_com_displacement = True
        com_displacement_range = [-0.1, 0.1]

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        
        randomize_friction = True
        friction_range = [0.1, 2.0]
        
        randomize_restitution = True  
        restitution_range = [0.0, 1.0]
        
        randomize_kp = True
        kp_range = [0.6, 1.2]
        
        randomize_kd = True
        kd_range = [0.8, 1.2]
        continue_keep = True
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.5, 1.5]
        initial_joint_pos_offset = [-0.1, 0.1]
        
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.5

        ball_interval_s = 0.5
        max_ball_vel = 0.5

        delay = True

        
    class rewards:
        class scales:
            
            # task rewards
            escapetask = 10.0
            success = 5.0
            hasescaped = 100.0

            # move rewards
            stayonline = -3.0
            noretreat = -3.0
            yaw = -1.0

            # feet rewards
            successland = 10.0
            feetorientaion = 10.0
            penalize_sharpcontact = -100.
            feet_slippage = 3.0


            # post rewards
            postorientation = 3.0
            postangvel = 3.0

            # reg rewards
            ang_vel_xy = -0.1
            dof_acc = -2.5e-7
            smoothness = -0.1
            collision = -100.

            torques = -1e-5 # -1e-4
            dof_vel = -1e-3

            dof_pos_limits = -3.0 #-2.0
            dof_vel_limits = -3.0 #0.0
            torque_limits = -5.0 #0.0
            kneeheight = -1.0
            landvel = -10.0
            
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)

   
        target_dof_pos_sigma = -20
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        escape_sigma = 3.0
        escape_th = 0.15

        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.95
        base_height_target = 0.73
        max_contact_force = 1000. # forces above this value are penalized
        clearance_height_target = -0.66
        least_feet_distance = 0.18
        least_feet_distance_lateral = 0.18
        least_knee_distance_lateral = 0.18

    class dataset:
        folder = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/escape"
        joint_mapping = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/escape/joint_id.txt"
        frame_rate = 30
        min_time = 0.1 # sec

    class amp:

        obs_type = 'dof'
        num_obs = 29 * 2  # (old and new)
        amp_coef = 0.5
        num_steps = 2

class G129CfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100 # per iteration
        max_iterations = 200000 # number of policy updates

        # logging
        save_interval = 200 # check for potential saves every this many iterations
        run_name = 'HIM'
        experiment_name = 'g1'
        wandb_project = "escape"
        logger = 'wandb'
        
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
    
    amp = G129Cfg.amp