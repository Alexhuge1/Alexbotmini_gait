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
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

# based on the original code of the humanoidgym humanoid environment
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class alexbotminiCfg(LeggedRobotCfg):
    """
    Configuration class for the alexbotmini humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 73
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 1024
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/alexbotmini/urdf/alexbotmini.urdf'

        name = "alexbotmini"
        foot_name = "6"
        knee_name = "4"

        terminate_after_contacts_on = ['base_link','rightlink2','leftlink2','rightlink1','leftlink1']
        penalize_contacts_on = ['base_link','rightlink2','leftlink2','rightlink1','leftlink1']
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.6    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.73]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'leftjoint1': -0.174,
            'leftjoint2': 0.,
            'leftjoint3': 0.,
            'leftjoint4': 0.314,
            'leftjoint5': 0.14,
            'leftjoint6': 0.,
            'rightjoint1': 0.174,
            'rightjoint2': 0.,
            'rightjoint3': 0.,
            'rightjoint4': -0.314,
            'rightjoint5': -0.14,
            'rightjoint6': 0.,
        }

        # default_joint_angles = {  # = target angles [rad] when action = 0.0
        #     'leftjoint1': -0.3,
        #     'leftjoint2': 0.,
        #     'leftjoint3': 0.,
        #     'leftjoint4': 0.8,
        #     'leftjoint5': 0.5,
        #     'leftjoint6': 0.,
        #     'rightjoint1': 0.3,
        #     'rightjoint2': 0.,
        #     'rightjoint3': 0.,
        #     'rightjoint4': -0.8,
        #     'rightjoint5': -0.5,
        #     'rightjoint6': 0.,
        # }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        # stiffness = {'1': 180.0, '2': 120.0, '3': 120.0, '4': 180.0, '5': 45 , '6': 45}
        # damping = {'1': 10, '2': 8, '3': 8.0, '4': 10, '5': 2.5 , '6' : 2.5}
        stiffness = {'1': 180*0.4, '2': 200*0.4, '3': 120*0.4, '4': 180*0.4, '5': 120*0.4 , '6': 120*0.4}
        damping = {'1': 10*0.8, '2': 8*0.8, '3': 8.0*0.8, '4': 10*0.8, '5': 6*0.8 , '6' : 6*0.8}
        # kps = np.array([180, 200, 120, 180, 120, 120, 180, 200, 120, 180, 120, 120], dtype=np.double)*0.4
        # kds = np.array([ 10, 8, 8, 10, 6, 6, 10, 8, 8, 10, 6, 6,], dtype=np.double)*0.8
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1  
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-2, 2]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        dynamic_randomization = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.6, 0.6]   # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3] # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.65
        min_dist = 0.2
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.25    # rad
        target_feet_height = 0.06        # m
        cycle_time = 0.64                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 400  # Forces above this value are penalized

        class scales:
            # reference motion tracking
            joint_pos = 2.0
            feet_clearance = 2.0
            feet_contact_number = 1.2
            # gait
            feet_air_time = 2.8
            foot_slip = -0.15
            feet_distance = 0.3
            knee_distance = 0.3
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.6
            tracking_ang_vel = 1.5
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.4
            orientation = 1.2
            base_height = 0.25
            base_acc = 0.3
            # energy
            action_smoothness = -0.02
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.
            # ankle movement reward
            ankle_movement = 0.5


    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.


class alexbotminiCfgPPO(LeggedRobotCfgPPO):
    seed = 3407
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 6001  # number of policy updates

        # logging
        save_interval = 1000  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'alexbotmini'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
