# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
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

import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt

# import isaacgym
from humanoid.envs import *
from humanoid.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime

from humanoid.scripts.ps5_joystick import ps5_joystick
import os
from fcntl import ioctl
import threading


class Logger:
    def __init__(self, dt):
        self.dt = dt
        # 添加电机位置环记录
        self.joint_log = defaultdict(list)
        # 添加奖励参数记录
        self.reward_log = defaultdict(list)

    def log_joint_states(self, joint_data):
        """记录所有关节状态"""
        for key, value in joint_data.items():
            self.joint_log[key].append(value)

    def log_rewards_detail(self, reward_dict):
        """记录详细奖励参数"""
        for key, value in reward_dict.items():
            self.reward_log[key].append(value)

    def save_data(self, joint_data, reward_data, filename):
        """保存详细数据到文件"""
        data = {
            'joint': joint_data,
            'reward': reward_data,
            'timestamps': np.arange(len(joint_data['dof_pos_target'])) * self.dt
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def plot_joint_states(self):
        """绘制关节状态曲线"""
        plt.figure(figsize=(12, 8))
        for i in range(12):
            plt.subplot(3, 4, i + 1)
            plt.plot([x[i] for x in self.joint_log['dof_pos_target']], label='Target')
            plt.plot([x[i] for x in self.joint_log['dof_pos_actual']], label='Actual')
            plt.title(f'Joint {i + 1}')
            plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_reward_components(self):
        """绘制奖励成分分解"""
        plt.figure(figsize=(10, 6))
        for name in self.reward_log.keys():
            if name != 'total_reward':
                plt.plot(self.reward_log[name], label=name, alpha=0.7)
        plt.plot(self.reward_log['total_reward'], label='Total', lw=2, c='k')
        plt.legend(ncol=3)
        plt.title('Reward Components')
        plt.show()


def play(args):
    # 启动事件处理线程ps5_joystick
    joystick = ps5_joystick()
    event_thread = threading.Thread(target=joystick.handle_events)
    event_thread.daemon = True
    event_thread.start()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 125)
    env_cfg.sim.max_gpu_contact_pairs = 2 ** 10
    env_cfg.terrain.mesh_type = 'trimesh'
    # env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.joint_angle_noise = 0.
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 1200  # number of steps before plotting states
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                      np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir + datetime.now().strftime('%b%d_%H-%M-%S') + '.mp4')
        # dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))

    # 创建奖励名称列表
    reward_names = [
        'tracking_lin_vel', 'tracking_ang_vel', 'vel_mismatch_exp',
        'low_speed', 'orientation', 'torques', 'action_smoothness',
        'joint_pos', 'feet_air_time', 'collision', 'feet_contact_forces'
    ]

    for i in tqdm(range(stop_state_log)):
        actions = policy(obs.detach())  # * 0.

        if FIX_COMMAND:
            stick_values = joystick.get_stick_values()

            # env.commands[:, 0] = 0.4
            # env.commands[:, 1] = 0
            # env.commands[:, 2] = 0
            # env.commands[:, 3] = 0

            env.commands[:, 0] = stick_values["left_stick_y"] / 32768  # max1.0
            # print('stick_values["left_stick_x"]/32768',stick_values["left_stick_x"]/32768)
            env.commands[:, 1] = -stick_values["left_stick_x"] / 32768
            # print('stick_values["left_stick_y"]/32768',stick_values["left_stick_y"]/32768)
            env.commands[:, 2] = stick_values["right_stick_x"] / 32768
            # print('stick_values["right_stick_x"]/32768',stick_values["right_stick_x"]/32768)
            env.commands[:, 3] = stick_values["right_stick_y"] / 32768
            # print('stick_values["right_stick_y"]/32768',stick_values["right_stick_y"]/32768)

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])

        # ================ 新增关节状态记录 ================
        joint_states = {
            'dof_pos_target': actions[robot_index].cpu().numpy() * env.cfg.control.action_scale,
            'dof_pos_actual': env.dof_pos[robot_index].cpu().numpy(),
            'dof_vel': env.dof_vel[robot_index].cpu().numpy(),
            'dof_torque': env.torques[robot_index].cpu().numpy()
        }
        logger.log_joint_states(joint_states)

        # ================ 新增奖励参数记录 ================
        reward_values = {}
        for name in reward_names:
            # 获取每个奖励项的当前值
            reward_values[name] = getattr(env, f'_reward_{name}')().mean().item()

        # 添加基础奖励
        reward_values['total_reward'] = rews.mean().item()

        logger.log_rewards_detail(reward_values)

        # ================ 新增实时显示 ================
        if i % 50 == 0:  # 每50步显示一次
            print("\n=== 关节状态 ===")
            print(f"目标位置: {joint_states['dof_pos_target'].round(3)}")
            print(f"实际位置: {joint_states['dof_pos_actual'].round(3)}")
            print(f"实际扭矩: {joint_states['dof_torque'].round(3)}")

            print("\n=== 奖励参数 ===")
            for k, v in reward_values.items():
                print(f"{k:20}: {v:.4f}")

    # 新增保存功能
    logger.save_data(
        joint_data=logger.joint_log,
        reward_data=logger.reward_log,
        filename=os.path.join(experiment_dir, 'detailed_log.pkl')
    )

    # 新增绘图功能
    logger.plot_joint_states()
    logger.plot_reward_components()

    if RENDER:
        video.release()


if __name__ == '__main__':
    EXPORT_POLICY = True
    RENDER = True
    FIX_COMMAND = True
    args = get_args()
    play(args)
