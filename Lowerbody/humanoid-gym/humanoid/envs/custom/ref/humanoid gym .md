# humanoid gym 调研

## 1. 介绍

* 基于isaac gym开发的RL框架，目前主要是用来训练locomotion。整体开发风格延续isaac gym，一个env文件加上一个config文件。
* 支持从simulation到real-world的zero-shot迁移；
* 也集成了从isaac gym到Mujoco的迁移，sim to sim。
  * 作为对比，我们公司同样开源了在Webots、Mujoco、Gazebo这三个平台的sim to sim流程。


## 2. Features

### 2.1 目前已有的features 

* Humanoid Robot Training
  * 详细的RL训练介绍。
  * 针对于locomotion的rewards设计。
* Sim to Sim
  * Mujoco中的参数经过精调和标定。仿真与真实环境高度对齐。
  * 在humanoid_gym框架中，将训练好的神经网络直接部署到mujoco中，进行验证。

### 2.2 将来会加入的features

* Denoising World Model Learning(即将到来)
  * 集成状态估计和系统辨识，服务于sim to real。
* 灵巧手操作（即将到来）

## 3. 训练效果

抬腿高度明显，行走步态尚可。在实机上部署，存在走偏的现象。

![demo](humanoid%20gym%20%E8%B0%83%E7%A0%94%E6%8A%A5%E5%91%8A.assets/demo.gif)

## 4. 训练设置

### 4.1 观测值

#### 4.1.1 policy obs

单个step的policy obs包含47个值。一共堆叠15个step的数据，作为policy网络的输入值。会给policy obs添加噪声。

| 名字                                        | 含义                                                         | 长度 |
| ------------------------------------------- | ------------------------------------------------------------ | ---- |
| self.command_input                          | 输入command，正弦相位、余弦相位、x轴y轴上的线速度、z轴上的角速度 | 5    |
| q                                           | 关节位置                                                     | 12   |
| dq                                          | 关节速度                                                     | 12   |
| self.actions                                | 上一个step的action                                           | 12   |
| self.base_ang_vel * self.obs_scales.ang_vel | base的角速度                                                 | 3    |
| self.base_euler_xyz * self.obs_scales.quat  | base的欧拉角                                                 | 3    |

#### 4.1.2 critic obs

单个step的critic obs包含73个值。一共堆叠3个step的数据，作为critic网络的输入值。不给critic obs添加噪声。

| 名字                                                         | 含义                                                         | 长度 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| self.command_input                                           | 输入command，正弦相位、余弦相位、x轴y轴上的线速度、z轴上的角速度 | 5    |
| (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos | 关节位置                                                     | 12   |
| self.dof_vel * self.obs_scales.dof_vel                       | 关节速度                                                     | 12   |
| self.actions                                                 | 上一个step的action                                           | 12   |
| diff                                                         | 关节位置与参考值的差异。参考值的变化是一个正弦函数。主要影响hip pitch，knee pitch，ankle pitch这三个关节。 | 12   |
| self.base_lin_vel * self.obs_scales.lin_vel                  | base的线速度                                                 | 3    |
| self.base_ang_vel * self.obs_scales.ang_vel                  | base的角速度                                                 | 3    |
| self.base_euler_xyz * self.obs_scales.quat                   | base的欧拉角                                                 | 3    |
| self.rand_push_force[:, :2]                                  | 在x、y轴上的线速度扰动                                       | 2    |
| self.rand_push_torque                                        | 在x、y、z轴上的力矩扰动                                      | 3    |
| self.env_frictions                                           | 环境摩擦力。没有具体实现，值都为0                            | 1    |
| self.body_mass / 30.                                         | 质量。没有具体实现，值都为0                                  | 1    |
| stance_mask                                                  | 期望的支撑相位掩码。值为1表示处于支撑相位                    | 2    |
| contact_mask                                                 | 触地掩码。值为1表示脚触地                                    | 2    |

### 4.2 输出值

* actions包含12个值，对应两条腿上12个执行器的目标位置，每条腿上6个执行器。
* 会给actions添加delay。每个step中会随机一个delay值。计算公式为 actions = (1 - delay) * actions + delay * self.last_actions。
* 会给actions添加噪声。

### 4.3 神经网络

| 对象               | 值              |
| ------------------ | --------------- |
| 优化算法           | PPO             |
| actor hidden dims  | [512，256，128] |
| critic hidden dims | [768，256，128] |

## 5. Rewards设计 

### 5.1 第一部分，涉及 reference motion tracking

| 函数名                      | 函数功能                                                     |
| --------------------------- | :----------------------------------------------------------- |
| _reward_joint_pos           | 处理当前关节角与参考关节角的差距。包含两个部分，指数形式的正向奖励+线性函数惩罚。当值比较小时，指数函数发挥作用。当值变大后，指数函数变得平坦。这时，线性函数发挥作用。 |
| _reward_feet_clearance      | 当抬脚高度达到指定高度，并且此时位于摆动相，则获得奖励。     |
| _reward_feet_contact_number | 触地的脚的数量是否符合规划的步态相位。如果符合获得奖励，否则获得惩罚。 |

### 5.2 第二部分，涉及 gait

| 函数名                | 函数功能                                                     |
| --------------------- | ------------------------------------------------------------ |
| _reward_feet_air_time | 将脚的滞空时间作为奖励值返回，最大值限制为0.5秒。即鼓励机器人每一步在空中停留0.5秒。 |
| _reward_foot_slip     | 在脚底板触地的同时，如果在x轴、y轴上有速度，则会受到惩罚。   |
| _reward_feet_distance | 计算两脚之间的距离。如果小于下限，或者大于上限，就会受到惩罚。 |
| _reward_knee_distance | 控制两个膝盖之间的距离。效果同上。                           |

### 5.3 第三部分， 涉及 contact

| 函数名                      | 函数功能                                 |
| --------------------------- | ---------------------------------------- |
| _reward_feet_contact_forces | 如果脚底板上的力超过上限，就会受到惩罚。 |

### 5.4 第四部分，涉及 velocity tracking

| 函数名                   | 函数功能                                                     |
| ------------------------ | ------------------------------------------------------------ |
| _reward_tracking_lin_vel | 对base在x轴、y轴上的线速度进行跟踪。                         |
| _reward_tracking_ang_vel | 对base在yaw轴上的角速度进行跟踪。                            |
| _reward_vel_mismatch_exp | 不希望base在Z轴上有线速度。不希望base在roll轴、pitch轴上有角速度。 |
| _reward_low_speed        | 希望base在x轴上的线速度可以跟上command。如果实际速度 > 50% command且 < 120% command，就可以得到奖励。 |
| _reward_track_vel_hard   | 同时进行对base在x轴、y轴上的线速度跟踪和在yaw轴上的角速度跟踪。类似于_reward_joint_pos，使用了指数形式的正向奖励+线性函数惩罚。 |

### 5.5 第五部分，涉及 base position

| 函数名                    | 函数功能                                                     |
| ------------------------- | ------------------------------------------------------------ |
| _reward_default_joint_pos | 包含两个部分。第一部分，惩罚所有关节位置与默认位置的偏差，权重较小。第二部分，惩罚髋关节的roll轴关节、yaw轴关节与默认位置的偏差，权重较大。 |
| _reward_orientation       | 确保机器人base在roll轴上、pitch轴上没有转动。包含两个部分。第一部分，计算base在roll轴、pitch轴上的值，如果有值则得到惩罚。
第二部分，计算机器人重力方向在xy平面上的投影，如果有值则得到惩罚。两部分功能相同，有冗余，但是可以增加可靠性和强健性。 |
| _reward_base_height       | 计算base实际高度与目标高度的偏差。计算的过程中考虑到了脚底板的厚度，为5cm。 |
| _reward_base_acc          | 对base的加速度进行限制，不希望base出现大的加速度。           |

### 5.6 第六部分，涉及 energy

| 函数名                    | 函数功能                                                     |
| ------------------------- | ------------------------------------------------------------ |
| _reward_action_smoothness | 用来降低连续step的actions之间的差距，以产生连续的动作。分为三个部分。第一部分，一阶差分。当前action与上一个action之间的差距。效果类似于dof_vel。第二部分，二阶差分。（action - last_action) - (last_acton - last_last_action)。效果类似于dof_acc。第三部分，action的绝对值。惩罚过大的动作幅度，鼓励小幅度、平滑的动作。 |
| _reward_torques           | 惩罚过大的关节力矩。                                         |
| _reward_dof_vel           | 惩罚过大的关节速度。                                         |
| _reward_dof_acc           | 惩罚过大的关节加速度。                                       |
| _reward_collision         | 避免base link发生碰撞，比如base link与手臂的碰撞。           |

