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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class XdogRoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 45
        num_privileged_obs = 202 + 187 + 12 - 24 + 12
        num_obs_history = 45 * 5
        num_observation_history = 5
        debug_viz = False

        num_actions = 12
        play_command = False

    class terrain(LeggedRobotCfg.terrain):
        measure_heights = True
        mesh_type = "plane"
        measure_foot_clearance =False
        
        terrain_proportions =[0.3, 0.3, 0.4, 0.0, 0.0] #[0.35, 0.35, 0.0, 0.0, 0.2,0.1] 
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.32]
        pos_z_range = [0.35, 0.40]
        default_joint_angles = {
            "FL_hip_joint": -0.1,
            "RL_hip_joint": -0.1,
            "FR_hip_joint": 0.1,
            "RR_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"hip": 25.0, "thigh": 25.0, "calf": 25.0}
        damping = {"hip": 0.5, "thigh": 0.5, "calf": 0.5}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/xdog/urdf/XDOG.urdf"
        name = "Xdog"
        foot_name = "foot"
        penalize_contacts_on = ["base", "thigh", "calf", "calflower"]
        collision_state = ["base", "thigh", "calf", "calflower"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0
        max_angular_velocity = 10.0
        max_linear_velocity = 10.0
        flip_visual_attachments = False

    class commands(LeggedRobotCfg.commands):
        heading_command = True
        resampling_time = 4.0
        curriculum = True
        max_curriculum = 2.5
        class ranges:
            """lin_vel_x = [-2.5, 2.5]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]"""
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]
            

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.99
        base_height_target = 0.32
        max_acc = 120.0
        class scales(LeggedRobotCfg.rewards.scales):
            dof_pos_limits = -5.0
            foot_acc = -0.01
            stand_still = -2.2
            #feet_air_time = 4.5
            #ang_vel_xy = -0.15 #-0.1
            #lin_vel_z = -1.0#
            #feet_slip = -1.3#
            #hip_pos = -0.6
            """tracking_lin_vel = 3.225#3.125
            tracking_ang_vel = 1.975#1.875
            
            
            base_height = -4.5"""
            


    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 1.5]
        randomize_restitution = True
        restitution_range = [0.0, 0.4]
        randomize_base_mass = True
        added_mass_range = [0.5, 3.0]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        max_push_force_xy = 15.0
        max_push_force_offset = 0.1

        randomize_motor_strength = True
        kp_range = [0.9, 1.1]
        randomize_Kp_factor = True
        kd_range = [0.9, 1.1]
        randomize_Kd_factor = True
        init_joint_range = [0.5, 1.5]
        motor_strength = [0.9, 1.1]
        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        randomize_com_displacement = True
        com_displacement_range = [-0.03, 0.03]
        randomize_motor_offset = False
        motor_offset_range = [-0.04, 0.04]
        randomize_force_inject = False
        force_inject_range = [-0.1, 0.1]
        randomize_base_inertia = False
        base_inertia_range = [0.8, 1.2]


class XdogRoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        sym_loss = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCritic_Decoder"
        run_name = ""
        experiment_name = "rough_xdog"
        max_iterations = 50000

        resume = False
        load_run = -1
        checkpoint = -1
