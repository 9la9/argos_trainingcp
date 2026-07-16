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
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RSL_RL_DIR = os.path.join(ROOT_DIR, "rsl_rl")
for path in (ROOT_DIR, RSL_RL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(ROOT_DIR, ".cache", "torch_extensions"))

from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, Logger

import numpy as np
import torch
import copy
from rsl_rl.modules.actor_critic_decoder import AC_Args
from rsl_rl.env.wrappers.history_wrapper import HistoryWrapper


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 20)
    env_cfg.env.play_command = True
    env_cfg.commands.heading_command = False
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_restitution = True
    env_cfg.domain_rand.randomize_force_inject = False
    env_cfg.domain_rand.randomize_Kp_factor = False
    env_cfg.domain_rand.randomize_Kd_factor = False
    env_cfg.domain_rand.randomize_link_mass = False

    env_cfg.env.debug_viz = False

    # high-quality rendering for play mode
    env_cfg.viewer.camera_width = 3840
    env_cfg.viewer.camera_height = 2160
    env_cfg.viewer.camera_fov = 50.0
    env_cfg.viewer.camera_supersampling = 4
    env_cfg.viewer.pos = [2.0, -3.0, 1.5]
    env_cfg.viewer.lookat = [2.0, 0.0, 0.5]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env = HistoryWrapper(env)

    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(True, device=env.device)

    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    logger = Logger(env.dt, path)
    robot_index = 0
    stop_state_log = 500
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
        # ==================== 导出 TorchScript 模型 ====================
    export_dir = os.path.join(ROOT_DIR, 'policy')
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, 'policy_waq_xdog_latest.pt')

    class ExportPolicy(torch.nn.Module):
        def __init__(self, actor_critic):
            super().__init__()
            self.num_obs = actor_critic.num_obs       # 45
            self.vae_encoder = copy.deepcopy(actor_critic.vae.cenet_encoder).to('cpu')
            self.latent_mu = copy.deepcopy(actor_critic.vae.latent_mu).to('cpu')
            self.actor_body = copy.deepcopy(actor_critic.actor_body).to('cpu')

        def forward(self, x):
            # x: [batch, 270] = [obs(45), obs_history(225)]
            obs = x[:, :self.num_obs]
            obs_history = x[:, self.num_obs:]
            latent_e = self.vae_encoder(obs_history)
            latent = self.latent_mu(latent_e)
            # [obs(45), z(16), privileged(3)] = 64d → action(12)
            action = self.actor_body(torch.cat([
                obs, latent[:, 3:], latent[:, :3]
            ], dim=-1))
            return action

    print("\n正在导出 TorchScript 模型...")
    try:
        export_model = ExportPolicy(ppo_runner.alg.actor_critic)
        export_model.eval()

        total_dim = env_cfg.env.num_observations + env_cfg.env.num_obs_history  # 270
        dummy_input = torch.zeros(1, total_dim, device='cpu')

        traced = torch.jit.trace(export_model, dummy_input)
        traced.save(export_path)
        print(f"✅ 导出成功 → {export_path}")
    except Exception as e:
        print(f"❌ 导出失败: {e}")
    # ==============================================================

    
    for i in range(100*int(env.max_episode_length)):
        obs_last = obs["obs"]
        obs_history = obs["obs_history"]
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions.detach())

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_x': env.base_pos[robot_index, 0].item(),
                    'base_y': env.base_pos[robot_index, 1].item(),
                    'base_z': env.base_pos[robot_index, 2].item(),
                    'torques_0': env.torques[robot_index, 0].cpu().numpy(),
                    'torques_1': env.torques[robot_index, 1].cpu().numpy(),
                    'torques_2': env.torques[robot_index, 2].cpu().numpy(),
                    'torques_3': env.torques[robot_index, 3].cpu().numpy(),
                    'torques_4': env.torques[robot_index, 4].cpu().numpy(),
                    'torques_5': env.torques[robot_index, 5].cpu().numpy(),
                    'torques_6': env.torques[robot_index, 6].cpu().numpy(),
                    'torques_7': env.torques[robot_index, 7].cpu().numpy(),
                    'torques_8': env.torques[robot_index, 8].cpu().numpy(),
                    'torques_9': env.torques[robot_index, 9].cpu().numpy(),
                    'torques_10': env.torques[robot_index, 10].cpu().numpy(),
                    'torques_11': env.torques[robot_index, 11].cpu().numpy(),
                    'obs': obs_last.detach().cpu().numpy(),
                    'obs_history': obs_history.detach().cpu().numpy(),
                    'actions': actions.detach().cpu().numpy(),
                }
            )


if __name__ == '__main__':
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)