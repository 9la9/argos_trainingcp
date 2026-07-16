import os
import sys
import copy

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RSL_RL_DIR = os.path.join(ROOT_DIR, "rsl_rl")
for path in (ROOT_DIR, RSL_RL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(ROOT_DIR, ".cache", "torch_extensions"))

from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import torch
import torch.nn as nn
from rsl_rl.env.wrappers.history_wrapper import HistoryWrapper


class PolicyExporterDecoder(nn.Module):
    """Wraps ActorCritic_Decoder's act_expert inference as a TorchScript exportable module.

    Inputs:
        obs         : (N, num_obs)         current proprioception
        obs_history : (N, num_obs * stack) stacked history fed to the encoder
    Output:
        actions     : (N, num_actions)
    """
    def __init__(self, actor_critic):
        super().__init__()
        self.cenet_encoder = copy.deepcopy(actor_critic.vae.cenet_encoder).cpu()
        self.latent_mu     = copy.deepcopy(actor_critic.vae.latent_mu).cpu()
        self.actor_body    = copy.deepcopy(actor_critic.actor_body).cpu()

    def forward(self, obs: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
        latent_e = self.cenet_encoder(obs_history)
        latent   = self.latent_mu(latent_e)
        return self.actor_body(torch.cat((obs, latent[:, 3:], latent[:, :3]), dim=-1))


def export_policy(actor_critic, path: str):
    os.makedirs(path, exist_ok=True)
    exporter = PolicyExporterDecoder(actor_critic)
    exporter.eval()
    scripted = torch.jit.script(exporter)
    out = os.path.join(path, "policy.pt")
    scripted.save(out)
    print(f"Policy exported to: {out}")


def export(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env = HistoryWrapper(env)

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    export_policy(ppo_runner.alg.actor_critic, path)


if __name__ == '__main__':
    args = get_args()
    export(args)
