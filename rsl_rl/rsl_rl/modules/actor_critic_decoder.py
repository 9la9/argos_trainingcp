import torch
import torch.nn as nn
from torch.distributions import Normal

try:
    from params_proto.neo_proto import PrefixProto
except Exception:
    class PrefixProto:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__()


class AC_Args(PrefixProto, cli=False):
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'

    cenet_encoder_branch_input_dims = [45 * 5]
    cenet_encoder_branch_latent_dims = [64]
    cenet_encoder_branch_hidden_dims = [[128]]

    cenet_decoder_branch_input_dims = [19]
    cenet_decoder_branch_output_dims = [45]
    cenet_decoder_branch_hidden_dims = [[64, 128]]


class Vae(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        activation = nn.ReLU()
        for branch_input_dim, branch_hidden_dims, branch_latent_dim in zip(
            AC_Args.cenet_encoder_branch_input_dims,
            AC_Args.cenet_encoder_branch_hidden_dims,
            AC_Args.cenet_encoder_branch_latent_dims,
        ):
            env_factor_encoder_layers = [
                nn.Linear(branch_input_dim, branch_hidden_dims[0]),
                activation,
            ]
            for l in range(len(branch_hidden_dims)):
                if l == len(branch_hidden_dims) - 1:
                    env_factor_encoder_layers.append(nn.Linear(branch_hidden_dims[l], branch_latent_dim))
                else:
                    env_factor_encoder_layers.append(nn.Linear(branch_hidden_dims[l], branch_hidden_dims[l + 1]))
                    env_factor_encoder_layers.append(activation)
            self.cenet_encoder = nn.Sequential(*env_factor_encoder_layers)
            self.add_module("cenet_encoder", self.cenet_encoder)

        self.latent_mu = nn.Linear(16 * 4, 19)
        self.latent_var = nn.Linear(16 * 4, 16)

        for branch_input_dim, branch_hidden_dims, branch_latent_dim in zip(
            AC_Args.cenet_decoder_branch_input_dims,
            AC_Args.cenet_decoder_branch_hidden_dims,
            AC_Args.cenet_decoder_branch_output_dims,
        ):
            env_factor_encoder_layers = [
                nn.Linear(branch_input_dim, branch_hidden_dims[0]),
                activation,
            ]
            for l in range(len(branch_hidden_dims)):
                if l == len(branch_hidden_dims) - 1:
                    env_factor_encoder_layers.append(nn.Linear(branch_hidden_dims[l], branch_latent_dim))
                else:
                    env_factor_encoder_layers.append(nn.Linear(branch_hidden_dims[l], branch_hidden_dims[l + 1]))
                    env_factor_encoder_layers.append(activation)
        self.cenet_decoder = nn.Sequential(*env_factor_encoder_layers)
        self.add_module("cenet_decoder", self.cenet_decoder)

        print(f"cenet_encoder Module: {self.cenet_encoder}")
        print(f"cenet_decoder Module: {self.cenet_decoder}")

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def cenet_forward(self, observations_history):
        latent_e = self.cenet_encoder(observations_history)
        latent_var = self.latent_var(latent_e)
        latent_mu = self.latent_mu(latent_e)
        z = self.reparameterize(latent_mu[:, 3:], latent_var)
        return latent_mu, latent_var, z


class ActorCritic_Decoder(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs, num_critic_obs, num_actions, **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]
            ))
        super().__init__()

        activation = get_activation(AC_Args.activation)
        self.num_obs = num_obs
        self.vae = Vae()

        actor_layers = [
            nn.Linear(AC_Args.cenet_decoder_branch_input_dims[0] + num_obs, AC_Args.actor_hidden_dims[0]),
            activation,
        ]
        for l in range(len(AC_Args.actor_hidden_dims)):
            if l == len(AC_Args.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)
        self.add_module("actor_body", self.actor_body)

        critic_layers = [
            nn.Linear(187 + num_obs + 3 + 15 + 12 - 24 + 12, AC_Args.critic_hidden_dims[0]),
            activation,
        ]
        for l in range(len(AC_Args.critic_hidden_dims)):
            if l == len(AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)
        self.add_module("critic_body", self.critic_body)

        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, observations_history):
        self.latent_mu, self.latent_var, self.z = self.vae.cenet_forward(observations_history)
        mean = self.actor_body(torch.cat((observations, self.z, self.latent_mu[:, :3]), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def get_all_proprioception(self, proprioception, num_envs, frame_stack):
        proprioception = proprioception.reshape(num_envs, frame_stack, self.num_obs)
        base_w, projected_gravity, commands, dof_pos, dof_vel, actions, flags = (
            proprioception[:, :, :3],
            proprioception[:, :, 3:6],
            proprioception[:, :, 6:9],
            proprioception[:, :, 9:21],
            proprioception[:, :, 21:33],
            proprioception[:, :, 33:45],
            proprioception[:, :, 45:self.num_obs],
        )
        return base_w, projected_gravity, commands, dof_pos, dof_vel, actions, flags

    def reflect_proprioception(self, state, num_envs, frame_stack):
        device = state.device
        reflect_direction = torch.tensor(
            [-1, 1, 1,
             -1, 1, 1,
             -1, 1, 1,
             -1, 1, 1],
            dtype=state.dtype,
            device=device,
        ).reshape(1, 1, 12)
        reflect_base_w = torch.tensor([-1, 1, -1], dtype=state.dtype, device=device).reshape(1, 1, 3)
        reflect_gravity = torch.tensor([1, -1, 1], dtype=state.dtype, device=device).reshape(1, 1, 3)
        reflect_command = torch.tensor([1, -1, -1], dtype=state.dtype, device=device).reshape(1, 1, 3)
        motor_index = torch.tensor(
            [3, 4, 5,
             0, 1, 2,
             9, 10, 11,
             6, 7, 8],
            dtype=torch.long,
            device=device,
        )

        base_w, projected_gravity, commands, dof_pos, dof_vel, actions, flags = self.get_all_proprioception(
            state, num_envs, frame_stack
        )

        base_w = base_w * reflect_base_w
        projected_gravity = projected_gravity * reflect_gravity
        commands = commands * reflect_command
        dof_pos = dof_pos[:, :, motor_index] * reflect_direction
        dof_vel = dof_vel[:, :, motor_index] * reflect_direction
        actions = actions[:, :, motor_index] * reflect_direction

        new_state = torch.cat([
            base_w, projected_gravity, commands, dof_pos, dof_vel, actions, flags
        ], dim=-1)
        return new_state.reshape(num_envs, -1)

    def get_mu_mirror(self, observations, observation_history):
        frame_stack = observation_history.shape[-1] // self.num_obs
        obs_mirror = self.reflect_proprioception(observations, observations.shape[0], 1)
        obs_history_mirror = self.reflect_proprioception(observation_history, observation_history.shape[0], frame_stack)
        latent_mu_mirror, _, z_mirror = self.vae.cenet_forward(obs_history_mirror)
        mean_mirror = self.actor_body(torch.cat((obs_mirror, z_mirror, latent_mu_mirror[:, :3]), dim=-1))

        reflect_direction = torch.tensor(
            [-1, 1, 1,
             -1, 1, 1,
             -1, 1, 1,
             -1, 1, 1],
            dtype=observations.dtype,
            device=observations.device,
        ).reshape(1, 12)
        motor_index = torch.tensor(
            [3, 4, 5,
             0, 1, 2,
             9, 10, 11,
             6, 7, 8],
            dtype=torch.long,
            device=observations.device,
        )
        return mean_mirror[:, motor_index] * reflect_direction

    def act(self, observations, observations_history, rew_buf, **kwargs):
        self.update_distribution(observations, observations_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob):
        observations = ob["obs"]
        latent_e = self.vae.cenet_encoder(ob["obs_history"])
        latent = self.vae.latent_mu(latent_e)
        return self.actor_body(torch.cat((observations, latent[:, 3:], latent[:, :3]), dim=-1))

    def evaluate(self, critic_observations, privileged_observations, base_vel, **kwargs):
        value = self.critic_body(torch.cat((
            critic_observations,
            base_vel,
            privileged_observations[:, 187:190 + 12],
            privileged_observations[:, 190 + 12:],
        ), dim=-1))
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    if act_name == "selu":
        return nn.SELU()
    if act_name == "relu":
        return nn.ReLU()
    if act_name == "crelu":
        return nn.ReLU()
    if act_name == "lrelu":
        return nn.LeakyReLU()
    if act_name == "tanh":
        return nn.Tanh()
    if act_name == "sigmoid":
        return nn.Sigmoid()
    print("invalid activation function!")
    return None
