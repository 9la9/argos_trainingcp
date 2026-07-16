import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl.modules import ActorCritic_Decoder
from rsl_rl.storage import RolloutStorage
import itertools

class PPO:
    actor_critic: ActorCritic_Decoder

    def __init__(self,
                 actor_critic,
                 num_learning_epochs=5,
                 num_mini_batches=4,
                 clip_param=0.2,
                 gamma=0.99,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.01,
                 learning_rate=5.e-4,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="adaptive",
                 desired_kl=0.01,
                 device='cpu',
                 **kwargs,
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None
        self.optimizer = optim.Adam(itertools.chain(
            self.actor_critic.actor_body.parameters(),
            self.actor_critic.critic_body.parameters(),
            [self.actor_critic.std],
        ), lr=learning_rate)
        self.vae_optimizer = optim.Adam(self.actor_critic.vae.parameters(), lr=5.e-4)  
        self.transition = RolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.num_adaptation_module_substeps = 1
        self.sym_loss = kwargs.get('sym_loss', 0.0)
    
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, obs_history,base_vel, rew_buf):
        self.transition.actions = self.actor_critic.act(obs, obs_history, rew_buf).detach()

        self.transition.values = self.actor_critic.evaluate(obs, privileged_obs,base_vel).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        self.transition.base_vel = base_vel 
        return self.transition.actions
    
    def process_env_step(self, rewards, dones,next_obs, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_observations = next_obs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs, last_critic_privileged_obs,last_base_vel):
        last_values= self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs,last_base_vel).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_recons_loss = 0
        mean_vel_loss = 0
        mean_kld_loss = 0
        mean_sym_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch,base_vel_batch,next_obs_batch, hid_states_batch, masks_batch,rew_buf_batch in generator:
            latent_mu, latent_var, z = self.actor_critic.vae.cenet_forward(obs_history_batch)
            recons = self.actor_critic.vae.cenet_decoder(torch.cat([z, latent_mu[:,:3]], dim = 1))
            delta_recon = recons - next_obs_batch
            recons_loss =torch.pow(delta_recon,2).mean(-1).mean()
            vel_loss = F.mse_loss(latent_mu[:,:3], base_vel_batch)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + latent_var - latent_mu[:,3:].pow(2) - latent_var.exp(), dim = 1))
            vae_loss = recons_loss+vel_loss + 4*kld_loss
            self.vae_optimizer.zero_grad()
            vae_loss.backward()

            nn.utils.clip_grad_norm_(self.actor_critic.vae.parameters(), self.max_grad_norm)
            self.vae_optimizer.step()
            
            mean_recons_loss += recons_loss.item()
            mean_vel_loss += vel_loss.item()
            mean_kld_loss += kld_loss.item()
            
            self.actor_critic.act(obs_batch, obs_history_batch, rew_buf_batch, masks=masks_batch, hidden_states=hid_states_batch[0])

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, privileged_obs_batch,base_vel_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.sym_loss > 0 and hasattr(self.actor_critic, "get_mu_mirror"):
                mu_mirror = self.actor_critic.get_mu_mirror(obs_batch, obs_history_batch)
                sym_loss = (mu_batch - mu_mirror).pow(2).mean()
            else:
                sym_loss = torch.tensor(0.0, device=self.device)

            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate


            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                            1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.sym_loss * sym_loss
                    
                    

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_batch.mean().item()
            mean_sym_loss += sym_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_sym_loss /= num_updates
        mean_recons_loss /= (num_updates * self.num_adaptation_module_substeps)
        mean_vel_loss /= (num_updates * self.num_adaptation_module_substeps)
        mean_kld_loss /= (num_updates * self.num_adaptation_module_substeps)
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_recons_loss, mean_vel_loss, mean_kld_loss, mean_sym_loss
