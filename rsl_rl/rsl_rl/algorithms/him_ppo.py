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

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import HIMRolloutStorage

class HIMPPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 value_smoothness_coef=0.1,
                 smoothness_upper_bound=1.0,
                 smoothness_lower_bound=0.1,
                 amp=None,
                 amp_normalizer=None,
                 motion_buffer=None
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.value_smoothness_coef = value_smoothness_coef
        self.smoothness_upper_bound = smoothness_upper_bound
        self.smoothness_lower_bound = smoothness_lower_bound

        # amp
        self.amp = {
            "lefthand": deepcopy(amp),
            "righthand": deepcopy(amp),
            "leftjump": deepcopy(amp),
            "rightjump": deepcopy(amp),
            "leftstep": deepcopy(amp),
            "rightstep": deepcopy(amp)
        }

        for model in self.amp.values():
            model.to(self.device)
        params = [
            {'params': self.actor_critic.parameters(), 'name': 'actor_critic'},
        ]

        for key in self.amp:
            params.append({
                'params': self.amp[key].trunk.parameters(),
                'weight_decay': 10e-4,
                'name': f'amp_trunk_{key}'
            })
            params.append({
                'params': self.amp[key].amp_linear.parameters(),
                'weight_decay': 10e-2,
                'name': f'amp_head_{key}'
            })
        self.optimizer = optim.Adam(params, lr=learning_rate)
        self.amp_normalizer = amp_normalizer
        self.motion_buffer = motion_buffer





        self.transition = HIMRolloutStorage.Transition()

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, amp_obs_shape):
        self.storage = HIMRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, amp_obs_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values

        if obs.isnan().any():
            obs = torch.zeros((obs.shape[0],obs.shape[1]), device=obs.device)
            critic_obs = torch.zeros((critic_obs.shape[0],critic_obs.shape[1]), device=obs.device)

        self.transition.actions = self.actor_critic.act(obs)[0].detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def process_amp_state(self, amp_state):
        self.transition.amp_observations = amp_state

    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)


    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_est_loss = 0
        mean_region_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, next_obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, cont_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, amp_obs_batch in generator:

                _, estball_batch, estregion_batch = self.actor_critic.act(obs_batch)
                
                gtball_batch = critic_obs_batch[:, -13:-7]
                gtregion_batch = (3 * critic_obs_batch[:, -14]).long()

                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)

                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
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

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                
                # est ball loss
                est_loss = (estball_batch - gtball_batch).pow(2).mean()
                region_loss = nn.CrossEntropyLoss()(estregion_batch, gtregion_batch)
                loss = surrogate_loss + est_loss + region_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Smooth loss
                epsilon = self.smoothness_lower_bound / (self.smoothness_upper_bound - self.smoothness_lower_bound)
                policy_smooth_coef = self.smoothness_upper_bound * epsilon; value_smooth_coef = self.value_smoothness_coef * policy_smooth_coef

                mix_weights = cont_batch * (torch.rand_like(cont_batch) - 0.5) * 2.0
                mix_obs_batch = obs_batch + mix_weights * (next_obs_batch - obs_batch)
                mix_critic_obs_batch = critic_obs_batch + mix_weights * (next_critic_obs_batch - critic_obs_batch)
                policy_smooth_loss = torch.square(torch.norm(mu_batch - self.actor_critic.act_inference(mix_obs_batch), dim=-1)).mean()
                value_smooth_loss = torch.square(torch.norm(value_batch - self.actor_critic.evaluate(mix_critic_obs_batch), dim=-1)).mean()
                smooth_loss = policy_smooth_coef * policy_smooth_loss + value_smooth_coef * value_smooth_loss

                loss += smooth_loss

                # amp loss
                if self.amp is not None:
                    # import ipdb; ipdb.set_trace()

                    motion_ids = 3 * critic_obs_batch[:,self.actor_critic.num_one_step_obs + 3]
                    amp_expert_obs_batch_mask = torch.zeros_like(amp_obs_batch)
                    motion_ids_0 = motion_ids == 0
                    if motion_ids_0.any():
                        amp_expert_obs_batch_mask[motion_ids_0] = self.motion_buffer["lefthand"].get_expert_obs(
                            batch_size=obs_batch[motion_ids_0].shape[0]
                        ).to(self.device)

                    motion_ids_1 = motion_ids == 1
                    if motion_ids_1.any():
                        amp_expert_obs_batch_mask[motion_ids_1] = self.motion_buffer["righthand"].get_expert_obs(
                            batch_size=obs_batch[motion_ids_1].shape[0]
                        ).to(self.device)

                    motion_ids_2 = motion_ids == 2
                    if motion_ids_2.any():
                        amp_expert_obs_batch_mask[motion_ids_2] = self.motion_buffer["leftjump"].get_expert_obs(
                            batch_size=obs_batch[motion_ids_2].shape[0]
                        ).to(self.device)

                    motion_ids_3 = motion_ids == 3
                    if motion_ids_3.any():
                        amp_expert_obs_batch_mask[motion_ids_3] = self.motion_buffer["rightjump"].get_expert_obs(
                            batch_size=obs_batch[motion_ids_3].shape[0]
                        ).to(self.device)

                    motion_ids_4 = motion_ids == 4
                    if motion_ids_4.any():
                        amp_expert_obs_batch_mask[motion_ids_4] = self.motion_buffer["leftstep"].get_expert_obs(
                            batch_size=obs_batch[motion_ids_4].shape[0]
                        ).to(self.device)

                    motion_ids_5 = motion_ids == 5
                    if motion_ids_5.any():
                        amp_expert_obs_batch_mask[motion_ids_5] = self.motion_buffer["rightstep"].get_expert_obs(
                            batch_size=obs_batch[motion_ids_5].shape[0]
                        ).to(self.device)

                                        
                    amp_expert_obs_batch = self.amp_normalizer.normalize_torch(amp_expert_obs_batch_mask, self.device)
                    amp_obs_batch = self.amp_normalizer.normalize_torch(amp_obs_batch, self.device)
            
                    amp_loss, expert_loss, policy_loss = 0.0, 0.0, 0.0
                    for motion_key, motion_mask in zip(
                        ["lefthand", "righthand", "leftjump", "rightjump" , "leftstep", "rightstep"],
                        [motion_ids_0, motion_ids_1, motion_ids_2, motion_ids_3, motion_ids_4, motion_ids_5]
                    ):
                        if motion_mask.any():
                            loss_part, expert_loss_part, policy_loss_part = self.amp[motion_key].compute_loss(
                                amp_obs_batch[motion_mask], amp_expert_obs_batch[motion_mask])
                            amp_loss += loss_part
                            expert_loss += expert_loss_part
                            policy_loss += policy_loss_part


                    loss += amp_loss
                    self.amp_normalizer.update(amp_obs_batch.cpu().detach().numpy())
                    self.amp_normalizer.update(amp_expert_obs_batch.cpu().detach().numpy())

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_est_loss += est_loss.item()
                mean_region_loss += region_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_est_loss /= num_updates
        mean_region_loss /= num_updates


        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_est_loss, mean_region_loss, amp_loss, expert_loss, policy_loss