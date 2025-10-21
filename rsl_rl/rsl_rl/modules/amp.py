import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch import autograd

DISC_LOGIT_INIT_SCALE = 1.0



import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class AMP(nn.Module):
    is_recurrent = False
    def __init__(self,  num_obs,
                        amp_coef,
                        hidden_dims=[512, 256],
                        activation='relu',
                        init_noise_std=1.0,
                        device='cuda:0',
                        **kwargs):
        if kwargs:
            print("AMP.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(AMP, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim = num_obs
        mlp_output_dim = 1 

        # discriminator
        disc_layers = []
        disc_layers.append(nn.Linear(mlp_input_dim, hidden_dims[0]))
        disc_layers.append(activation)

        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                ln = nn.Linear(hidden_dims[l], mlp_output_dim)
                torch.nn.init.uniform_(ln.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
                torch.nn.init.zeros_(ln.bias) 
                self.amp_linear = ln
            else:
                ln = nn.Linear(hidden_dims[l], hidden_dims[l+1])
                torch.nn.init.uniform_(ln.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
                torch.nn.init.zeros_(ln.bias) 
                disc_layers.append(ln)
                disc_layers.append(activation)
        self.trunk = nn.Sequential(*disc_layers)

        print(f"Discriminator MLP: {self.trunk}")
        self.device = device
        self.amp_coef = amp_coef
        
    def compute_grad_pen(self, expert_state, policy_state, lambda_=5):
        # alpha = torch.rand(expert_state.size(0), 1)
        # alpha = alpha.expand_as(expert_state).to(expert_state.device)

        # mixup_data = alpha * expert_state + (1 - alpha) * policy_state
        # mixup_data.requires_grad = True

        # disc = self.disc(mixup_data)
        # ones = torch.ones(disc.size()).to(disc.device)
        # grad = autograd.grad(
        #     outputs=disc,
        #     inputs=mixup_data,
        #     grad_outputs=ones,
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True)[0]

        # grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        # return grad_pen
        expert_state.requires_grad = True
        disc_demo_logit = self.trunk(expert_state)
        disc_demo_logit = self.amp_linear(disc_demo_logit)
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, expert_state, grad_outputs=torch.ones_like(disc_demo_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        grad_pen = disc_grad_penalty * lambda_
        return grad_pen
    
    def forward(self, x):
        disc_demo_logit = self.trunk(x)
        disc_demo_logit = self.amp_linear(disc_demo_logit)
        return disc_demo_logit

    def compute_loss(self, agent_obs, expert_obs):
        # agent_obs[:agent_obs.shape[0]//20] = expert_obs[:agent_obs.shape[0]//20]
        policy_d = self.amp_linear(self.trunk(agent_obs))
        expert_d = self.amp_linear(self.trunk(expert_obs))

        expert_loss = (expert_d - 1).pow(2).mean()
        # print(agent_obs.mean(), expert_obs.mean(), policy_d.mean(), expert_d.mean())
        policy_loss = (policy_d + 1).pow(2).mean()
    
        gail_loss = expert_loss + policy_loss
        grad_pen = self.compute_grad_pen(expert_obs, agent_obs) * 0.1

        loss = (gail_loss + grad_pen)

        return loss, expert_loss, policy_loss

    # def predict_reward(self, agent_obs, normalizer):
    #     with torch.no_grad():
    #         self.eval()
    #         agent_obs = normalizer.normalize_torch(agent_obs, self.device) ## expect agent_obs like expert_obs
    #         d = self.amp_linear(self.trunk((agent_obs)))
    #         print(agent_obs[0])
    #         self.train()
    #         return torch.clamp(1 - 0.25 * torch.square(d - 1), min=0)


    def predict_reward(self, agent_obs, num_samples=10, sigma=0.2):
        with torch.no_grad():
            self.eval()
            
            # Normalize input
            # agent_obs = normalizer.normalize_torch(agent_obs, self.device)
            
            # Create noise samples (batch_size, num_samples, obs_dim)
            noise = torch.randn((*agent_obs.shape[:-1], num_samples, agent_obs.shape[-1]), 
                            device=self.device) * sigma
            perturbed_obs = agent_obs.unsqueeze(-2) + noise  # (batch_size, num_samples, obs_dim)
            original_shape = perturbed_obs.shape
            perturbed_obs = perturbed_obs.view(-1, original_shape[-1])
        
            d_all = self.amp_linear(self.trunk(perturbed_obs))
            d_all = d_all.view(*original_shape[:-1])  # (batch_size, num_samples)

            squared_errors = torch.square(d_all - 1)
            # Compute reward
            reward = torch.clamp(1 - 0.25 * squared_errors.min(dim=-1).values, min=0).unsqueeze(-1)
            self.train()
            return reward



def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None