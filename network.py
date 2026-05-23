import torch
import torch.nn as nn
from torch.distributions import Beta
import numpy as np


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),     nn.Tanh(),
            nn.Linear(256, 256),     nn.Tanh(),
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(256, n_actions), nn.Softplus())
        self.beta_head  = nn.Sequential(
            nn.Linear(256, n_actions), nn.Softplus())
        self.critic     = nn.Linear(256, 1)

        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        f     = self.shared(x)
        alpha = self.alpha_head(f) + 1.0
        beta  = self.beta_head(f)  + 1.0
        value = self.critic(f)
        return alpha, beta, value

    def get_action(self, obs):
        alpha, beta, value = self(obs)
        dist     = Beta(alpha, beta)
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        action_env    = action.squeeze(0).detach().numpy().copy()
        action_env[0] = action_env[0] * 2.0 - 1.0  # [0,1]→[-1,1]
        return action.squeeze(0), log_prob, value.squeeze(-1), action_env

    def evaluate(self, obs, actions):
        alpha, beta, values = self(obs)
        dist      = Beta(alpha, beta)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy   = dist.entropy().sum(-1)
        return log_probs, values.squeeze(-1), entropy