import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SimpleFFN(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(SimpleFFN, self).__init__()

        self._input_size = input_dim
        self._output_size = output_dim
        self._device = device

        self.ffn1 = nn.Linear(input_dim, output_dim)
        self.ffn2 = nn.Linear(output_dim, output_dim)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self._device)

        obs = obs.view(-1, self._input_size)

        x = F.relu(self.ffn1(obs))
        x = F.relu(self.ffn2(x))

        return x


class Actor(SimpleFFN):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(Actor, self).__init__(input_dim, hidden_dim, device)

        self.mu = nn.Linear(hidden_dim, output_dim)
        self.logvar = nn.Parameter(torch.zeros(1, output_dim))

        self._init_weights()

    def _init_weights(self):
        super()._init_weights()

        nn.init.orthogonal_(self.mu.weight, 0.01)
        nn.init.zeros_(self.mu.bias)

    def get_dist(self, obs):
        x = super().forward(obs)
        mu = self.mu(x)

        logvar = torch.clamp(self.logvar, -20, 2)
        std = torch.exp(0.5 * logvar)

        return Normal(mu, std)

    def forward(self, obs):
        dist = self.get_dist(obs)

        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        return action, log_prob, action

    def evaluate(self, obs, raw_action):
        dist = self.get_dist(obs)
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class Critic(SimpleFFN):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(Critic, self).__init__(input_dim, hidden_dim, device)

        self.value = nn.Linear(hidden_dim, output_dim)

    def _init_weights(self):
        super()._init_weights()

        nn.init.orthogonal_(self.value.weight, 1.0)

    def forward(self, obs):
        x = super().forward(obs)
        value = self.value(x)

        return value
