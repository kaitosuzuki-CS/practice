import torch
import torch.nn as nn
from torch.distributions import Normal

from model.components import MLP


class Actor(nn.Module):
    def __init__(self, encoder, hps):
        super(Actor, self).__init__()

        self.encoder = encoder

        self.mlp = MLP(hps.input_dim, hps.hidden_dim, 2 * hps.action_dim)

    def init_weights(self):
        self.mlp.init_weights()

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)

        x = self.mlp(x)

        mu, logvar = x.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)

        normal = Normal(mu, std)

        x_t = normal.rsample()

        y_t = torch.tanh(x_t)
        action = y_t

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mu
