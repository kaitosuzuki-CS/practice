import torch
import torch.nn as nn

from model.components import MLP


class SoftCritic(nn.Module):
    def __init__(self, encoder, hps):
        super(SoftCritic, self).__init__()

        self.encoder = encoder

        self.mlp = MLP(hps.input_dim, hps.hidden_dim, 1)

    def init_weights(self):
        self.mlp.init_weights()

    def forward(self, x, a):
        x = self.encoder(x)

        x = torch.cat([x, a], dim=-1)
        x = self.mlp(x)

        return x
