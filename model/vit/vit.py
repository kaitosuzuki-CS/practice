import torch
import torch.nn as nn

from model.components import Embedding
from model.vit import AdaLNViTBlock, LoRAViTBlock


class LoRAViT(nn.Module):
    def __init__(self, embed_dim, hps):
        super(LoRAViT, self).__init__()

        self._embed_dim = embed_dim
        self._hps = hps
        self._learnable = hps.block.learnable

        self.embedding = Embedding(embed_dim, hps.embedding)
        self.block = LoRAViTBlock(embed_dim, hps.block)

    def init_weights(self):
        self.embedding.init_weights()
        self.block.init_weights()

    def forward(self, x, num_steps):
        x, _ = self.embedding(x)

        B, N, D = x.shape
        t = (
            torch.linspace(0, 1, num_steps, device=x.device)
            if self._learnable
            else range(num_steps)
        )
        for _t in t:
            x = self.block(x, _t)

        cls_token = x[:, 0, :]

        return x, cls_token


class AdaLNViT(nn.Module):
    def __init__(self, embed_dim, hps):
        super(AdaLNViT, self).__init__()

        self._embed_dim = embed_dim
        self._hps = hps

        self.embedding = Embedding(embed_dim, hps.embedding)
        self.block = AdaLNViTBlock(embed_dim, hps.block)

    def init_weights(self):
        self.embedding.init_weights()
        self.block.init_weights()

    def forward(self, x, num_steps):
        x, _ = self.embedding(x)

        B, N, D = x.shape
        t = torch.linspace(0, 1, num_steps, device=x.device)
        for _t in t:
            x = self.block(x, _t)

        cls_token = x[:, 0, :]

        return x, cls_token
