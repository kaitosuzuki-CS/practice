import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Bottleneck, Decoder, Encoder


class FlowModel(nn.Module):
    def __init__(self, hps):
        super().__init__()

        self._hps = hps

        self.t_emb_dim = hps.t_emb_dim
        self.c_emb_dim = hps.c_emb_dim
        self.num_classes = hps.num_classes

        self.t_proj = nn.Sequential(
            nn.Linear(in_features=hps.t_emb_dim, out_features=hps.t_emb_dim),
            nn.SiLU(),
            nn.Linear(in_features=hps.t_emb_dim, out_features=hps.t_emb_dim),
        )

        self.c_proj = nn.Sequential(
            nn.Linear(in_features=hps.num_classes, out_features=hps.c_emb_dim),
            nn.SiLU(),
            nn.Linear(in_features=hps.c_emb_dim, out_features=hps.c_emb_dim),
        )

        self.encoder = Encoder(
            im_channels=hps.im_channels,
            t_emb_dim=hps.t_emb_dim,
            c_emb_dim=hps.c_emb_dim,
            hps=hps.encoder,
        )
        self.bottleneck = Bottleneck(
            t_emb_dim=hps.t_emb_dim, c_emb_dim=hps.c_emb_dim, hps=hps.bottleneck
        )
        self.decoder = Decoder(
            im_channels=hps.im_channels,
            t_emb_dim=hps.t_emb_dim,
            c_emb_dim=hps.c_emb_dim,
            hps=hps.decoder,
        )

    def _get_t_emb(self, t, max_positions=10000):
        t = t * max_positions
        half_dim = self.t_emb_dim // 2
        emb = np.log(max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(t.device)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        if self.t_emb_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb)[:, :1]], dim=1)

        return emb

    def _get_c_emb(self, c):
        return F.one_hot(c, num_classes=self.num_classes).float()

    def forward(self, x, t, c, with_condition=True):
        """
        Args:
            x: (B, im_channels, H, W)
            t: (B,) normalized to [0, 1)
            c: (B,) integer class labels
            with_condition (bool, optional): Whether to use class conditioning. Defaults to True.

        Returns:
            (B, im_channels, H, W)
        """

        B, C, H, W = x.shape

        t_emb = self._get_t_emb(t)
        t_emb = self.t_proj(t_emb)

        if with_condition:
            c_emb = self._get_c_emb(c)
            c_emb = self.c_proj(c_emb)
        else:
            c_emb = torch.zeros(B, self.c_emb_dim, device=x.device)

        x, skip_connections = self.encoder(x, t_emb, c_emb)
        x = self.bottleneck(x, t_emb, c_emb)
        x = self.decoder(x, t_emb, c_emb, skip_connections)

        return x
