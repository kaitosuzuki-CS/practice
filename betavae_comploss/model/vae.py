import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.decoder import Decoder
from model.encoder import Encoder

parent_dir = Path(__file__).resolve().parent.parent


class VAE(nn.Module):
    def __init__(self, hps):
        super(VAE, self).__init__()

        self._hps = hps

        self.img_size = hps.img_size
        self.latent_features = hps.latent_features

        self.encoder = Encoder(self.img_size, self.latent_features, hps.encoder)
        self.decoder = Decoder(self.img_size, self.latent_features, hps.decoder)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def load_weights(self, path):
        path = os.path.join(parent_dir, path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found at {path}")

        self.load_state_dict(torch.load(path)["model_state_dict"])

    def forward(self, x, eval=False):
        B, C, H, W = x.shape

        x = x.view(B, -1)

        z, mu, logvar = self.encoder(x)
        if eval:
            recon_x = self.decoder(mu)
        else:
            recon_x = self.decoder(z)

        recon_x = recon_x.view(B, C, H, W)

        return recon_x, mu, logvar
