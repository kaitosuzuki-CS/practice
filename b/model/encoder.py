import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components import LinearLayer


class Encoder(nn.Module):
    def __init__(self, img_size, latent_features, hps):
        super(Encoder, self).__init__()

        self._img_size = img_size
        self._latent_features = latent_features
        self._hps = hps

        C, W, H = img_size
        self.in_layer = nn.Sequential(
            nn.Linear(C * W * H, hps.hidden_features[0]), nn.LeakyReLU(0.2)
        )

        self.layers = nn.ModuleList(
            [
                LinearLayer(
                    hps.hidden_features[i],
                    hps.hidden_features[i + 1],
                    hps.dropout,
                )
                for i in range(len(hps.hidden_features) - 1)
            ]
        )

        self.mu = nn.Linear(hps.hidden_features[-1], latent_features)
        self.logvar = nn.Linear(hps.hidden_features[-1], latent_features)

    def _reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)

        return mu + eps * std

    def forward(self, x):
        B, D = x.shape

        x = self.in_layer(x)

        for layer in self.layers:
            x = layer(x)

        mu = self.mu(x)
        logvar = self.logvar(x)

        z = self._reparametrize(mu, logvar)

        return z, mu, logvar
