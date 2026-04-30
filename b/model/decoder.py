import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components import LinearLayer


class Decoder(nn.Module):
    def __init__(self, img_size, latent_features, hps):
        super(Decoder, self).__init__()

        self._img_size = img_size
        self._latent_features = latent_features
        self._hps = hps

        C, W, H = img_size
        self.in_layer = nn.Sequential(
            nn.Linear(latent_features, hps.hidden_features[0]), nn.LeakyReLU(0.2)
        )

        self.layers = nn.ModuleList(
            [
                LinearLayer(
                    hps.hidden_features[i], hps.hidden_features[i + 1], hps.dropout
                )
                for i in range(len(hps.hidden_features) - 1)
            ]
        )

        self.out_layer = nn.Sequential(
            nn.Linear(hps.hidden_features[-1], C * W * H), nn.Sigmoid()  # type: ignore
        )

    def forward(self, z):
        B, D = z.shape

        x = self.in_layer(z)

        for layer in self.layers:
            x = layer(x)

        x = self.out_layer(x)

        return x
