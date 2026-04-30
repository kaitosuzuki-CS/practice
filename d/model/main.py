import torch
import torch.nn as nn

from model.components.positional_embedding import get_time_embedding
from model.models.bottleneck import Bottleneck
from model.models.decoder import Decoder
from model.models.encoder import Encoder


class Model(nn.Module):
    def __init__(self, hps):
        super(Model, self).__init__()

        self.im_channels = hps.im_channels
        self.t_emb_dim = hps.t_emb_dim

        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
        )

        self.encoder = Encoder(self.im_channels, self.t_emb_dim, hps.encoder)
        self.bottleneck = Bottleneck(self.t_emb_dim, hps.bottleneck)
        self.decoder = Decoder(self.im_channels, self.t_emb_dim, hps.decoder)

    def init_weights(self):
        for m in self.modules():
            if (
                isinstance(m, nn.Linear)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print(f"Total Parameters: {sum(p.numel() for p in self.parameters())}")
        print(
            f"Trainable Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def forward(self, x, t):
        t_emb = get_time_embedding(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        x, skip_connections = self.encoder(x, t_emb)
        x = self.bottleneck(x, t_emb)
        x = self.decoder(x, t_emb, skip_connections)

        return x
