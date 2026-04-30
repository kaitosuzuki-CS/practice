import torch
import torch.nn as nn

from model.blocks.encoder_block import EncoderBlock


class Encoder(nn.Module):
    def __init__(self, im_channels, t_emb_dim, hps):
        super(Encoder, self).__init__()

        self.in_conv = nn.Conv2d(
            im_channels, hps.in_channels[0], kernel_size=3, stride=1, padding=1
        )

        self.encoder_block = EncoderBlock(t_emb_dim, hps)

    def forward(self, x, t_emb):
        x = self.in_conv(x)

        x, skip_connections = self.encoder_block(x, t_emb)

        return x, skip_connections
