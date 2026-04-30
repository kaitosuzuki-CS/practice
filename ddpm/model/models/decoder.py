import torch
import torch.nn as nn

from model.blocks.decoder_block import DecoderBlock


class Decoder(nn.Module):
    def __init__(self, im_channels, t_emb_dim, hps):
        super(Decoder, self).__init__()

        self.decoder_block = DecoderBlock(t_emb_dim, hps)

        self.output_conv = nn.Sequential(
            nn.GroupNorm(hps.num_groups, hps.out_channels[-1]),
            nn.Conv2d(
                hps.out_channels[-1], im_channels, kernel_size=3, stride=1, padding=1
            ),
        )

    def forward(self, x, t_emb, skip_connections):
        x = self.decoder_block(x, t_emb, skip_connections)

        x = self.output_conv(x)

        return x
