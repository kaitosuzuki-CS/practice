import torch
import torch.nn as nn

from model.components.subblocks import AttentionBlock, ResidualBlock


class EncoderLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        num_groups,
        num_layers,
        t_emb_dim,
        num_heads,
        downsample,
    ):
        super(EncoderLayer, self).__init__()

        self.downsample = downsample

        self.residual_block = ResidualBlock(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            num_groups,
            t_emb_dim,
        )
        self.attn_block = AttentionBlock(out_channels, num_groups, num_heads)

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResidualBlock(
                            out_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            padding,
                            num_groups,
                            t_emb_dim,
                        ),
                        AttentionBlock(out_channels, num_groups, num_heads),
                    ]
                )
                for i in range(num_layers - 1)
            ]
        )

        if self.downsample:
            self.downsample_conv = nn.Conv2d(
                out_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        else:
            self.downsample_conv = nn.Identity()

    def forward(self, x, t_emb):
        x = self.residual_block(x, t_emb)
        x = self.attn_block(x)

        for layer in self.layers:
            x = layer[0](x, t_emb)
            x = layer[1](x)

        x = self.downsample_conv(x)

        return x
