import torch
import torch.nn as nn

from model.components.subblocks import AttentionBlock, ResidualBlock


class DecoderLayer(nn.Module):
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
        upsample,
    ):
        super(DecoderLayer, self).__init__()

        self.upsample = upsample

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

        if self.upsample:
            self.upsample_conv = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=4, stride=2, padding=1
            )
        else:
            self.upsample_conv = nn.Identity()

    def forward(self, x, t_emb, skip_connection):
        x = self.upsample_conv(x)

        x = torch.cat([x, skip_connection], dim=1)
        x = self.residual_block(x, t_emb)
        x = self.attn_block(x)

        for layer in self.layers:
            x = layer[0](x, t_emb)
            x = layer[1](x)

        return x
