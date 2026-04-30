import torch
import torch.nn as nn

from model.components.subblocks import AttentionBlock, ResidualBlock


class BottleneckLayer(nn.Module):
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
    ):
        super(BottleneckLayer, self).__init__()

        self.residual_block1 = ResidualBlock(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            num_groups,
            t_emb_dim,
        )

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        AttentionBlock(out_channels, num_groups, num_heads),
                        ResidualBlock(
                            out_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            padding,
                            num_groups,
                            t_emb_dim,
                        ),
                    ]
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x, t_emb):
        x = self.residual_block1(x, t_emb)

        for layer in self.layers:
            x = layer[0](x)
            x = layer[1](x, t_emb)

        return x
