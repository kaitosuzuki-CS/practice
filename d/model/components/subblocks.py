import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        num_groups,
        t_emb_dim,
    ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

        self.t_emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels))

        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

        self.residual_input_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1
        )

    def forward(self, x, t_emb):
        _x = self.conv1(x)
        _x = _x + self.t_emb_layers(t_emb)[:, :, None, None]
        _x = self.conv2(_x)
        x = _x + self.residual_input_conv(x)

        return x


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_groups, num_heads):
        super(AttentionBlock, self).__init__()

        self.norm = nn.GroupNorm(num_groups, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape

        _x = x.reshape(B, C, H * W)
        _x = self.norm(_x)

        _x = _x.transpose(1, 2)
        _x, _ = self.attn(_x, _x, _x)

        _x = _x.transpose(1, 2).reshape(B, C, H, W)
        x = x + _x
        return x
