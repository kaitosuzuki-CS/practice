import torch
import torch.nn as nn

from ..components import AttentionBlock, ResidualBlock


class EncoderLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        c_emb_dim,
        kernel_size,
        stride,
        padding,
        num_heads,
        num_groups,
        num_layers,
        downsample=False,
    ):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._t_emb_dim = t_emb_dim
        self._c_emb_dim = c_emb_dim
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._num_heads = num_heads
        self._num_groups = num_groups
        self._num_layers = num_layers
        self._downsample = downsample

        self.residual_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            t_emb_dim=t_emb_dim,
            c_emb_dim=c_emb_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_groups=num_groups,
        )
        self.attn_block = AttentionBlock(
            channels=out_channels,
            t_emb_dim=t_emb_dim,
            c_emb_dim=c_emb_dim,
            num_heads=num_heads,
            num_groups=num_groups,
        )

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResidualBlock(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            t_emb_dim=t_emb_dim,
                            c_emb_dim=c_emb_dim,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            num_groups=num_groups,
                        ),
                        AttentionBlock(
                            channels=out_channels,
                            t_emb_dim=t_emb_dim,
                            c_emb_dim=c_emb_dim,
                            num_heads=num_heads,
                            num_groups=num_groups,
                        ),
                    ]
                )
                for _ in range(num_layers)
            ]
        )

        if downsample:
            self.downsample_layer = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        else:
            self.downsample_layer = nn.Identity()

    def forward(self, x, t_emb, c_emb):
        """
        Args:
            x: (B, in_channels, H_in, W_in)
            t_emb: (B, t_emb_dim)
            c_emb: (B, c_emb_dim)

        Returns:
            (B, out_channels, H_out, W_out)
        """

        x = self.residual_block(x, t_emb, c_emb)
        x = self.attn_block(x, t_emb, c_emb)

        for res_block, attn_block in self.layers:  # type: ignore
            x = res_block(x, t_emb, c_emb)
            x = attn_block(x, t_emb, c_emb)

        x = self.downsample_layer(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, t_emb_dim, c_emb_dim, hps):
        super().__init__()

        self._t_emb_dim = t_emb_dim
        self._c_emb_dim = c_emb_dim
        self._hps = hps

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    in_channels=hps.in_channels[i],
                    out_channels=hps.out_channels[i],
                    t_emb_dim=t_emb_dim,
                    c_emb_dim=c_emb_dim,
                    kernel_size=hps.kernel_size[i],
                    stride=hps.stride[i],
                    padding=hps.padding[i],
                    num_heads=hps.num_heads,
                    num_groups=hps.num_groups,
                    num_layers=hps.num_layers,
                    downsample=hps.downsample[i],
                )
                for i in range(len(hps.in_channels))
            ]
        )

    def forward(self, x, t_emb, c_emb):
        """
        Args:
            x: (B, C_in, H_in, W_in)
            t_emb: (B, t_emb_dim)
            c_emb: (B, c_emb_dim)

        Returns:
            x: (B, C_out, H_out, W_out)
            skip_connections: List of skip connections (B, C_skip, H_skip, W_skip)
        """

        skip_connections = []

        for layer in self.layers:
            skip_connections.append(x)
            x = layer(x, t_emb, c_emb)

        return x, skip_connections


class Encoder(nn.Module):
    def __init__(self, im_channels, t_emb_dim, c_emb_dim, hps):
        super().__init__()

        self._im_channels = im_channels
        self._t_emb_dim = t_emb_dim
        self._c_emb_dim = c_emb_dim
        self._hps = hps

        self.in_conv = nn.Conv2d(
            in_channels=im_channels,
            out_channels=hps.in_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.encoder_block = EncoderBlock(
            t_emb_dim=t_emb_dim, c_emb_dim=c_emb_dim, hps=hps
        )

    def forward(self, x, t_emb, c_emb):
        """
        Args:
            x: (B, im_channels, H_in, W_in)
            t_emb: (B, t_emb_dim)
            c_emb: (B, c_emb_dim)

        Returns:
            x: (B, C_out, H_out, W_out)
            skip_connections: List of skip connections (B, C_skip, H_skip, W_skip)
        """

        x = self.in_conv(x)
        x, skip_connections = self.encoder_block(x, t_emb, c_emb)

        return x, skip_connections
