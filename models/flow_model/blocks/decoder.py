import torch
import torch.nn as nn

from ..components import AttentionBlock, ResidualBlock


class DecoderLayer(nn.Module):
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
        upsample=False,
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
        self._upsample = upsample

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

        if upsample:
            self.upsample_layer = nn.ConvTranspose2d(
                in_channels=in_channels // 2,
                out_channels=in_channels // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        else:
            self.upsample_layer = nn.Identity()

    def init_weights(self):
        for layer in [self.upsample_layer]:
            for m in layer.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.xavier_uniform_(m.weight)

                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                if isinstance(m, nn.GroupNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        self.residual_block.init_weights()
        self.attn_block.init_weights()

        for res_block, attn_block in self.layers:  # type: ignore
            res_block.init_weights()
            attn_block.init_weights()

    def forward(self, x, t_emb, c_emb, skip_connection):
        """
        Args:
            x: (B, in_channels // 2, H_in, W_in)
            t_emb: (B, t_emb_dim)
            c_emb: (B, c_emb_dim)
            skip_connection: (B, C_skip, H_skip, W_skip)

        Returns:
            (B, out_channels, H_out, W_out)
        """

        x = self.upsample_layer(x)

        x = torch.cat([x, skip_connection], dim=1)
        x = self.residual_block(x, t_emb, c_emb)
        x = self.attn_block(x, t_emb, c_emb)

        for res_block, attn_block in self.layers:  # type: ignore
            x = res_block(x, t_emb, c_emb)
            x = attn_block(x, t_emb, c_emb)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, t_emb_dim, c_emb_dim, hps):
        super().__init__()

        self._t_emb_dim = t_emb_dim
        self._c_emb_dim = c_emb_dim
        self._hps = hps

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
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
                    upsample=hps.upsample[i],
                )
                for i in range(len(hps.in_channels))
            ]
        )

    def init_weights(self):
        for layer in self.layers:
            layer.init_weights()  # type: ignore

    def forward(self, x, t_emb, c_emb, skip_connections):
        """
        Args:
            x: (B, C_in, H_in, W_in)
            t_emb: (B, t_emb_dim)
            c_emb: (B, c_emb_dim)
            skip_connections: List of skip connections (B, C_skip, H_skip, W_skip)

        Returns:
            (B, C_out, H_out, W_out)
        """

        for layer in self.layers:
            skip_connection = skip_connections.pop()
            x = layer(x, t_emb, c_emb, skip_connection)

        return x


class Decoder(nn.Module):
    def __init__(self, im_channels, t_emb_dim, c_emb_dim, hps):
        super().__init__()

        self._im_channels = im_channels
        self._t_emb_dim = t_emb_dim
        self._c_emb_dim = c_emb_dim
        self._hps = hps

        self.decoder_block = DecoderBlock(
            t_emb_dim=t_emb_dim, c_emb_dim=c_emb_dim, hps=hps
        )
        self.out_conv = nn.Sequential(
            nn.GroupNorm(
                num_groups=(
                    hps.num_groups
                    if hps.out_channels[-1] % hps.num_groups == 0
                    else hps.out_channels[-1]
                ),
                num_channels=hps.out_channels[-1],
            ),
            nn.Conv2d(
                in_channels=hps.out_channels[-1],
                out_channels=im_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def init_weights(self):
        for layer in [self.out_conv]:
            for m in layer.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.xavier_uniform_(m.weight)

                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                if isinstance(m, nn.GroupNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        self.decoder_block.init_weights()

    def forward(self, x, t_emb, c_emb, skip_connections):
        """
        Args:
            x: (B, C_in, H_in, W_in)
            t_emb: (B, t_emb_dim)
            c_emb: (B, c_emb_dim)
            skip_connections: List of skip connections (B, C_skip, H_skip, W_skip)

        Returns:
            (B, im_channels, H_out, W_out)
        """

        x = self.decoder_block(x, t_emb, c_emb, skip_connections)
        x = self.out_conv(x)

        return x
