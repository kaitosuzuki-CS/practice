import torch
import torch.nn as nn

from .adaln import AdaLN


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        c_emb_dim,
        kernel_size,
        stride,
        padding,
        num_groups,
    ):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._t_emb_dim = t_emb_dim
        self._c_emb_dim = c_emb_dim
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._num_groups = num_groups

        self.adaln1 = AdaLN(
            embed_dim=in_channels,
            t_emb_dim=t_emb_dim,
            c_emb_dim=c_emb_dim,
            with_gate=False,
        )
        self.norm1 = nn.GroupNorm(
            num_groups=num_groups if in_channels % num_groups == 0 else in_channels,
            num_channels=in_channels,
        )
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.adaln2 = AdaLN(
            embed_dim=out_channels,
            t_emb_dim=t_emb_dim,
            c_emb_dim=c_emb_dim,
            with_gate=False,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=num_groups if out_channels % num_groups == 0 else out_channels,
            num_channels=out_channels,
        )
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.residual_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def init_weights(self):
        for layer in [
            self.norm1,
            self.conv1,
            self.norm2,
            self.conv2,
            self.residual_conv,
        ]:
            for m in layer.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.xavier_uniform_(m.weight)

                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                if isinstance(m, nn.GroupNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        self.adaln1.init_weights()
        self.adaln2.init_weights()

    def forward(self, x, t_emb, c_emb):
        """
        Args:
            x: (B, in_channels, H, W)
            t_emb: (B, t_emb_dim)
            c_emb: (B, c_emb_dim)

        Returns:
            (B, out_channels, H, W)
        """

        residual = x

        shift_m1, scale_m1 = 0, 0
        if t_emb is not None and c_emb is not None:
            m1 = self.adaln1(t_emb, c_emb)  # B, 2 * C
            shift_m1, scale_m1 = (
                m1.unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
            )  # B, C, 1, 1

        x = self.norm1(x) * (1 + scale_m1) + shift_m1
        x = self.act1(x)
        x = self.conv1(x)

        shift_m2, scale_m2 = 0, 0
        if t_emb is not None and c_emb is not None:
            m2 = self.adaln2(t_emb, c_emb)  # B, 2 * C
            shift_m2, scale_m2 = (
                m2.unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
            )  # B, C, 1, 1

        x = self.norm2(x) * (1 + scale_m2) + shift_m2
        x = self.act2(x)
        x = self.conv2(x)

        residual = self.residual_conv(residual)

        out = x + residual

        return out


class AttentionBlock(nn.Module):
    def __init__(self, channels, t_emb_dim, c_emb_dim, num_heads, num_groups):
        super().__init__()

        self._channels = channels
        self._t_emb_dim = t_emb_dim
        self._c_emb_dim = c_emb_dim
        self._num_heads = num_heads
        self._num_groups = num_groups

        self.adaln = AdaLN(
            embed_dim=channels, t_emb_dim=t_emb_dim, c_emb_dim=c_emb_dim, with_gate=True
        )
        self.norm = nn.GroupNorm(
            num_groups=num_groups if channels % num_groups == 0 else channels,
            num_channels=channels,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )

    def init_weights(self):
        for layer in [self.norm, self.attn]:
            for m in layer.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.xavier_uniform_(m.weight)

                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                if isinstance(m, nn.GroupNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        self.adaln.init_weights()

    def forward(self, x, t_emb, c_emb):
        """
        Args:
            x: (B, channels, H, W)
            t_emb: (B, t_emb_dim)
            c_emb: (B, c_emb_dim)

        Returns:
            (B, channels, H, W)
        """

        B, C, H, W = x.shape

        shift_m, scale_m, gate_m = 0, 0, 1
        if t_emb is not None and c_emb is not None:
            m = self.adaln(t_emb, c_emb)  # B, 3 * C
            shift_m, scale_m, gate_m = m.unsqueeze(1).chunk(3, dim=-1)  # B, 1, C

        _x = self.norm(x)
        _x = _x.reshape(B, C, H * W).transpose(1, 2).contiguous()  # B, H*W, C

        _x = _x * (1 + scale_m) + shift_m
        _x, _ = self.attn(_x, _x, _x)

        _x = _x * gate_m
        _x = _x.transpose(1, 2).reshape(B, C, H, W).contiguous()

        x = x + _x

        return x
