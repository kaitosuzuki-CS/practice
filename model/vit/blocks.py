import torch
import torch.nn as nn

from model.vit import AdaLNViTLayer, LoRAViTLayer


def get_t_emb(t, t_embed_dim):
    factor = 10000 ** (
        (
            torch.arange(start=0, end=t_embed_dim // 2, device=t.device)
            / (t_embed_dim // 2)
        )
    )

    t = t.unsqueeze(-1)
    t_emb = t[:, None].repeat(1, t_embed_dim // 2) / factor
    t_emb = torch.cat((torch.sin(t_emb), torch.cos(t_emb)), dim=-1)

    return t_emb


class LoRAViTBlock(nn.Module):
    def __init__(self, embed_dim, hps):
        super(LoRAViTBlock, self).__init__()

        self._embed_dim = embed_dim
        self._hps = hps
        self._proj_lora = hps.proj_lora

        if self._proj_lora:
            self._t_embed_dim = hps.t_embed_dim
            self.t_proj = nn.Sequential(
                nn.Linear(hps.t_embed_dim, hps.t_embed_dim),
                nn.GELU(),
                nn.Linear(hps.t_embed_dim, hps.t_embed_dim),
            )

        self.layers = nn.ModuleList(
            [
                LoRAViTLayer(
                    embed_dim=embed_dim,
                    hidden_dim=hps.hidden_dim,
                    r=hps.r,
                    alpha=hps.alpha,
                    num_heads=hps.num_heads,
                    bias=hps.bias,
                    t_embed_dim=getattr(hps, "t_embed_dim", 128),  # type: ignore
                    max_numsteps=getattr(hps, "max_numsteps", 8),
                    proj_lora=hps.proj_lora,
                    dropout=hps.dropout,
                )
                for _ in range(hps.num_layers)
            ]
        )

    def init_weights(self):
        if self._proj_lora:
            for m in self.t_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        for layer in self.layers:
            layer.init_weights()  # type: ignore

    def forward(self, x, t):
        if self._proj_lora:
            t = get_t_emb(t, self._t_embed_dim)
            t = self.t_proj(t)

        for layer in self.layers:
            x = layer(x, t)

        return x


class AdaLNViTBlock(nn.Module):
    def __init__(self, embed_dim, hps):
        super(AdaLNViTBlock, self).__init__()

        self._hps = hps
        self._t_embed_dim = hps.t_embed_dim

        self.t_proj = nn.Sequential(
            nn.Linear(hps.t_embed_dim, hps.t_embed_dim),
            nn.GELU(),
            nn.Linear(hps.t_embed_dim, hps.t_embed_dim),
        )

        self.layers = nn.ModuleList(
            [
                AdaLNViTLayer(
                    embed_dim=embed_dim,
                    hidden_dim=hps.hidden_dim,
                    t_embed_dim=hps.t_embed_dim,
                    num_heads=hps.num_heads,
                    dropout=hps.dropout,
                )
                for _ in range(hps.num_layers)
            ]
        )

    def init_weights(self):
        for m in self.t_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for layer in self.layers:
            layer.init_weights()  # type: ignore

    def forward(self, x, t):
        t_emb = get_t_emb(t, self._t_embed_dim)
        t_emb = self.t_proj(t_emb)

        for layer in self.layers:
            x = layer(x, t_emb)

        return x
