import torch
import torch.nn as nn

from model.components import AdaLN, LinearLayer, LoRAAttentionLayer, LoRALinearLayer


class LoRAViTLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        r,
        alpha,
        num_heads,
        bias=True,
        t_embed_dim=128,
        max_numsteps=8,
        learnable=False,
        dropout=0,
    ):
        super(LoRAViTLayer, self).__init__()

        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._r = r
        self._alpha = alpha
        self._num_heads = num_heads
        self._bias = bias
        self._t_embed_dim = t_embed_dim
        self._max_numsteps = max_numsteps
        self._learnable = learnable
        self._dropout = dropout

        self.norm_attn = nn.RMSNorm(embed_dim)
        self.attn = LoRAAttentionLayer(
            embed_dim=embed_dim,
            r=r,
            alpha=alpha,
            num_heads=num_heads,
            bias=bias,
            t_embed_dim=t_embed_dim,
            max_numsteps=max_numsteps,
            learnable=learnable,
            dropout=dropout,
        )
        self.dropout_attn = nn.Dropout(dropout)

        self.norm_ffn = nn.RMSNorm(embed_dim)
        self.ffn = LoRALinearLayer(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            r=r,
            alpha=alpha,
            bias=bias,
            t_embed_dim=t_embed_dim,
            max_numsteps=max_numsteps,
            learnable=learnable,
            dropout=dropout,
        )
        self.dropout_ffn = nn.Dropout(dropout)

    def init_weights(self):
        nn.init.ones_(self.norm_attn.weight)
        nn.init.ones_(self.norm_ffn.weight)

        self.attn.init_weights()
        self.ffn.init_weights()

    def forward(self, x, t):
        B, N, D = x.shape

        _x = self.norm_attn(x)
        _x, attn = self.attn(_x, _x, _x, t)
        x = x + self.dropout_attn(_x)

        _x = self.norm_ffn(x)
        _x = self.ffn(_x, t)
        x = x + self.dropout_ffn(_x)

        return x


class AdaLNViTLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        t_embed_dim,
        num_heads,
        dropout=0,
    ):
        super(AdaLNViTLayer, self).__init__()

        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._t_embed_dim = t_embed_dim
        self._num_heads = num_heads
        self._dropout = dropout

        self.adaln_attn = AdaLN(embed_dim, t_embed_dim, with_gate=True)
        self.norm_attn = nn.RMSNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout_attn = nn.Dropout(dropout)

        self.adaln_ffn = AdaLN(embed_dim, t_embed_dim, with_gate=True)
        self.norm_ffn = nn.RMSNorm(embed_dim)
        self.ffn = LinearLayer(embed_dim, hidden_dim, dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def init_weights(self):
        nn.init.ones_(self.norm_attn.weight)
        nn.init.ones_(self.norm_ffn.weight)

        self.adaln_attn.init_weights()
        self.adaln_ffn.init_weights()
        self.ffn.init_weights()

        for m in self.attn.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, t_emb):
        B, N, D = x.shape

        m1 = self.adaln_attn(t_emb)
        shift_m1, scale_m1, gate_m1 = m1.unsqueeze(1).chunk(3, dim=-1)
        _x = self.norm_attn(x) * (1 + scale_m1) + shift_m1
        _x, attn = self.attn(_x, _x, _x)
        x = x + self.dropout_attn(_x) * gate_m1

        m2 = self.adaln_ffn(t_emb)
        shift_m2, scale_m2, gate_m2 = m2.unsqueeze(1).chunk(3, dim=-1)
        _x = self.norm_ffn(x) * (1 + scale_m2) + shift_m2
        _x = self.ffn(_x)
        x = x + self.dropout_ffn(_x) * gate_m2

        return x
