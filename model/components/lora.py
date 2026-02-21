import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        r,
        alpha,
        bias=True,
        t_embed_dim=128,
        max_numsteps=8,
        proj_lora=False,
    ):
        super(LoRALinear, self).__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._r = r
        self._alpha = alpha
        self._bias = bias
        self.t_embed_dim = t_embed_dim
        self._max_numsteps = max_numsteps
        self._proj_lora = proj_lora

        self.weight = nn.Parameter(torch.randn(self._out_features, self._in_features))
        self.bias = nn.Parameter(torch.randn(self._out_features)) if bias else None

        if proj_lora:
            self.proj_lora_lora_B = nn.Linear(t_embed_dim, r * out_features, bias=False)
            self.proj_lora_lora_A = nn.Linear(t_embed_dim, r * in_features, bias=False)
        else:
            self.lora_B = nn.Parameter(torch.zeros(max_numsteps, out_features, r))
            self.lora_A = nn.Parameter(torch.zeros(max_numsteps, r, in_features))

    def init_weights(self):
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        if self._proj_lora:
            nn.init.zeros_(self.proj_lora_lora_B.weight)
            nn.init.zeros_(self.proj_lora_lora_A.weight)

    def forward(self, x, t):
        batch_size, N, D = x.shape

        B = A = None
        if self._proj_lora:
            B = self.proj_lora_lora_B(t).view(self._out_features, self._r)  # type: ignore
            A = self.proj_lora_lora_A(t).view(self._r, self._in_features)  # type: ignore
        else:
            B = self.lora_B[t]  # type: ignore
            A = self.lora_A[t]  # type: ignore

        out = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(x, B @ A)

        return out + (self._alpha / self._r) * lora_out


class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        r,
        alpha,
        bias=True,
        t_embed_dim=128,
        max_numsteps=8,
        proj_lora=False,
        dropout=0,
    ):
        super(LoRALinearLayer, self).__init__()

        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._r = r
        self._alpha = alpha
        self._bias = bias
        self._t_embed_dim = t_embed_dim
        self._max_numsteps = max_numsteps
        self._proj_lora = proj_lora
        self._dropout = dropout

        self.in_layer = LoRALinear(
            in_features=embed_dim,
            out_features=hidden_dim,
            r=r,
            alpha=alpha,
            bias=bias,
            t_embed_dim=t_embed_dim,
            max_numsteps=max_numsteps,
            proj_lora=proj_lora,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj = LoRALinear(
            in_features=hidden_dim,
            out_features=embed_dim,
            r=r,
            alpha=alpha,
            bias=bias,
            t_embed_dim=t_embed_dim,
            max_numsteps=max_numsteps,
            proj_lora=proj_lora,
        )

    def init_weights(self):
        self.in_layer.init_weights()
        self.proj.init_weights()

    def forward(self, x, t):
        out = self.in_layer(x, t)
        out = self.act(out)
        out = self.dropout(out)
        out = self.proj(out, t)

        return out


class LoRAAttentionLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        r,
        alpha,
        num_heads,
        bias=True,
        t_embed_dim=128,
        max_numsteps=8,
        proj_lora=False,
        dropout=0,
    ):
        super(LoRAAttentionLayer, self).__init__()

        self._embed_dim = embed_dim
        self._r = r
        self._alpha = alpha
        self._num_heads = num_heads
        self._head_dim = embed_dim // num_heads
        self._bias = bias
        self._t_embed_dim = t_embed_dim
        self._max_numsteps = max_numsteps
        self._proj_lora = proj_lora
        self._dropout = dropout

        self.q_proj = LoRALinear(
            in_features=embed_dim,
            out_features=embed_dim,
            r=r,
            alpha=alpha,
            bias=bias,
            t_embed_dim=t_embed_dim,
            max_numsteps=max_numsteps,
            proj_lora=proj_lora,
        )
        self.k_proj = LoRALinear(
            in_features=embed_dim,
            out_features=embed_dim,
            r=r,
            alpha=alpha,
            bias=bias,
            t_embed_dim=t_embed_dim,
            max_numsteps=max_numsteps,
            proj_lora=proj_lora,
        )
        self.v_proj = LoRALinear(
            in_features=embed_dim,
            out_features=embed_dim,
            r=r,
            alpha=alpha,
            bias=bias,
            t_embed_dim=t_embed_dim,
            max_numsteps=max_numsteps,
            proj_lora=proj_lora,
        )
        self.dropout = nn.Dropout(dropout)

        self.o_proj = LoRALinear(
            in_features=embed_dim,
            out_features=embed_dim,
            r=r,
            alpha=alpha,
            bias=bias,
            t_embed_dim=t_embed_dim,
            max_numsteps=max_numsteps,
            proj_lora=proj_lora,
        )

    def init_weights(self):
        self.q_proj.init_weights()
        self.k_proj.init_weights()
        self.v_proj.init_weights()
        self.o_proj.init_weights()

    def forward(self, q, k, v, t):
        B, N, D = q.shape

        q = self.q_proj(q, t)
        k = self.k_proj(k, t)
        v = self.v_proj(v, t)

        q_view = q.view(B, N, self._num_heads, self._head_dim)
        q_view = q_view.transpose(1, 2)  # B, num_heads, N, head_dim

        k_view = k.view(B, N, self._num_heads, self._head_dim)
        k_view = k_view.transpose(1, 2)  # B, num_heads, N, head_dim

        v_view = v.view(B, N, self._num_heads, self._head_dim)
        v_view = v_view.transpose(1, 2)  # B, num_heads, N, head_dim

        attn = q_view @ k_view.transpose(-1, -2)  # B, num_heads, N, N
        attn = F.softmax(attn, dim=-1) / np.sqrt(self._embed_dim)
        attn = self.dropout(attn)

        out = attn @ v_view  # B, num_heads, N, head_dim
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        out = self.o_proj(out, t)

        return out, attn
