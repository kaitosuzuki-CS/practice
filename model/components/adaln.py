import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLN(nn.Module):
    def __init__(self, embed_dim, t_embed_dim, with_gate=True):
        super(AdaLN, self).__init__()

        self._embed_dim = embed_dim
        self._t_embed_dim = t_embed_dim
        self._with_gate = with_gate

        self.act = nn.GELU()

        if with_gate:
            self.proj = nn.Linear(t_embed_dim, 3 * embed_dim)
        else:
            self.proj = nn.Linear(t_embed_dim, 2 * embed_dim)

    def init_weights(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        out = self.act(x)
        out = self.proj(out)

        return out


class LinearLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout):
        super(LinearLayer, self).__init__()

        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._dropout = dropout

        self.in_layer = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.in_layer(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.proj(out)

        return out
