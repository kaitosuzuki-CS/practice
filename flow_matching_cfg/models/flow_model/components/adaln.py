import torch
import torch.nn as nn


class AdaLN(nn.Module):
    def __init__(self, embed_dim, t_emb_dim, c_emb_dim, with_gate=False):
        super().__init__()

        self._embed_dim = embed_dim
        self._t_emb_dim = t_emb_dim
        self._c_emb_dim = c_emb_dim
        self._with_gate = with_gate

        self.act = nn.SiLU()
        self.proj = nn.Linear(
            in_features=t_emb_dim + c_emb_dim,
            out_features=3 * embed_dim if with_gate else 2 * embed_dim,
        )

    def init_weights(self):
        for layer in [self.proj]:
            for m in layer.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.zeros_(m.weight)

                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, t_emb, c_emb):
        """
        Args:
            t_emb: (B, t_emb_dim)
            c_emb: (B, c_emb_dim)

        Returns:
            (B, 2 * embed_dim) if not with_gate else (B, 3 * embed_dim)
        """

        emb = torch.cat([t_emb, c_emb], dim=-1)

        emb = self.act(emb)
        emb = self.proj(emb)

        return emb
