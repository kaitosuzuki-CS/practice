import torch
import torch.nn as nn

from model.layers.decoder_layer import DecoderLayer


class DecoderBlock(nn.Module):
    def __init__(self, t_emb_dim, hps):
        super(DecoderBlock, self).__init__()

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    in_channels=hps.in_channels[i],
                    out_channels=hps.out_channels[i],
                    kernel_size=hps.kernel_sizes[i],
                    stride=hps.strides[i],
                    padding=hps.paddings[i],
                    num_groups=hps.num_groups,
                    num_layers=hps.num_res_layers,
                    t_emb_dim=t_emb_dim,
                    num_heads=hps.num_heads,
                    upsample=hps.upsample[i],
                )
                for i in range(hps.num_layers)
            ]
        )

    def forward(self, x, t_emb, skip_connections):
        for layer in self.layers:
            skip_connection = skip_connections.pop()
            x = layer(x, t_emb, skip_connection)

        return x
