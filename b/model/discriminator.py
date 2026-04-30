import torch
import torch.nn as nn

from model.components import ConvLayer, Swish


class PatchDiscriminator(nn.Module):
    def __init__(self, hps):
        super(PatchDiscriminator, self).__init__()

        self._hps = hps

        self.in_layer = nn.Sequential(
            nn.Conv2d(hps.im_channels, hps.hidden_channels[0], 3, 1, 1), Swish()
        )

        self.layers = nn.ModuleList(
            [
                ConvLayer(
                    in_channels=hps.hidden_channels[i],
                    out_channels=hps.hidden_channels[i + 1],
                    kernel_size=hps.kernel_size[i],
                    stride=hps.stride[i],
                    padding=hps.padding[i],
                )
                for i in range(len(hps.hidden_channels) - 1)
            ]
        )

        self.out_layer = nn.Conv2d(
            hps.hidden_channels[-1], 1, kernel_size=1, stride=1, padding=0
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.in_layer(x)

        for layer in self.layers:
            x = layer(x)

        x = self.out_layer(x)

        return x
