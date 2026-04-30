import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.in_linear = nn.Linear(input_dim, hidden_dim)
        self.in_relu = nn.ReLU(inplace=True)

        self.out_linear = nn.Linear(hidden_dim, output_dim)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.in_linear(x)
        x = self.in_relu(x)

        return self.out_linear(x)


class Encoder(nn.Module):
    def __init__(self, observation_shape, hps):
        super(Encoder, self).__init__()

        self.in_conv = nn.Conv2d(
            observation_shape[0], hps.latent_dim, kernel_size=3, stride=2, padding=1
        )
        self.in_relu = nn.ReLU(inplace=True)

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        hps.latent_dim,
                        hps.latent_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                )
                for _ in range(hps.num_layers - 1)
            ]
        )

        self.flatten = nn.Flatten()

        self.mlp = MLP(
            hps.latent_dim * (hps.input_shape // 2) ** 2,
            hps.hidden_dim,
            hps.output_dim,
        )

        self.output_layer = nn.Sequential(nn.LayerNorm(hps.output_dim), nn.Tanh())

    def init_weights(self):
        self.mlp.init_weights()

        nn.init.xavier_uniform_(self.in_conv.weight)
        if self.in_conv.bias is not None:
            nn.init.zeros_(self.in_conv.bias)

        for m in self.layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.output_layer.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.in_relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.flatten(x)

        x = self.mlp(x)

        return self.output_layer(x)
