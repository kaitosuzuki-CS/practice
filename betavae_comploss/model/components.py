import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(LinearLayer, self).__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._dropout = dropout

        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.norm = nn.BatchNorm1d(out_features)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = Swish()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x
