from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import VGG16_Weights, vgg16


class VGG_PIPS(nn.Module):
    def __init__(self):
        super(VGG_PIPS, self).__init__()

        vgg_pretrained_features = vgg16(
            weights=VGG16_Weights.IMAGENET1K_FEATURES
        ).features

        self.slice1 = nn.Sequential(*[vgg_pretrained_features[i] for i in range(4)])  # type: ignore
        self.slice2 = nn.Sequential(*[vgg_pretrained_features[i] for i in range(4, 9)])  # type: ignore
        self.slice3 = nn.Sequential(*[vgg_pretrained_features[i] for i in range(9, 16)])  # type: ignore
        self.slice4 = nn.Sequential(*[vgg_pretrained_features[i] for i in range(16, 23)])  # type: ignore
        self.slice5 = nn.Sequential(*[vgg_pretrained_features[i] for i in range(23, 30)])  # type: ignore

        self.resize_and_normalize = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.expand(-1, 3, -1, -1)),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        for p in self.parameters():
            p.requires_grad = False

    def _preprocess(self, x):
        return self.resize_and_normalize(x)

    def vgg_process(self, x):
        slice1_out = self.slice1(x)
        slice2_out = self.slice2(slice1_out)
        slice3_out = self.slice3(slice2_out)
        slice4_out = self.slice4(slice3_out)
        slice5_out = self.slice5(slice4_out)

        vgg_outputs = namedtuple(
            "VGGOutputs", ["slice1", "slice2", "slice3", "slice4", "slice5"]
        )
        return vgg_outputs(slice1_out, slice2_out, slice3_out, slice4_out, slice5_out)

    def forward(self, x_pred, x):
        x_pred = self._preprocess(x_pred)
        x_pred_features = self.vgg_process(x_pred)

        x = self._preprocess(x)
        x_features = self.vgg_process(x)

        loss = 0.0
        for x_pred_feat, x_feat in zip(x_pred_features, x_features):
            x_pred_feat, x_feat = F.normalize(x_pred_feat, dim=1), F.normalize(
                x_feat, dim=1
            )
            loss += torch.mean((x_pred_feat - x_feat) ** 2, dim=(1, 2, 3))

        return loss.mean()  # type: ignore
