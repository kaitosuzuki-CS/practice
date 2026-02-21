import torch
import torch.nn as nn

from .vit import AdaLNViT, LoRAViT


class LoRAClassifier(nn.Module):
    def __init__(self, hps):
        super(LoRAClassifier, self).__init__()

        self._hps = hps
        self._embed_dim = hps.embed_dim

        self.vit = LoRAViT(hps.embed_dim, hps)
        self.classifier = nn.Sequential(
            nn.RMSNorm(hps.embed_dim), nn.Linear(hps.embed_dim, hps.num_classes)
        )

    def init_weights(self):
        self.vit.init_weights()

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.RMSNorm):
                nn.init.ones_(m.weight)

        print(f"Total Parameters: {sum(p.numel() for p in self.parameters())}")
        print(
            f"Trainable Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def init_weights_with_ckpt(self, ckpt_path):
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(ckpt_path)["model_state_dict"]
        )

        print(f"Missing Keys: {missing_keys}")
        print(f"Unexpected Keys: {unexpected_keys}")

    def forward(self, x, num_steps):
        B, C, H, W = x.shape

        x, cls_token = self.vit(x, num_steps)
        logits = self.classifier(cls_token)

        return logits


class AdaLNClassifier(nn.Module):
    def __init__(self, hps):
        super(AdaLNClassifier, self).__init__()

        self._hps = hps
        self._embed_dim = hps.embed_dim

        self.vit = AdaLNViT(hps.embed_dim, hps)
        self.classifier = nn.Sequential(
            nn.RMSNorm(hps.embed_dim), nn.Linear(hps.embed_dim, hps.num_classes)
        )

    def init_weights(self):
        self.vit.init_weights()

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.RMSNorm):
                nn.init.ones_(m.weight)

        print(f"Total Parameters: {sum(p.numel() for p in self.parameters())}")
        print(
            f"Trainable Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def init_weights_with_ckpt(self, ckpt_path):
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(ckpt_path)["model_state_dict"]
        )

        print(f"Missing Keys: {missing_keys}")
        print(f"Unexpected Keys: {unexpected_keys}")

    def forward(self, x, num_steps):
        B, C, H, W = x.shape

        x, cls_token = self.vit(x, num_steps)
        logits = self.classifier(cls_token)

        return logits
