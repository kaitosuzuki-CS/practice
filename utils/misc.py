import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torchvision.utils import make_grid, save_image

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

parent_dir = Path(__file__).resolve().parent.parent


class HPS:
    def __init__(self, hps):
        for k, v in hps.items():
            if isinstance(v, dict):
                setattr(self, k, HPS(v))
            else:
                setattr(self, k, v)


def load_config(config_path):
    config_path = os.path.join(parent_dir, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return HPS(config)


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def print_parameter_count(module):
    print(f"Total Parameters: {sum(p.numel() for p in module.parameters())}")
    print(
        f"Trainable Parameters: {sum(p.numel() for p in module.parameters() if p.requires_grad)}"
    )


def create_grid(images, labels, nrow=8, padding=4):
    """
    Args:
        images: (B, C, H, W); C = 1 or 3
        labels: (B,)
    """

    B, C, H, W = images.shape

    images = images.clamp(0, 1)
    images = images.expand(-1, 3, 1, 1)

    grid = make_grid(images, nrow=nrow, padding=padding, pad_value=1.0)
    np_grid = grid.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np_grid)
    ax.axis("off")

    for i in range(min(images.size(0), labels.size(0))):
        row = i // nrow
        col = i % nrow

        x = padding + col * (W + padding) + (W / 2)
        y = padding + row * (H + padding) + H

        ax.text(
            x,
            y,
            str(labels[i].item()),
            color="black",
            fontsize=2 * padding,
            ha="center",
            va="top",
        )


def plot_grid(images, labels, nrow=8, padding=4):
    create_grid(images, labels, nrow, padding)

    plt.tight_layout()
    plt.show()


def save_grid(images, labels, save_path, nrow=8, padding=4, dpi=300):
    create_grid(images, labels, nrow, padding)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
