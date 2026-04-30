import copy
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

parent_dir = Path(__file__).resolve().parent.parent


class HPS:
    def __init__(self, hps):
        for k, v in hps.items():
            if isinstance(v, dict):
                setattr(self, k, HPS(v))
            else:
                setattr(self, k, v)


def load_hps(model_config_path=None, train_config_path=None):
    model_hps = train_hps = None
    if model_config_path is not None:
        model_config_path = os.path.join(parent_dir, model_config_path)

        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model config file not found")

        with open(model_config_path, "r") as f:
            model_hps = json.load(f)
            model_hps = HPS(model_hps)

    if train_config_path is not None:
        train_config_path = os.path.join(parent_dir, train_config_path)

        if not os.path.exists(train_config_path):
            raise FileNotFoundError(f"Train config file not found")

        with open(train_config_path, "r") as f:
            train_hps = json.load(f)
            train_hps = HPS(train_hps)

    return model_hps, train_hps


def plot_latent_grid(model, title="Title", grid_size=8, save_path=None, device="cpu"):
    model.to(device)
    model.eval()

    latent_range = np.linspace(-3, 3, grid_size)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))

    for i, val_y in enumerate(latent_range):
        for j, val_x in enumerate(latent_range):
            z = (
                torch.tensor([val_x, val_y], dtype=torch.float32)
                .to(device)
                .unsqueeze(0)
            )
            with torch.no_grad():
                img = model.decoder(z).cpu().squeeze(0)
                img = img.reshape(1, 28, 28)

            if img.shape[0] == 1:
                img = img.squeeze(0)

            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].axis("off")

    fig.suptitle(title, fontsize=14, y=0.92)

    fig.text(0.5, 0.04, "Latent Dimension 1", ha="center")
    fig.text(0.04, 0.5, "Latent Dimension 2", va="center", rotation="vertical")

    plt.tight_layout(rect=(0.05, 0.05, 1, 0.94))

    if save_path is not None:
        save_path = os.path.join(parent_dir, save_path)
        plt.savefig(save_path)

    plt.show()
    plt.close(fig)


class EarlyStopping:
    def __init__(self, patience, tol):
        self.patience = patience
        self.tol = tol
        self.counter = 0

        self.best_loss = float("inf")
        self.best_model = None

    def step(self, model, loss):
        if loss < self.best_loss - self.tol:
            self.best_loss = loss
            self.counter = 0
            self.best_model = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return self.best_model, self.best_loss

        return None, None


def set_seeds(hps):
    seed = hps.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
