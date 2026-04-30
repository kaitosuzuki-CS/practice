import argparse
import os
from pathlib import Path

import torch

from model.vae import VAE
from utils.misc import load_hps, plot_latent_grid

parent_dir = Path(__file__).resolve().parent

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for beta-VAE inference")
    parser.add_argument(
        "--model_config", type=str, required=True, help="Path to model config file"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        required=False,
        help="Grid size of generated samples palette",
        default=8,
    )
    parser.add_argument(
        "--title",
        type=str,
        required=False,
        help="Title of generated samples palette",
        default="Latent Space Grid",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        requierd=False,
        help="Path to save the generated samples palette",
        default="samples",
    )

    args = parser.parse_args()

    model_config_path = args.model_config
    ckpt_path = args.ckpt_path
    grid_size = args.grid_size
    title = args.title
    save_path = args.save_path

    model_hps, _ = load_hps(model_config_path, save_path)

    model = VAE(model_hps.model_config)  # type: ignore
    model.load_weights(ckpt_path)

    plot_latent_grid(
        model, title=title, grid_size=grid_size, save_path=save_path, device=device
    )
