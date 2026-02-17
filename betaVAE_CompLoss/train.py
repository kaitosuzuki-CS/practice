import argparse
import os
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from model.discriminator import PatchDiscriminator
from model.pips import VGG_PIPS
from model.vae import VAE
from training_schemes import train_with_composite, train_without_composite
from utils.dataset import create_dataset
from utils.misc import load_hps, set_seeds

parent_dir = Path(__file__).resolve().parent

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for beta-VAE training")
    parser.add_argument(
        "--model_config", type=str, required=True, help="Path to model config file"
    )
    parser.add_argument(
        "--train_config", type=str, required=True, help="Path to training config file"
    )
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["composite", "simple"],
        required=False,
        help="Training scheme (simple: without adv/pips loss, composite: with adv/pips loss)",
        default="composite",
    )

    args = parser.parse_args()

    model_config_path = args.model_config
    train_config_path = args.train_config
    scheme = args.scheme

    model_hps, train_hps = load_hps(model_config_path, train_config_path)

    train_loader, val_loader = create_dataset(train_hps.data)  # type: ignore

    set_seeds(train_hps)

    if scheme == "composite":
        model = VAE(model_hps.model_config)  # type: ignore
        discriminator = PatchDiscriminator(model_hps.discriminator_config)  # type: ignore
        pips = VGG_PIPS()
        train_with_composite(
            model, discriminator, pips, train_hps, train_loader, device
        )
    else:
        model = VAE(model_hps.model_config)  # type: ignore
        train_without_composite(model, train_hps, train_loader, device)
