import argparse

import torch

from model import RecursiveViT
from utils import create_dataset, load_config, set_seeds

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Recursive ViT with LoRA or AdaLN timestep conditioning"
    )
    parser.add_argument(
        "--model-config-path",
        type=str,
        default="configs/model_config.yml",
        help="Path to the model config file.",
    )
    parser.add_argument(
        "--train-config-path",
        type=str,
        default="configs/train_config.yml",
        help="Path to the training config file.",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        choices=["lora", "adaln"],
        help="Training version: (lora/adaln)",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to the pt/pth checkpoint file",
    )

    args = parser.parse_args()
    model_config_path = args.model_config_path
    train_config_path = args.train_config_path
    version = args.version
    ckpt_path = args.ckpt_path

    hps = load_config(model_config_path)
    train_hps = load_config(train_config_path)

    train_loader, val_loader = create_dataset(train_hps.data)  # type:ignore
    model = RecursiveViT(hps, train_hps, train_loader, val_loader, device, version)

    model.infer(ckpt_path)
