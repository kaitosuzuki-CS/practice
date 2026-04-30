import argparse
import torch

from models import FlowMatchingCFG
from utils import load_config, create_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Classifier-Free Guidance FLow Matching on MNIST"
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
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to the pt/pth checkpoint file.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=64, help="Number of samples to generate."
    )
    parser.add_argument(
        "--num-timesteps", type=int, default=100, help="Number of integration steps."
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="samples.png",
        help="Path to save the generated samples",
    )

    args = parser.parse_args()
    model_config_path = args.model_config_path
    train_config_path = args.train_config_path
    ckpt_path = args.ckpt_path
    num_samples = args.num_samples
    num_timesteps = args.num_timesteps
    save_path = args.save_path

    hps = load_config(model_config_path)
    train_hps = load_config(train_config_path)

    model = FlowMatchingCFG(hps=hps, train_hps=train_hps, device=device)
    model.infer(
        ckpt_path=ckpt_path,
        num_samples=num_samples,
        num_timesteps=num_timesteps,
        save_path=save_path,
    )
