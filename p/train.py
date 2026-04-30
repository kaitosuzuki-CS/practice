import argparse
import os
from pathlib import Path

import torch

os.environ["MUJOCO_GL"] = "egl" if torch.cuda.is_available() else "osmesa"
from dm_control import suite

from PPO import PPO
from utils import load_config

parent_dir = Path(__file__).parent
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_environment(domain_name, task_name):
    env = suite.load(domain_name=domain_name, task_name=task_name)

    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain_name",
        type=str,
        required=False,
        default="cheetah",
        help="Domain name for the environment",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=False,
        default="run",
        help="Task name for the environment",
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the config file"
    )

    args = parser.parse_args()
    domain_name = args.domain_name
    task_name = args.task_name
    config_path = os.path.join(parent_dir, args.config_path)

    hps = load_config(config_path)
    env = create_environment(domain_name, task_name)

    ppo = PPO(env, hps.model_hps, hps.train_hps, device)  # type: ignore
    ppo.train()
