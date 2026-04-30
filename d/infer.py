import argparse
import os
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from model.components.noise_scheduler import LinearNoiseScheduler
from model.main import Model
from utils.hps import load_hps
from utils.misc import set_seeds

parent_dir = Path(__file__).resolve().parent

device = "cuda" if torch.cuda.is_available() else "cpu"


def sample(model, scheduler, inference_hps, model_hps, device):
    xt = torch.randn(
        (
            inference_hps.num_samples,
            model_hps.im_channels,
            model_hps.im_size_h,
            model_hps.im_size_w,
        )
    ).to(device)

    save_path = os.path.join(parent_dir, inference_hps.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in tqdm(reversed(range(inference_hps.num_timesteps))):
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        xt, x0_pred = scheduler.sample_prev_timestep(
            xt, noise_pred, torch.as_tensor(i).to(device)
        )

        if i == inference_hps.num_timesteps - 1 or i % 100 == 0:
            ims = torch.clamp(xt, -1.0, 1.0).detach().cpu()
            ims = (ims + 1.0) / 2.0

            grid = make_grid(ims, nrow=inference_hps.num_grid_rows)
            img = transforms.ToPILImage()(grid)

            img.save(os.path.join(save_path, f"x0_{i}.png"))
            img.close()


def infer(model_hps, inference_hps):
    ckpt_path = os.path.join(parent_dir, inference_hps.ckpt_path)

    model = Model(model_hps).to(device)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device)["model_state_dict"]
    )

    model.eval()

    scheduler = LinearNoiseScheduler(
        num_timesteps=inference_hps.num_timesteps,
        beta_start=inference_hps.beta_start,
        beta_end=inference_hps.beta_end,
    )

    with torch.no_grad():
        sample(model, scheduler, inference_hps, model_hps, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for DDPM training")
    parser.add_argument("--data", type=str, default="mnist", help="Config file to use")

    args = parser.parse_args()
    dataset = args.data.lower()

    config_file = os.path.join(parent_dir, "config", f"{dataset}_config.json")
    model_hps, train_hps, inference_hps = load_hps(config_file)

    set_seeds(train_hps)

    infer(model_hps, inference_hps)
