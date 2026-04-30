import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from model.components.noise_scheduler import LinearNoiseScheduler
from model.main import Model
from utils.dataset import create_dataset
from utils.hps import load_hps
from utils.loss import CustomLoss
from utils.misc import EarlyStopping, set_seeds

parent_dir = Path(__file__).resolve().parent

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, hps, train_loader, val_loader, device):
    save_dir = os.path.join(parent_dir, hps.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.to(device)
    model.init_weights()

    loss_hps = hps.loss
    optimizer_hps = hps.optimizer
    early_stopping_hps = hps.early_stopping
    scheduler_hps = hps.scheduler
    misc_hps = hps.misc

    optimizer = Adam(
        model.parameters(),
        lr=optimizer_hps.params.lr,
        weight_decay=optimizer_hps.params.weight_decay,
    )

    scheduler = None
    if scheduler_hps is not None:
        num_training_steps = hps.num_epochs * len(train_loader)
        num_warmup_steps = int(scheduler_hps.warmup_ratio * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    loss_fn = CustomLoss(nn.MSELoss(), device)

    early_stopping = None
    if early_stopping_hps.patience is not None:
        early_stopping = EarlyStopping(
            early_stopping_hps.patience, early_stopping_hps.tol
        )

    num_timesteps = misc_hps.num_timesteps
    beta_scheduler = LinearNoiseScheduler(
        num_timesteps, misc_hps.beta_start, misc_hps.beta_end
    )

    for epoch in range(1, hps.num_epochs + 1):
        model.train()

        train_loss = 0.0
        for x, _ in tqdm(train_loader, leave=False):
            optimizer.zero_grad()

            x = x.to(device)

            noise = torch.randn_like(x).to(device)
            t = torch.randint(0, num_timesteps, (x.shape[0],)).to(device)

            x_noisy = beta_scheduler.add_noise(x, noise, t)
            noise_pred = model(x_noisy, t)

            loss = loss_fn(noise_pred, noise)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            train_loss += loss.detach().item()

        with torch.no_grad():
            model.eval()

            val_loss = 0.0
            for x, _ in tqdm(val_loader, leave=False):
                x = x.to(device)

                noise = torch.randn_like(x).to(device)
                t = torch.randint(0, num_timesteps, (x.shape[0],)).to(device)

                x_noisy = beta_scheduler.add_noise(x, noise, t)
                noise_pred = model(x_noisy, t)

                loss = loss_fn(noise_pred, noise)

                val_loss += loss.detach().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"----Epoch {epoch}----")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f'LR: {optimizer.param_groups[0]["lr"]}')

        if epoch % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                f"{save_dir}/checkpoint_{epoch}.pth",
            )

        if early_stopping is not None:
            early_stop = early_stopping.step(model, val_loss)

            if early_stop:
                break

    if early_stopping and early_stopping.best_model is not None:
        best_model = early_stopping.best_model
        best_loss = early_stopping.best_loss
    else:
        best_model = model
        best_loss = val_loss

    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "loss": best_loss,
        },
        f"{save_dir}/best_model.pth",
    )

    return best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for DDPM training")
    parser.add_argument("--data", type=str, default="mnist", help="Config file to use")

    args = parser.parse_args()
    dataset = args.data.lower()

    config_file = os.path.join(parent_dir, "config", f"{dataset}_config.json")
    model_hps, train_hps, inference_hps = load_hps(config_file)

    set_seeds(train_hps)

    train_loader, val_loader = create_dataset(dataset, train_hps.data)

    model = Model(model_hps)

    _model = train(model, train_hps, train_loader, val_loader, device)
