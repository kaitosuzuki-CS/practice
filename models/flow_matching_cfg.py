import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from models.flow_model import FlowModel
from utils import EarlyStopping, set_seeds
from utils.misc import save_grid

parent_dir = Path(__file__).resolve().parent.parent


class FlowMatchingCFG:
    def __init__(
        self, hps, train_hps, train_loader=None, val_loader=None, device="cpu"
    ):
        self._hps = hps
        self._train_hps = train_hps
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._device = device

        self._init_hyperparameters()

        self.model = FlowModel(hps)

    def _init_hyperparameters(self):
        self.optimizer_hps = self._train_hps.optimizer
        self.scheduler_hps = getattr(self._train_hps, "scheduler", None)
        self.early_stopping_hps = getattr(self._train_hps, "early_stopping", None)

        self.lr = float(self.optimizer_hps.lr)
        self.betas = tuple(
            map(float, getattr(self.optimizer_hps, "betas", (0.9, 0.999)))
        )
        self.weight_decay = float(getattr(self.optimizer_hps, "weight_decay", 0))

        if self.scheduler_hps is not None:
            self.warmup_epochs = int(self.scheduler_hps.warmup_epochs)

        if self.early_stopping_hps is not None:
            self.patience = float(self.early_stopping_hps.patience)
            self.min_delta = float(self.early_stopping_hps.min_delta)

        self.num_epochs = int(self._train_hps.num_epochs)
        self.accum_steps = int(getattr(self._train_hps, "accum_steps", 1))

        self.checkpoint_dir = os.path.join(
            parent_dir, str(getattr(self._train_hps, "checkpoint_dir", "checkpoints"))
        )
        self.checkpoint_freq = int(getattr(self._train_hps, "checkpoint_freq", 10))

        self.samples_dir = os.path.join(
            parent_dir, str(getattr(self._train_hps, "samples_dir", "samples"))
        )

        self.dropout_rate = float(self._train_hps.dropout_rate)
        self.seed = int(getattr(self._train_hps, "seed", 42))

    def _init_training_scheme(self):
        self.optim = Adam(
            params=self.model.parameters(),
            lr=self.lr,
            betas=self.betas,  # type: ignore
            weight_decay=self.weight_decay,
        )

        self.scheduler = None
        if self.scheduler_hps is not None:
            num_warmup_steps = self.warmup_epochs * np.ceil(
                len(self._train_loader) / self.accum_steps
            )
            num_training_steps = self.num_epochs * np.ceil(
                len(self._train_loader) / self.accum_steps
            )
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optim,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        self.early_stopping_hps = None
        if self.early_stopping_hps is not None:
            self.early_stopping = EarlyStopping(
                patience=self.patience, min_delta=self.min_delta
            )

    def _init_weights(self):
        self.model.init_weights()

    def _init_weights_with_ckpt(self, ckpt_path, freeze=True):
        ckpt_path = os.path.join(parent_dir, self.checkpoint_dir, ckpt_path)
        self.model.init_weights_with_ckpt(ckpt_path, freeze)

    def move_to_device(self, device):
        self.model = self.model.to(device)

        print(f"Moved to {device}")

    def train(self):
        assert (
            self._train_loader and self._val_loader
        ), "Need to be initialized with dataloaders"

        set_seeds(self.seed)
        self._init_weights()
        self.move_to_device(self._device)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._init_training_scheme()

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            self.optim.zero_grad(set_to_none=True)

            total_loss = 0.0
            num_batches = 0
            for x1, label in tqdm(self._train_loader, leave=False):
                num_batches += 1
                batch_size = x1.shape[0]

                x1 = x1.to(self._device)
                label = label.to(self._device)

                x0 = torch.randn_like(x1)
                target = x1 - x0

                t = torch.rand(batch_size, device=self._device)
                xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1

                if np.random.uniform() < self.dropout_rate:
                    pred = self.model(xt, t, label, with_condition=False)
                else:
                    pred = self.model(xt, t, label, with_condition=True)

                loss = ((target - pred) ** 2).mean()
                loss = loss / self.accum_steps

                loss.backward()

                if num_batches % self.accum_steps == 0:
                    self.optim.step()
                    self.optim.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step()

                total_loss += loss.item() * self.accum_steps

            if num_batches % self.accum_steps != 0:
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)

                if self.scheduler is not None:
                    self.scheduler.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0

                for x1, label in tqdm(self._val_loader, leave=False):
                    batch_size = x1.shape[0]

                    x1 = x1.to(self._device)
                    label = label.to(self._device)

                    x0 = torch.randn_like(x1)
                    target = x1 - x0

                    t = torch.rand(batch_size, device=self._device)
                    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1

                    if np.random.uniform() < self.dropout_rate:
                        pred = self.model(xt, t, None, with_condition=False)
                    else:
                        pred = self.model(xt, t, label, with_condition=True)

                    loss = ((target - pred) ** 2).mean()
                    val_loss += loss.item()

            train_loss /= len(self._train_loader)
            val_loss /= len(self._val_loader)

            print(f"----Epoch {epoch}----")
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"LR: {self.optim.param_groups[0]['lr']:.6f}")

            if epoch % self.checkpoint_freq == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "scheduler_state_dict": (
                            self.scheduler.state_dict()
                            if self.scheduler is not None
                            else None
                        ),
                        "loss": val_loss,
                    },
                    os.path.join(self.checkpoint_dir, f"checkpoint_{epoch}.pt"),
                )

            if self.early_stopping is not None:
                self.early_stopping(self.model, val_loss)

                if self.early_stopping.stop:
                    if self.early_stopping.best_model is not None:
                        self.model = self.early_stopping.best_model

                    break

        torch.save(
            {"model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoint_dir, "best_model.pt"),
        )

        print("Training complete")

    def infer(
        self,
        ckpt_path,
        num_samples,
        num_timesteps,
        im_size=(32, 32),
        batch_size=None,
        save_path="results.png",
        nrow: int = 8,
        padding: int = 4,
        dpi: int = 300,
    ):
        set_seeds(self.seed)
        self._init_weights_with_ckpt(ckpt_path, freeze=True)
        self.move_to_device(self._device)

        batch_size = batch_size or num_samples

        t = torch.linspace(0, 1, num_timesteps, device=self._device)
        dt = 1 / num_timesteps

        generated_samples = []
        labels = []

        self.model.eval()
        with torch.no_grad():
            for N in range(0, num_samples, batch_size):
                B = min(batch_size, num_samples - N)

                xt = torch.randn(self.model.im_channels, *im_size, device=self._device)
                label = torch.randint(
                    0, self.model.num_classes, dtype=torch.long, device=self._device
                )

                for _t in tqdm(t, leave=False):
                    _t = _t.expand(B)
                    pred_with_c = self.model(xt, _t, label, with_condition=True)
                    pred_without_c = self.model(xt, _t, None, with_condition=False)

                    pred = (
                        (1 + self._hps.beta) * pred_with_c
                    ) - self._hps.beta * pred_without_c
                    xt = xt + dt * pred

                generated_samples.append(xt)
                labels.append(label)

        generated_samples = torch.cat(generated_samples, dim=0)
        generated_samples = generated_samples.detach().cpu()

        labels = torch.cat(labels, dim=0)
        labels = labels.detach().cpu()

        if save_path is not None:
            save_path = os.path.join(parent_dir, self.samples_dir, save_path)
            save_grid(generated_samples, labels, save_path, nrow, padding, dpi)

        return generated_samples, labels
