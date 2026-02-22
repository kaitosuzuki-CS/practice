import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from model import AdaLNClassifier, LoRAClassifier
from utils import EarlyStopping, set_seeds

parent_dir = Path(__file__).resolve().parent.parent


class RecursiveViT:
    def __init__(
        self, hps, train_hps, train_loader, val_loader, device, version="lora"
    ):
        assert version in [
            "lora",
            "adaln",
        ], "Need to select a training version: lora or adaln"

        self._hps = hps
        self._train_hps = train_hps
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._device = device

        self._init_hyperparameters()

        if version == "lora":
            self.model = LoRAClassifier(hps)
        else:
            self.model = AdaLNClassifier(hps)

    def _init_hyperparameters(self):
        self.loss_hps = self._train_hps.loss
        self.optimizer_hps = self._train_hps.optimizer
        self.scheduler_hps = getattr(self._train_hps, "scheduler", None)
        self.early_stopping_hps = getattr(self._train_hps, "early_stopping", None)

        self.lr = float(self.optimizer_hps.lr)
        self.betas = tuple(map(float, self.optimizer_hps.betas))
        self.weight_decay = float(getattr(self.optimizer_hps, "weight_decay", 0))

        if self.scheduler_hps is not None:
            self.warmup_epochs = int(self.scheduler_hps.warmup_epochs)

        if self.early_stopping_hps is not None:
            self.patience = int(self.early_stopping_hps.patience)
            self.min_delta = float(self.early_stopping_hps.min_delta)

        self.num_epochs = int(self._train_hps.num_epochs)
        self.accum_steps = int(self._train_hps.accum_steps)
        self.max_numsteps = int(self._train_hps.max_numsteps)
        self.checkpoints_dir = os.path.join(
            parent_dir, str(self._train_hps.checkpoints_dir)
        )
        self.checkpoints_freq = int(self._train_hps.checkpoints_freq)

        self.seed = getattr(self._train_hps, "seed", 42)

    def _init_training_scheme(self):
        optim = Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=self.betas,  # type: ignore
            weight_decay=self.weight_decay,
        )

        scheduler = None
        if self.scheduler_hps is not None:
            num_training_steps = int(
                self.num_epochs * np.ceil(len(self._train_loader) / self.accum_steps)
            )
            num_warmup_steps = int(
                self.warmup_epochs * np.ceil(len(self._train_loader) / self.accum_steps)
            )
            scheduler = get_cosine_schedule_with_warmup(
                optim,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        early_stopping = None
        if self.early_stopping_hps is not None:
            early_stopping = EarlyStopping(
                patience=self.patience, min_delta=self.min_delta
            )

        return optim, scheduler, early_stopping

    def _init_weights(self):
        self.model.init_weights()

    def train(self):
        set_seeds(self.seed)
        self.model = self.model.to(self._device)
        self._init_weights()

        optim, scheduler, early_stopping = self._init_training_scheme()

        os.makedirs(self.checkpoints_dir, exist_ok=True)

        best_model = None
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()

            optim.zero_grad(set_to_none=True)
            train_loss = 0.0
            num_batches = 0
            for x, labels, _ in tqdm(self._train_loader, leave=False):
                num_batches += 1
                x, labels = x.to(self._device), labels.to(self._device)

                t = np.random.randint(1, self.max_numsteps + 1)
                logits = self.model(x, t)

                loss = F.cross_entropy(logits, labels)
                loss = loss / self.accum_steps

                loss.backward()

                if num_batches % self.accum_steps == 0:
                    optim.step()
                    optim.zero_grad(set_to_none=True)

                    if scheduler is not None:
                        scheduler.step()

                train_loss += loss.item() * self.accum_steps

            if num_batches % self.accum_steps != 0:
                optim.step()
                optim.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

            with torch.no_grad():
                self.model.eval()

                val_loss = 0.0
                for x, labels, _ in tqdm(self._val_loader, leave=False):
                    x, labels = x.to(self._device), labels.to(self._device)

                    t = np.random.randint(1, self.max_numsteps + 1)
                    logits = self.model(x, t)

                    loss = F.cross_entropy(logits, labels)

                    val_loss += loss.item()

            train_loss /= len(self._train_loader)
            val_loss /= len(self._val_loader)

            print(f"----Epoch {epoch}----")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"LR: {optim.param_groups[0]['lr']:.6f}")

            if epoch % self.checkpoints_freq == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "scheduler_state_dict": (
                            scheduler.state_dict() if scheduler is not None else None
                        ),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    f"{self.checkpoints_dir}/checkpoint_{epoch}.pt",
                )

            if early_stopping is not None:
                best_model = early_stopping(self.model, val_loss)
                if early_stopping.stop:
                    break

        if best_model is None:
            best_model = self.model

        torch.save(
            {"model_state_dict": best_model.state_dict()},
            f"{self.checkpoints_dir}/final_model.pt",
        )

    def infer(self, ckpt_path):
        ckpt_path = os.path.join(parent_dir, ckpt_path)
        self.model.init_weights_with_ckpt(ckpt_path)

        self.model = self.model.to(self._device)
        self.model.eval()

        with torch.no_grad():
            for t in range(1, self.max_numsteps + 1):
                num_correct = 0
                num_total = 0

                wrong_indices = []
                wrong_labels = []
                wrong_preds = []

                for x, labels, indices in tqdm(self._val_loader, leave=False):
                    x, labels = x.to(self._device), labels.to(self._device)

                    logits = self.model(x, t)
                    preds = logits.argmax(dim=-1)

                    correct_mask = preds == labels

                    num_correct += correct_mask.sum().item()
                    num_total += x.shape[0]

                    batch_wrong = indices[~correct_mask.cpu()]
                    labels_wrong = labels[~correct_mask.cpu()]
                    preds_wrong = preds[~correct_mask.cpu()]

                    wrong_indices.extend(batch_wrong.tolist())
                    wrong_labels.extend(labels_wrong.tolist())
                    wrong_preds.extend(preds_wrong.tolist())

                print(f"----Numstep {t}----")
                print(f"Accuracy: {num_correct / num_total}")
                print(f"Wrong Indices: {wrong_indices}")
                print(f"Wrong Labels: {wrong_labels}")
                print(f"Wrong Predictions: {wrong_preds}")
