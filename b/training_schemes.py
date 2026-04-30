import os
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from utils.loss import CompositeLoss, SimpleLoss
from utils.misc import EarlyStopping

parent_dir = Path(__file__).resolve().parent


def train_with_composite(model, discriminator, pips, train_hps, train_loader, device):
    model.to(device)
    model.init_weights()

    discriminator.to(device)
    discriminator.init_weights()

    pips.to(device)

    loss_hps = train_hps.loss
    optimizer_hps = train_hps.optimizer
    early_stopping_hps = train_hps.early_stopping
    scheduler_hps = train_hps.scheduler

    optimizer_g = Adam(
        model.parameters(),
        lr=optimizer_hps.params_g.lr,
        weight_decay=optimizer_hps.params_g.weight_decay,
    )
    optimizer_d = Adam(discriminator.parameters(), lr=optimizer_hps.params_d.lr)

    scheduler_g = scheduler_d = None
    num_training_steps = train_hps.num_epochs * len(train_loader)
    if scheduler_hps.params_g is not None:
        num_warmup_steps = int(scheduler_hps.params_g.warmup_ratio * num_training_steps)
        scheduler_g = get_cosine_schedule_with_warmup(
            optimizer_g,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    if scheduler_hps.params_d is not None:
        num_warmup_steps = int(scheduler_hps.params_d.warmup_ratio * num_training_steps)
        scheduler_d = get_cosine_schedule_with_warmup(
            optimizer_d,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    disc_warmup_steps = int(optimizer_hps.params_d.warmup_ratio * num_training_steps)

    loss_fn = CompositeLoss(
        loss_hps.params, discriminator, pips, len(train_loader), device
    )

    early_stopping = None
    if early_stopping_hps is not None:
        early_stopping = EarlyStopping(
            early_stopping_hps.patience, early_stopping_hps.tol
        )

    save_dir = os.path.join(parent_dir, train_hps.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    best_model = None
    num_updates = 0
    for epoch in range(1, train_hps.num_epochs + 1):
        model.train()

        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        train_pips_loss = 0.0
        train_disc_loss_g = 0.0
        train_disc_loss_d = 0.0
        for x, _ in tqdm(train_loader, leave=False):
            x = x.to(device)

            out, mu, logvar = model(x)
            g_loss, recon_loss, kl_loss, pips_loss = loss_fn(out, x, mu, logvar)

            if num_updates >= disc_warmup_steps:
                disc_loss_g = loss_fn.discriminator_loss(out, ones=True)
                g_loss += loss_fn.lambda_disc * disc_loss_g

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            if scheduler_g is not None:
                scheduler_g.step()

            if num_updates >= disc_warmup_steps:
                disc_fake_loss = loss_fn.discriminator_loss(out.detach(), ones=False)
                disc_real_loss = loss_fn.discriminator_loss(x, ones=True)
                disc_loss_d = (disc_fake_loss + disc_real_loss) / 2

                optimizer_d.zero_grad()
                disc_loss_d.backward()
                optimizer_d.step()

                if scheduler_d is not None:
                    scheduler_d.step()

            train_loss += g_loss.detach().item()
            train_recon_loss += recon_loss.detach().item()
            train_kl_loss += kl_loss.detach().item()
            train_pips_loss += pips_loss.detach().item()

            if num_updates >= disc_warmup_steps:
                train_disc_loss_g += disc_loss_g.detach().item()
                train_disc_loss_d += disc_loss_d.detach().item()

            loss_fn.step()

            num_updates += 1

        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)
        train_pips_loss /= len(train_loader)
        train_disc_loss_g /= len(train_loader)
        train_disc_loss_d /= len(train_loader)

        print(f"----Epoch {epoch}----")
        print(
            f"Train Loss: {train_loss:.6f}, Recon: {train_recon_loss:.6f}, KL: {train_kl_loss:.6f}, PIPS: {train_pips_loss:.6f}"
        )
        print(f"Disc (G): {train_disc_loss_g:.6f}, Disc (D): {train_disc_loss_d:.6f}")
        print(
            f'LR (G): {optimizer_g.param_groups[0]["lr"]}, LR (D): {optimizer_d.param_groups[0]['lr']}, Beta: {loss_fn.beta}'
        )

        if epoch % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "decoder_state_dict": model.decoder.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                    "scheduler_g_state_dict": (
                        scheduler_g.state_dict() if scheduler_g is not None else None
                    ),
                    "scheduler_d_state_dict": (
                        scheduler_d.state_dict() if scheduler_d is not None else None
                    ),
                    "loss": train_loss,
                },
                f"{save_dir}/checkpoint_{epoch}.pth",
            )

        if early_stopping is not None:
            best_model, best_loss = early_stopping.step(model, train_loss)
            if best_model is not None:
                torch.save(
                    {
                        "model_state_dict": best_model,
                        "loss": best_loss,
                    },
                    f"{save_dir}/best_model.pth",
                )

                return best_model

    if best_model is None:
        best_model = model.state_dict()

    torch.save({"model_state_dict": best_model}, f"{save_dir}/final_model.pth")

    return best_model


def train_without_composite(model, train_hps, train_loader, device):
    model.to(device)
    model.init_weights()

    loss_hps = train_hps.loss
    optimizer_hps = train_hps.optimizer
    early_stopping_hps = train_hps.early_stopping
    scheduler_hps = train_hps.scheduler

    optimizer = Adam(
        model.parameters(),
        lr=optimizer_hps.params.lr,
        weight_decay=optimizer_hps.params.weight_decay,
    )

    scheduler = None
    num_training_steps = train_hps.num_epochs * len(train_loader)
    if scheduler_hps.params is not None:
        num_warmup_steps = int(scheduler_hps.params.warmup_ratio * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    loss_fn = SimpleLoss(loss_hps.params, len(train_loader), device)

    early_stopping = None
    if early_stopping_hps is not None:
        early_stopping = EarlyStopping(
            early_stopping_hps.patience, early_stopping_hps.tol
        )

    save_dir = os.path.join(parent_dir, train_hps.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    best_model = None
    for epoch in range(1, train_hps.num_epochs + 1):
        model.train()

        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        for x, labels in tqdm(train_loader, leave=False):
            x = x.to(device)

            out, mu, logvar = model(x)
            loss, recon_loss, kl_loss = loss_fn(out, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            train_loss += loss.detach().item()
            train_recon_loss += recon_loss.detach().item()
            train_kl_loss += kl_loss.detach().item()

            loss_fn.step()

        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)

        print(f"----Epoch {epoch}----")
        print(
            f"Train Loss: {train_loss:.6f}, Recon: {train_recon_loss:.6f}, KL: {train_kl_loss:.6f}"
        )
        print(f'LR: {optimizer.param_groups[0]["lr"]}, Beta: {loss_fn.beta}')

        if epoch % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "decoder_state_dict": model.decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler is not None else None
                    ),
                    "loss": train_loss,
                },
                f"{save_dir}/checkpoint_{epoch}.pth",
            )

        if early_stopping is not None:
            best_model, best_loss = early_stopping.step(model, train_loss)
            if best_model is not None:
                torch.save(
                    {
                        "model_state_dict": best_model,
                        "loss": best_loss,
                    },
                    f"{save_dir}/best_model.pth",
                )

                return best_model

    if best_model is None:
        best_model = model.state_dict()

    torch.save({"model_state_dict": best_model}, f"{save_dir}/final_model.pth")

    return best_model
