import torch
import torch.nn.functional as F


class CompositeLoss:
    def __init__(self, hps, discriminator, pips, num_updates_per_epoch, device):

        self._hps = hps
        self._device = device

        self.beta = 0
        self.target_beta = hps.beta
        self.beta_step_size = self.target_beta / (
            hps.warmup_epochs * num_updates_per_epoch
        )
        self.free_bits = hps.free_bits

        self.discriminator = discriminator
        self.pips = pips

        self.lambda_recon = hps.lambda_recon
        self.lambda_pips = hps.lambda_pips
        self.lambda_disc = hps.lambda_disc

    def step(self):
        self.beta = min(self.target_beta, self.beta + self.beta_step_size)

    def kl_loss(self, mu, logvar):
        loss = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()), dim=1)
        loss = torch.clamp(loss, min=self.free_bits)
        return loss.mean()

    def recon_loss(self, x_pred, x):
        loss = F.mse_loss(x_pred, x, reduction="none").view(x.shape[0], -1)
        loss = torch.sum(loss, dim=1)

        return loss.mean()

    def discriminator_loss(self, x, ones=True):
        disc_output = self.discriminator(x)

        target = torch.ones_like(disc_output) if ones else torch.zeros_like(disc_output)
        loss = F.binary_cross_entropy_with_logits(disc_output, target)

        return loss

    def __call__(self, x_pred, x, mu, logvar):
        recon_loss = self.recon_loss(x_pred, x)
        kl_loss = self.kl_loss(mu, logvar)
        pips_loss = self.pips(x_pred, x)

        total_loss = (
            self.lambda_recon * recon_loss
            + self.beta * kl_loss
            + self.lambda_pips * pips_loss
        )

        return total_loss, recon_loss, kl_loss, pips_loss


class SimpleLoss:
    def __init__(self, hps, num_updates_per_epoch, device):
        self.hps = hps
        self.device = device

        self.beta = 0
        self.target_beta = hps.beta
        self.beta_step_size = self.target_beta / (
            hps.warmup_epochs * num_updates_per_epoch
        )
        self.free_bits = hps.free_bits

        self.lambda_recon = hps.lambda_recon

    def step(self):
        self.beta = min(self.target_beta, self.beta + self.beta_step_size)

    def kl_loss(self, mu, logvar):
        loss = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()), dim=1)
        loss = torch.clamp(loss, min=self.free_bits)
        return loss.mean()

    def recon_loss(self, x_pred, x):
        loss = F.mse_loss(x_pred, x, reduction="none").view(x.shape[0], -1)
        loss = torch.sum(loss, dim=1)

        return loss.mean()

    def __call__(self, x_pred, x, mu, logvar):
        recon_loss = self.recon_loss(x_pred, x)
        kl_loss = self.kl_loss(mu, logvar)

        total_loss = self.lambda_recon * recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss
