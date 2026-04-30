import torch


class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):

        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

    def add_noise(self, original, noise, t):
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(original.device)[t].reshape(
            batch_size
        )
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.to(
            original.device
        )[t].reshape(batch_size)

        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)

        return sqrt_alpha_cumprod * original + sqrt_one_minus_alpha_cumprod * noise

    def sample_prev_timestep(self, xt, noise_pred, t):
        x0 = (
            xt - (self.sqrt_one_minus_alpha_cumprod[t] * noise_pred)
        ) / self.sqrt_alpha_cumprod[t]
        x0 = torch.clamp(x0, -1.0, 1.0)

        mean = xt - (
            (self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cumprod[t])
        )
        mean = mean / torch.sqrt(self.alphas[t])

        if t == 0:
            return mean, x0
        else:
            variance = (1.0 - self.alpha_cumprod[t - 1]) / (1.0 - self.alpha_cumprod[t])
            variance = variance * self.betas[t]
            sigma = variance**0.5

            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0
