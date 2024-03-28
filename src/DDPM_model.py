import torch
import torch.nn as nn
from schedulers import cosine_beta_schedule, ddpm_schedules
from schedulers import inverse_linear_schedule, constant_noise_schedule
from typing import Tuple

# Fix a seed for reproducibility
# torch.manual_seed(1029)


class DDPM(nn.Module):
    def __init__(
        self,
        gt,
        noise_schedule_choice,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        self.gt = gt

        if noise_schedule_choice == "linear":
            noise_schedule = ddpm_schedules(betas[0], betas[1], n_T)
        if noise_schedule_choice == "cosine":
            noise_schedule = cosine_beta_schedule(0.008, n_T)
        if noise_schedule_choice == "inverse":
            noise_schedule = inverse_linear_schedule(betas[0], betas[1], n_T)
        if noise_schedule_choice == "constant":
            noise_schedule = constant_noise_schedule(n_T)

        # `register_buffer` will track these tensors for device placement, but
        # not store them as model parameters. This is useful for constants.
        self.register_buffer("beta_t", noise_schedule["beta_t"])
        self.beta_t  # Exists! Set by register_buffer
        self.register_buffer("alpha_t", noise_schedule["alpha_t"])
        self.alpha_t

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Algorithm 18.1 in Prince"""

        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        z_t, eps = self.degrade(x, t)
        # This is the z_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this z_t.

        return self.criterion(eps, self.gt(z_t, t / self.n_T))

    def degrade(self, x: torch.Tensor, t) -> torch.Tensor:
        # First component of the sum
        alpha_t = self.alpha_t[t, None, None, None]
        eps = torch.randn_like(x)
        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps
        return z_t, eps

    def sample(self, n_sample: int, size, device, time=0,
               z_t=None) -> torch.Tensor:
        """Algorithm 18.2 in Prince"""

        _one = torch.ones(n_sample, device=device)

        if z_t is None:
            z_t = torch.randn(n_sample, *size, device=device)
        z_t = z_t.clone()

        for i in range(self.n_T, time, -1):
            alpha_t = self.alpha_t[i]
            beta_t = self.beta_t[i]

            # First line of loop:
            z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * \
                self.gt(z_t, (i/self.n_T) * _one)
            z_t /= torch.sqrt(1 - beta_t)

            if i > 1:
                # Last line of loop:
                z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)

        return z_t

    def conditional_sample(self, x, t, device):
        z_t, _ = self.degrade(x, self.n_T)

        x_restored = self.sample(x.shape[0], x.shape[1:], device, time=t,
                                 z_t=z_t)
        return x_restored
