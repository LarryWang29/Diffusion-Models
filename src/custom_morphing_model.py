"""
Alternative model for diffusing MNIST images by using Fashion MNIST dataset.
"""

import torch
import torch.nn as nn
import numpy as np
from schedulers import cosine_beta_schedule, ddpm_schedules, \
    inverse_linear_schedule, constant_noise_schedule
from typing import Tuple
from torchvision.datasets import FashionMNIST
from torchvision import transforms


class ColdDiffusion(nn.Module):
    def __init__(
        self,
        restore_nn,
        noise_schedule_choice,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        self.restore_nn = restore_nn

        tf = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1.0))])
        self.mixing_dataset = FashionMNIST(root="data", download=True,
                                           transform=tf)

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

        # Randomly sample an image from the mixing dataset
        num_of_samples = x.shape[0]
        # Sample the same number of images as the batch size
        z = torch.stack([self.mixing_dataset[np.random.randint(len(self.mixing_dataset))][0]
                        for _ in range(num_of_samples)]).to(x.device)
        # z = self.mixing_dataset[np.random.randint(len(self.mixing_dataset))][0]
        # z = z[None, ...].to(x.device)

        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)

        z_t = self.degrade(x, t, z)
        # This is the z_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this z_t.

        return self.criterion(x, self.restore_nn(z_t, t / self.n_T))

    def degrade(self, x: torch.Tensor, t, z) -> torch.Tensor:
        # First component of the sum
        alpha_t = self.alpha_t[t, None, None, None]
        first_component = torch.sqrt(alpha_t) * x
        second_component = torch.sqrt(1 - alpha_t) * z
        return first_component + second_component

    def restore(self, z_t, t):
        # Restore the image directly at time t using the restore_nn
        return self.restore_nn(z_t, t / self.n_T)

    def sample(self, n_sample: int, size, device, time=0,
               z_t=None) -> torch.Tensor:
        """Algorithm 18.2 in Prince"""
        # z_t = torch.randn(n_sample, *size, device=device)
        _one = torch.ones(n_sample, device=device)
        # Sample random images from the mixing dataset according to size
        if z_t is None:
            z_t = torch.stack([self.mixing_dataset[np.random.randint(len(self.mixing_dataset))][0]
                               for _ in range(n_sample)]).to(device)
        restored_image = z_t.clone()

        for t in range(self.n_T, time, -1):

            # degrade of restoration at time t
            restoration = self.restore(restored_image, t * _one)
            first_component = self.degrade(restoration, t, z_t)
            second_component = self.degrade(restoration, t-1, z_t)
            restored_image = restored_image - first_component + second_component

        return restored_image

    def conditional_sample(self, x, z, t, device):
        # Degrade it all the way to n_T
        z_t = self.degrade(x, self.n_T, z)

        x_restored = self.sample(1, None, device, time=t, z_t=z_t)
        return x_restored
