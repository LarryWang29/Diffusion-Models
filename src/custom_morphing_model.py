"""!@file src/custom_morphing_model.py
@!brief Custom morphing model for Cold Diffusion.

@details This file contains the implementation of a custom morphing
diffusion model. The model is used to morph images from one dataset
(MNIST) to another dataset (FashionMNIST) using a UNet model. The
neural network for the "restoration operator" is a modified UNet model.

@author Larry Wang
@Date 22/03/2024
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
    ):
        """!@brief Initialise the Cold Diffusion morphing model.
        
        @details This function initialises the Cold Diffusion morphing model.
        The model is used to morph images from one dataset (MNIST) to another
        dataset (FashionMNIST) using a UNet model. The neural network for the
        "restoration operator" is a modified UNet model.

        @param restore_nn The neural network used for the "restoration operator".
        @param noise_schedule_choice The choice of noise schedule for the model.
        Possible choices are "linear", "cosine", "inverse", and "constant".
        @param betas The beta values for the noise schedule; this is only specific for 
        the "linear" and "inverse" noise schedules.
        @param n_T The total number of diffusion time steps for the model.
        @param criterion The loss function used for the model. The default is the
        Mean Squared Error loss.

        @return None
        """
        super().__init__()

        self.restore_nn = restore_nn

        # Transform the morphing dataset to be between -0.5 and 0.5
        tf = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1.0))])
        self.mixing_dataset = FashionMNIST(root="data", download=True,
                                           transform=tf)

        # Choose the noise schedule
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
        """!@brief Forward pass of the Cold Diffusion model.

        @details This function performs the forward pass of the Cold Diffusion
        model; the algorithm used is Algorithm 18.1 in Prince.
        
        @param x The input image to be morphed, can be a batch of images.
        """
        # Randomly sample a batch of images from the morphing dataset
        num_of_samples = x.shape[0]
        z = torch.stack([self.mixing_dataset[np.random.randint(len(self.mixing_dataset))][0]
                        for _ in range(num_of_samples)]).to(x.device)

        # Randomly sample a time step from 1 to n_T
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)

        # Degrade the image at time to time t
        z_t = self.degrade(x, t, z)

        # Calculate the loss between the degraded image and the restored image
        return self.criterion(x, self.restore_nn(z_t, t / self.n_T))

    def degrade(self, x: torch.Tensor, t, z) -> torch.Tensor:
        """!@brief Degrade the image to time t.

        @details This function degrades the image to time t using the
        degradation operator. The degradation operator is defined as
        z_t = sqrt(alpha_t) * x + sqrt(1 - alpha_t) * z, where alpha_t
        is the noise schedule at time t.

        @param x The input image(s) to be degraded.
        @param t The time step to degrade the image to.
        @param z The random image(s) from the morphing dataset used for degrading.
        """
        # First component of the sum
        alpha_t = self.alpha_t[t, None, None, None]
        first_component = torch.sqrt(alpha_t) * x

        # Second component of the sum
        second_component = torch.sqrt(1 - alpha_t) * z
        return first_component + second_component

    def restore(self, z_t, t):
        """!@brief Restore the image via direct restoration.

        @details This function restores the image using the
        restoration operator.

        @param z_t The degraded image to be restored.
        @param t The time step to feed into the restoration operator.
        """
        return self.restore_nn(z_t, t / self.n_T)

    def sample(self, n_sample: int, size, device, time=0,
               z_t=None) -> torch.Tensor:
        """!@brief Sample images from the model.

        @details This function samples images from the model. If z_t isn't
        provided, it will sample random images from the morphing dataset
        according to the size and restore these images back original
        MNISt images. The sampling algorithm follows the Cold Diffusion
        paper.

        @param n_sample The number of images to sample.
        @param size Not used here, but kept for compatibility with the
        DDPM model.
        @param device The device to run the model on.
        @param time The time step to sample the images to. Default is 0,
        which is the original MNIST image.
        @param z_t The degraded image to restore back to the original
        MNIST image.

        @return The restored image.
        """
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
        """!@brief Conditional sample from the model.

        @details This function samples images from the model conditioned
        on the input image x. The image is degraded to time n_T and
        restored back to time t.

        @param x The input image(s) to condition on.
        @param z The random image(s) from the morphing dataset used for
        degrading.
        @param t The time step to restore the image to.
        @param device The device to run the model on.

        @return The conditionally sampled restored image.
        """
        # Degrade it all the way to n_T
        z_t = self.degrade(x, self.n_T, z)

        x_restored = self.sample(1, None, device, time=t, z_t=z_t)
        return x_restored
