"""!@file src/DDPM_model.py
@brief Implementation of the DDPM model.

@details This file contains the implementation of the DDPM model. The DDPM model
is used to model the diffusion process of an image. The model is trained using a
neural network model that predicts the "error term" of the image. the architecture
of the neural network model is a CNN model.

@author Larry Wang
@Date 22/03/2024
"""

import torch
import torch.nn as nn
from schedulers import cosine_beta_schedule, ddpm_schedules
from schedulers import inverse_linear_schedule, constant_noise_schedule
from typing import Tuple



class DDPM(nn.Module):
    def __init__(
        self,
        gt,
        noise_schedule_choice,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        """
        !@brief Initialise the DDPM model.

        @details This function initialises the DDPM model. The model is used to model
        the diffusion process of an image. The model is trained using a neural network
        model that predicts the "error term" of the image. The architecture of the neural
        network model is a CNN model. The noise schedule for the model can be chosen from
        "linear", "cosine", "inverse", and "constant".

        @param gt The neural network used for predicting the noise.
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

        self.gt = gt

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
        """
        !@brief Forward pass of the DDPM model.

        @details This function performs the forward pass of the DDPM model. The forward
        pass of the model predicts the "error term" of the image. The "error term" is
        predicted using the neural network model that is trained to predict the noise
        of the image. The loss function used for the model is the Mean Squared Error loss.

        @param x The input image to the model.

        @return The predicted "error term" of the image.
        """

        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        z_t, eps = self.degrade(x, t)
        # This is the z_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this z_t.

        return self.criterion(eps, self.gt(z_t, t / self.n_T))

    def degrade(self, x: torch.Tensor, t) -> torch.Tensor:
        """
        !@brief Degrade the image.

        @details This function degrades the image using the diffusion process to 
        timestep t. The diffusion process is modelled using the equation:
        z_t = sqrt(alpha_t) * x + sqrt(1 - alpha_t) * eps
        where z_t is the degraded image, x is the input image, alpha_t is the alpha
        value at time t, and eps is Gaussian noise.

        @param x The input image to be degraded.
        @param t The timestep to degrade the image to.

        @return The degraded image and the Gaussian noise used to degrade the image.
        """
        alpha_t = self.alpha_t[t, None, None, None]
        eps = torch.randn_like(x)
        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps
        return z_t, eps

    def sample(self, n_sample: int, size, device, time=0,
               z_t=None) -> torch.Tensor:
        """
        !@brief Sample from the model.

        @details This function samples from the model. The sampling process is
        performed using the reverse diffusion process. The reverse diffusion
        follows from algorithm 18.1 in Prince.

        @param n_sample The number of samples to generate.
        @param size The size of the each sample.
        @param device The device to place the samples on.
        @param time The time to sample to restore the image to, default is 0.
        @param z_t The 'starting point' of the reverse diffusion process, default is None.

        @return The generated samples.
        """

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
        """!@brief Conditional sample from the model.

        @details This function samples images from the model conditioned
        on the input image x. The image is degraded to time n_T and
        restored back to time t.

        @param x The input image(s) to condition on.
        @param t The time step to restore the image to.
        @param device The device to run the model on.

        @return The conditionally sampled restored image.
        """
        z_t, _ = self.degrade(x, self.n_T)

        x_restored = self.sample(x.shape[0], x.shape[1:], device, time=t,
                                 z_t=z_t)
        return x_restored
