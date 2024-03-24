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


class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 input_shape, time_emb_dim=32):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, in_channels)
        )

        self.ds_conv = nn.Conv2d(in_channels, in_channels, kernel_size=7,
                                 padding=3, groups=in_channels)

        self.net = nn.Sequential(
            # layernorm
            nn.LayerNorm((in_channels, *input_shape)),
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=3,
                      padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
        )

        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, t):
        h = self.ds_conv(x)

        condition = self.time_mlp(t)[:, :, None, None]
        # print(condition.shape)
        h = h + condition

        h = self.net(h)
        return h + self.res_conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        hidden_dims=[32, 64, 128],
        time_embeddings=32,
        act=nn.GELU,
    ):
        super().__init__()
        self.conv_block1 = ConvNextBlock(in_channels, hidden_dims[0],
                                         (28, 28), time_emb_dim=32)
        self.maxpool_block1 = self.maxpool_block(2, 2)
        self.conv_block2 = ConvNextBlock(hidden_dims[0], hidden_dims[1],
                                         (14, 14), time_emb_dim=32)
        self.maxpool_block2 = self.maxpool_block(2, 2)
        self.conv_block3 = ConvNextBlock(hidden_dims[1], hidden_dims[2],
                                         (7, 7), time_emb_dim=32)
        self.maxpool_block3 = self.maxpool_block(2, 2)

        self.middle_block = ConvNextBlock(hidden_dims[2], hidden_dims[2]*2,
                                          (3, 3), time_emb_dim=32)
        self.transpose_block1 = self.transpose_block(hidden_dims[2]*2, hidden_dims[2], 3, 2)

        self.upconv_block1 = ConvNextBlock(hidden_dims[2]*2, hidden_dims[2],
                                           (7, 7), time_emb_dim=32)
        self.transpose_block2 = self.transpose_block(hidden_dims[1]*2, hidden_dims[1], 2, 2)
        self.upconv_block2 = ConvNextBlock(hidden_dims[1]*2, hidden_dims[1],
                                           (14, 14), time_emb_dim=32)
        self.transpose_block3 = self.transpose_block(hidden_dims[1], hidden_dims[0], 2, 2)
        self.upconv_block3 = ConvNextBlock(hidden_dims[1], hidden_dims[0],
                                           (28, 28), time_emb_dim=32)

        self.final = self.final_block(hidden_dims[0], out_channels, 1, 1)

        self.time_embed = nn.Sequential(
            nn.Linear(time_embeddings * 2, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, hidden_dims[0]),
        )
        frequencies = torch.tensor(
            [0] + [2 * np.pi * 1.5**i for i in range(time_embeddings - 1)]
        )
        self.register_buffer("frequencies", frequencies)

    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        phases = torch.concat(
            (
                torch.sin(t[:, None] * self.frequencies[None, :]),
                torch.cos(t[:, None] * self.frequencies[None, :]) - 1,
            ),
            dim=1,
        )

        return self.time_embed(phases)

    def maxpool_block(self, kernel_size, stride):
        return nn.MaxPool2d(kernel_size, stride)

    def transpose_block(self, in_channels, out_channels, kernel_size, stride):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                  stride)

    def final_block(self, in_channels, out_channels, kernel_size, stride):
        final = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        return final

    def forward(self, x, t):
        t = self.time_encoding(t)

        conv1 = self.conv_block1(x, t)
        maxpool1 = self.maxpool_block1(conv1)
        conv2 = self.conv_block2(maxpool1, t)
        maxpool2 = self.maxpool_block2(conv2)
        conv3 = self.conv_block3(maxpool2, t)
        maxpool3 = self.maxpool_block3(conv3)

        middle = self.middle_block(maxpool3, t)

        transpose1 = self.transpose_block1(middle)
        upconv1 = self.upconv_block1(torch.cat([transpose1, conv3], dim=1), t)
        transpose2 = self.transpose_block2(upconv1)
        upconv2 = self.upconv_block2(torch.cat([transpose2, conv2], dim=1), t)
        transpose3 = self.transpose_block3(upconv2)
        upconv3 = self.upconv_block3(torch.cat([transpose3, conv1], dim=1), t)

        final = self.final(upconv3)

        return final


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
        z = self.mixing_dataset[np.random.randint(len(self.mixing_dataset))][0]
        z = z[None, ...].to(x.device)

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
