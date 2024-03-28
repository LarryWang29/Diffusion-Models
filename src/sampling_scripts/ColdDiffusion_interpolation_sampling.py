"""!@file ColdDiffusion_addnoise_sampling.py
@brief Sample images from the Cold Diffusion model using different 
interpolations.

@details This file contains the code to sample images from the Cold Diffusion
model with added noise. This is an attempt to diversify the images generated
by the model. It produces figures 17 in the report.

@author Larry Wang
@Date 27/03/2024
"""

import sys
sys.path.append("./src")

import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from custom_morphing_model import ColdDiffusion
from neural_network_models import UNet


# Change the path to the outer directory
scheduler = "cosine"
checkpoint = torch.load(
    f"./ColdDiffusion_checkpoints/UNet_checkpoints/{scheduler}_checkpoints/{scheduler}_epoch050.pt"
)
gt = UNet(in_channels=1, out_channels=1, hidden_dims=(32, 64, 128), act=nn.GELU)

model = ColdDiffusion(
    restore_nn=gt, noise_schedule_choice=scheduler, betas=(1e-4, 0.02), n_T=1000
)
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
degrading_dataset = FashionMNIST("./data", train=False, download=True, transform=tf)
model.load_state_dict(checkpoint)
model.to("cuda")
model.eval()

torch.manual_seed(42)


def interpolation_sampling(model, num_interpolations):
    # Make directory to store output
    if not os.path.exists("./figures/ColdDiffusion"):
        os.makedirs("./figures/ColdDiffusion")
    # Get 25 images from the degrading dataset
    degrading_dataloader = DataLoader(
        degrading_dataset,
        batch_size=25 * num_interpolations,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    for batch in degrading_dataloader:
        z_t = batch[0]
        break

    z_t = z_t.to("cuda")

    # Create an tensor to store the interpolated images
    z_t_interpolate = torch.zeros(25, 1, 28, 28, device="cuda")

    # Create 25 images that are linearly interpolated between num_interpolations images
    for i in range(25):
        for j in range(num_interpolations):
            z_t_interpolate[i] += z_t[num_interpolations * i + j] / num_interpolations

    with torch.no_grad():
        samples = model.sample(25, (1, 28, 28), device="cuda", z_t=z_t_interpolate)

        # Visualize the samples using torchvision.utils.make_grid
        samples = samples.cpu()
        samples += 0.5
        grid = make_grid(samples, nrow=5)

        plt.figure(figsize=(6, 6))
        # Hide the ticks
        plt.axis("off")
        # Display the images

        plt.imshow(grid.permute(1, 2, 0))
        plt.tight_layout()
        plt.savefig(
            f"./figures/ColdDiffusion/interpolation_sampling_{num_interpolations}.png"
        )


interpolations = [1, 2, 5, 10, 20]
for num_interpolations in interpolations:
    interpolation_sampling(model, num_interpolations)
