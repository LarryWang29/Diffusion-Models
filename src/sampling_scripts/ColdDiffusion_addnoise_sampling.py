import sys
sys.path.append("./src")

import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Temporatrily import Unet to test functionality
from custom_morphing_model import UNet, ColdDiffusion


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
dataset = MNIST("./data", train=True, download=True, transform=tf)
validation_dataset = MNIST("./data", train=False, download=True, transform=tf)
degrading_dataset = FashionMNIST("./data", train=False, download=True, transform=tf)
dataloader = DataLoader(
    dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True
)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True
)
model.load_state_dict(checkpoint)
model.to("cuda")
model.eval()

# # Sample 100 images from the model
# with torch.no_grad():
#     samples = model.sample(100, (1, 28, 28), device="cuda")

torch.manual_seed(1029)
# Get 25 images from the degrading dataset
degrading_dataloader = DataLoader(
    degrading_dataset, batch_size=25, shuffle=True, num_workers=4, drop_last=True
)
for batch in degrading_dataloader:
    z_t = batch[0]
    break

z_t = z_t.to("cuda")

# Add different levels of noise to z_t
noise_levels = [0, 0.001, 0.005, 0.01, 0.05, 0.25, 0.5]
# noise_levels = [0.025]

with torch.no_grad():
    # create directory to store output
    if not os.path.exists("./figures/ColdDiffusion"):
        os.makedirs("./figures/ColdDiffusion")
    # Sample at different noise levels
    for noise in noise_levels:
        z_t += noise * torch.randn(25, 1, 28, 28, device="cuda")
        # Normalise z_t to be between -0.5 and 0.5
        z_t = torch.clamp(z_t, -0.5, 0.5)
        samples = model.sample(25, (1, 28, 28), device="cuda", z_t=z_t)

        # Visualize the samples using torchvision.utils.make_grid
        samples = samples.cpu()
        samples += 0.5
        grid = make_grid(samples, nrow=5)

        plt.figure(figsize=(5, 5))
        # Hide the ticks
        plt.axis("off")
        # Display the images

        plt.imshow(grid.permute(1, 2, 0))
        plt.tight_layout()
        plt.savefig(f"./figures/ColdDiffusion/Sample_with_noise_{noise}.png")
