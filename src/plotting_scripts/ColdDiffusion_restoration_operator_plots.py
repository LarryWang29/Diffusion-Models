import sys
sys.path.append("./src")

import torch
import numpy as np
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST,FashionMNIST
from torchvision.utils import make_grid
from custom_morphing_model import ColdDiffusion
from neural_network_models import UNet
import matplotlib.pyplot as plt

torch.manual_seed(42)
def make_restoration_operator_plots():
    # Fix a seed for reproducibility

    # Change the path to the outer directory
    scheduler = 'linear'
    # checkpoint = torch.load(f"./DDPM_checkpoints/CNN_checkpoints/{scheduler}_checkpoints/{scheduler}_epoch050.pt")
    checkpoint = torch.load(f"./ColdDiffusion_checkpoints/UNet_checkpoints/{scheduler}_checkpoints/{scheduler}_epoch050.pt")
    gt = UNet(in_channels=1, out_channels=1, hidden_dims=(32, 64, 128),
            act=nn.GELU)

    model = ColdDiffusion(restore_nn=gt, noise_schedule_choice=scheduler,
                betas=(1e-4, 0.02), n_T=1000)
    tf = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1.0))])
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    degrading_dataset = FashionMNIST("./data", train=True, download=True,
                                        transform=tf)
    model.load_state_dict(checkpoint)
    model.to("cuda")
    model.eval()

    # model.conditional_sample(real_images, )
    t_array = torch.linspace(100, 1000, 10, dtype=torch.int32).to("cuda")

    # Assess the model's ability to directly restore the images
    # Randomly choose an image from the degrading dataset
    idx = np.random.randint(0, len(degrading_dataset))
    z = degrading_dataset[idx][0].unsqueeze(0).to("cuda")

    x = dataset[idx][0].unsqueeze(0).to("cuda")

    # Degrade the image to different time steps and restore it directly using the restoration operator
    degraded_images = []
    restored_images = []
    for t in t_array:
        degraded_image = model.degrade(x, t, z)
        degraded_images.append(degraded_image + 0.5)
        restored_image = model.restore(degraded_image, t *  torch.ones(1, device="cuda"))
        restored_images.append(restored_image + 0.5)
    
    # Use make grid to plot the images
    degraded_images = torch.cat(degraded_images, dim=0)
    restored_images = torch.cat(restored_images, dim=0)
    restored_images = torch.cat([degraded_images, restored_images], dim=0)
    restored_images = make_grid(restored_images, nrow=10)
    plt.figure(figsize=(10, 2))
    plt.imshow(restored_images.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.tight_layout()
    plt.title("Restoration using the restoration operator, at different time steps of degradation")
    plt.savefig("./figures/ColdDiffusion/restoration_operator.png", bbox_inches="tight")

    # Randomly sample 10 images from the degrading dataset
    idxs = np.random.choice(len(degrading_dataset), 10)
    z = torch.cat([degrading_dataset[idx][0].unsqueeze(0) for idx in idxs], dim=0).to("cuda")

    direct_restorations = model.restore(z, 1000 * torch.ones(10, device="cuda"))
    with torch.no_grad():
        iterative_restorations = model.sample(10, (1, 28, 28), device="cuda", z_t=z)

    # Use make grid to plot the images
    restored_images = torch.cat([direct_restorations, iterative_restorations], dim=0)
    restored_images = make_grid(restored_images, nrow=10)
    restored_images = restored_images + 0.5
    plt.figure(figsize=(10, 2))
    plt.imshow(restored_images.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.tight_layout()
    plt.title("Direct and iterative restoration of 10 images from the degrading dataset")
    plt.savefig("./figures/ColdDiffusion/restoration_operator_comparison.png", bbox_inches="tight")

make_restoration_operator_plots()
