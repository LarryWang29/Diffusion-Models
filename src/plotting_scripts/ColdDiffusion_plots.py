"""!@file ColdDiffusion_plots.py
@brief Plotting script for the Cold Diffusion model.

@details This file contains the code to generate plots for the Cold Diffusion
model. The script generates plots for the diffusion evolution of images and
the samples generated by the model. The script saves the plots to a directory.
Specifically, this script generate figure 11 in the report.

@author Larry Wang
@Date 27/03/2024
"""

import sys
sys.path.append("./src")

import torch
import torch.nn as nn
import os
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.utils import make_grid
from custom_morphing_model import ColdDiffusion
from neural_network_models import UNet
import matplotlib.pyplot as plt

# Fix a seed for reproducibility
torch.manual_seed(1029)


def generate_ColdDiffusion_diffusion_plot(scheduler, epoch_number, num_samples):
    """
    !@brief Generate a plot of the diffusion evolution of images using the Cold Diffusion model.

    @param scheduler The noise schedule choice for the model.
    @param epoch_number The epoch number of the model checkpoint.
    @param num_samples The number of samples to generate.

    @return None
    """
    checkpoint = torch.load(f"./ColdDiffusion_checkpoints/UNet_checkpoints/{scheduler}_checkpoints/{scheduler}_epoch{epoch_number:03d}.pt")
    n_hidden = (32, 64, 128)
    gt = UNet(in_channels=1, out_channels=1, hidden_dims=n_hidden,
              time_embeddings=32, act=nn.GELU)
    model = ColdDiffusion(restore_nn=gt, noise_schedule_choice=scheduler,
                          betas=(1e-4, 0.02), n_T=1000)
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,), (1.0))])
    dataset = MNIST("./data", train=False, download=True, transform=tf)
    degradation_dataset = FashionMNIST("./data", train=False, download=True, transform=tf)

    model.load_state_dict(checkpoint)
    model.to("cuda")
    model.eval()

    # Make directory to store output
    if not os.path.exists("./figures/diffusion_evolution"):
        os.makedirs("./figures/diffusion_evolution")

    # Sample 5 images from the mnist dataset
    images = []
    for i in range(num_samples):
        image, _ = dataset[10*i+29]
        image = image.unsqueeze(0)
        image = image.to("cuda")
        # print(torch.max(image), torch.min(image))
        images.append(image)

    degradation_images = []
    for i in range(num_samples):
        degradation_image, _ = degradation_dataset[10*i+29]
        degradation_image = degradation_image.unsqueeze(0)
        degradation_image = degradation_image.to("cuda")
        degradation_images.append(degradation_image)

    # Degrade the images
    t_array = torch.linspace(0, 1000, 6, dtype=torch.int)
    degraded_images = []
    for i in range(num_samples):
        for time in t_array:
            with torch.no_grad():
                degraded_image = model.degrade(images[i], t=time,
                                               z=degradation_images[i])
                degraded_images.append(degraded_image)
        final_degraded_image = degraded_images[-1]    # Restore the images and append them to the list
        for time in (t_array[:-1].flip(0)):
            with torch.no_grad():
                restored_image = model.sample(1, (1, 28, 28), device="cuda", time=time,
                                              z_t=final_degraded_image)
                degraded_images.append(restored_image)

    degraded_images_stacked = torch.cat(degraded_images, dim=0)
    degraded_images_stacked = degraded_images_stacked.cpu()
    degraded_images_stacked = degraded_images_stacked + 0.5
    grid = make_grid(degraded_images_stacked, nrow=11)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(int(num_samples * 5), 8))
    plt.imshow(grid, aspect='auto')
    # plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Illustration of forward and backward diffusion of {num_samples} images using {scheduler} scheduler",
              fontsize=18)
    plt.tight_layout()
    plt.savefig(f"./figures/diffusion_evolution/ColdDiffusion_{scheduler}_epoch{epoch_number:03d}_diffusion_evolution.png")


def generate_new_samples_plot(scheduler, epoch_number, num_samples):
    """
    !@brief Generate a plot of the samples generated by the Cold Diffusion model.

    @param scheduler The noise schedule choice for the model.
    @param epoch_number The epoch number of the model checkpoint.
    @param num_samples The number of samples to generate.

    @return None
    """
    checkpoint = torch.load(f"./ColdDiffusion_checkpoints/UNet_checkpoints/{scheduler}_checkpoints/{scheduler}_epoch{epoch_number:03d}.pt")
    hidden_dims = (32, 64, 128)
    gt = UNet(in_channels=1, out_channels=1, hidden_dims=hidden_dims,
              time_embeddings=32, act=nn.GELU)
    model = ColdDiffusion(restore_nn=gt, noise_schedule_choice=scheduler,
                          betas=(1e-4, 0.02), n_T=1000)

    model.load_state_dict(checkpoint)
    model.to("cuda")
    model.eval()

    # Make directory to store output
    if not os.path.exists("./figures/generated_samples"):
        os.makedirs("./figures/generated_samples")

    # Sample 100 images from the model
    with torch.no_grad():
        samples = model.sample(num_samples, (1, 28, 28), device="cuda")
    samples = samples.cpu()

    norm_samples = samples + 0.5
    # norm_samples.cpu()
    grid = make_grid(norm_samples, nrow=5)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(10, 2 * num_samples // 5))
    plt.imshow(grid, aspect='auto')

    # Hide the ticks
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Samples generated using {scheduler} scheduler at epoch {epoch_number}", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"./figures/generated_samples/ColdDiffusion_{scheduler}_epoch{epoch_number:03d}_samples.png")


generate_ColdDiffusion_diffusion_plot("linear", 50, 4)
generate_ColdDiffusion_diffusion_plot("cosine", 50, 4)
generate_new_samples_plot("linear", 50, 25)
generate_new_samples_plot("cosine", 50, 25)
for t in range(5, 45, 5):
    generate_new_samples_plot("linear", t, 10)
    generate_new_samples_plot("cosine", t, 10)
