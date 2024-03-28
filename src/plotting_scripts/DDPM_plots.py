# This scripts plots the diffused images at different time steps
import sys
sys.path.append("./src")

import torch
import torch.nn as nn
import os
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from DDPM_model import DDPM
from neural_network_models import CNN
import matplotlib.pyplot as plt

# Fix a seed for reproducibility
# torch.manual_seed(1029)


def generate_DDPM_diffusion_plot(scheduler, epoch_number, num_samples):
    checkpoint = torch.load(f"./DDPM_checkpoints/CNN_checkpoints/{scheduler}_checkpoints/{scheduler}_epoch{epoch_number:03d}.pt")
    n_hidden = (16, 32, 32, 16)
    gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden,
             act=nn.GELU)
    model = DDPM(gt=gt, noise_schedule_choice=scheduler,
                 betas=(1e-4, 0.02), n_T=1000)
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,), (1.0))])
    dataset = MNIST("./data", train=False, download=True, transform=tf)

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

    # Degrade the images
    t_array = torch.linspace(0, 1000, 6, dtype=torch.int)
    degraded_images = []
    for i in range(num_samples):
        for time in t_array:
            with torch.no_grad():
                degraded_image = model.degrade(images[i], t=time)[0]
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
    plt.savefig(f"./figures/diffusion_evolution/{scheduler}_epoch{epoch_number:03d}_diffusion_evolution.png")


def generate_new_samples_plot(scheduler, epoch_number, num_samples):
    checkpoint = torch.load(f"./DDPM_checkpoints/CNN_checkpoints/{scheduler}_checkpoints/{scheduler}_epoch{epoch_number:03d}.pt")
    n_hidden = (16, 32, 32, 16)
    gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden,
             act=nn.GELU)
    model = DDPM(gt=gt, noise_schedule_choice=scheduler,
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
    plt.savefig(f"./figures/generated_samples/{scheduler}_epoch{epoch_number:03d}_samples.png")


generate_DDPM_diffusion_plot("cosine", 50, 4)
generate_DDPM_diffusion_plot("inverse", 50, 4)
generate_DDPM_diffusion_plot("linear", 50, 4)
generate_DDPM_diffusion_plot("constant", 50, 4)
# generate_new_samples_plot('linear', 50, 25)
# generate_new_samples_plot('cosine', 50, 25)
# generate_new_samples_plot('inverse', 50, 25)
# generate_new_samples_plot('constant', 50, 25)
# for t in range(5, 55, 5):
#     generate_new_samples_plot('linear', t, 10)
#     generate_new_samples_plot('cosine', t, 10)
#     generate_new_samples_plot('inverse', t, 10)
#     generate_new_samples_plot('constant', t, 10)
