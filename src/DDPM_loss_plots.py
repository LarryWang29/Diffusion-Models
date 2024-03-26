# Read in the loss data from the DDPM training and plot it

import matplotlib.pyplot as plt
import pandas as pd
import torch
from schedulers import cosine_beta_schedule, ddpm_schedules, \
    inverse_linear_schedule, constant_noise_schedule
from scipy.stats import kstest
from torchvision.datasets import MNIST
from torchvision import transforms


def plot_losses():
    # Read in the loss from DDPM training and plot it
    schedulers = ["linear", "cosine", "inverse", "constant"]

    for scheduler in schedulers:
        loss_directory = f"./DDPM_results/CNN_results/losses_{scheduler}.csv"

        # Read in the loss data
        loss_data = pd.read_csv(loss_directory)

        # Plot the loss data)
        plt.figure(figsize=(6, 5))
        plt.plot(loss_data["Training Loss"], label="Train Loss", marker="o",
                 markevery=5)
        plt.plot(loss_data[" Validation Loss"], label="Validation Loss",
                 marker="o", markevery=5)
        plt.title(f"Train and Validation Loss for {scheduler} scheduler")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./figures/{scheduler}_loss.png")


def plot_FID_scores():
    # Read in the FID scores from DDPM training and plot it
    schedulers = ["linear", "cosine", "inverse", "constant"]

    plt.figure(figsize=(6, 5))
    for scheduler in schedulers:
        fid_directory = f"./DDPM_results/CNN_results/fid_scores_{scheduler}.csv"

        # Read in the FID scores
        fid_data = pd.read_csv(fid_directory)

        # Plot the FID scores
        plt.plot(torch.linspace(5, 50, 10), fid_data["FID"],
                 label=f"{scheduler}", marker="o",
                 markevery=1)
    plt.title("FID Scores for each scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("FID Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/FID_DDPM_plots.png")


def plot_KS_statistics():
    # Define a DDPM model
    cosine_schedule = cosine_beta_schedule()
    inverse_schedule = inverse_linear_schedule()
    constant_schedule = constant_noise_schedule()
    linear_schedule = ddpm_schedules()

    # Get an image from the MNIST dataset
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,), (1.0))])
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    image, _ = dataset[0]

    p_value_cosine = []
    p_value_inverse = []
    p_value_constant = []
    p_value_linear = []

    for t in range(1, 1001, 1):
        image_1 = image.clone()
        image_2 = image.clone()
        image_3 = image.clone()
        image_4 = image.clone()

        # Diffuse the image
        z_t1 = torch.sqrt(cosine_schedule["alpha_t"][t]) * image_1 + \
            torch.randn_like(image_1) * (1-torch.sqrt(cosine_schedule["alpha_t"][t]))
        z_t2 = torch.sqrt(linear_schedule["alpha_t"][t]) * image_2 + \
            torch.randn_like(image_2) * (1-torch.sqrt(linear_schedule["alpha_t"][t]))
        z_t3 = torch.sqrt(inverse_schedule["alpha_t"][t]) * image_1 + \
            torch.randn_like(image_3) * (1-torch.sqrt(inverse_schedule["alpha_t"][t]))
        z_t4 = torch.sqrt(constant_schedule["alpha_t"][t]) * image_1 + \
            torch.randn_like(image_4) * (1-torch.sqrt(constant_schedule["alpha_t"][t]))

        # Calculate the p-value
        p_value_cosine.append(kstest(z_t1.flatten().numpy(), "norm").pvalue)
        p_value_linear.append(kstest(z_t2.flatten().numpy(), "norm").pvalue)
        p_value_inverse.append(kstest(z_t3.flatten().numpy(), "norm").pvalue)
        p_value_constant.append(kstest(z_t4.flatten().numpy(), "norm").pvalue)

    # Plot the p-values
    t_array = torch.linspace(1, 1001, 1000, dtype=torch.int)
    plt.figure(figsize=(8, 6))
    plt.semilogy(t_array, p_value_cosine, label="Cosine")
    plt.semilogy(t_array, p_value_linear, label="Linear")
    plt.semilogy(t_array, p_value_inverse, label="Inverse")
    plt.semilogy(t_array, p_value_constant, label="Constant")
    plt.axhline(y=0.05, color='r', linestyle='--',
                label='p=0.05 Threshold')

    x_ticks = []
    # Highlight the points where p-values cross the threshold with vertical line
    for i, p_value in enumerate(p_value_cosine):
        if p_value > 0.05:
            plt.axvline(x=i+1, color='k', linestyle='--')
            x_ticks.append(i+1)
            break
    for i, p_value in enumerate(p_value_linear):
        if p_value > 0.05:
            plt.axvline(x=i+1, color='k', linestyle='--')
            x_ticks.append(i+1)
            break
    for i, p_value in enumerate(p_value_inverse):
        if p_value > 0.05:
            plt.axvline(x=i+1, color='k', linestyle='--')
            x_ticks.append(i+1)
            break
    for i, p_value in enumerate(p_value_constant):
        if p_value > 0.05:
            plt.axvline(x=i+1, color='k', linestyle='--')
            x_ticks.append(i+1)
            break

    plt.xticks(x_ticks)
    plt.title("Kolmogorov-Smirnov p-values for each scheduler")
    plt.xlabel("Time")
    plt.ylabel("P-value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/KS_p_values.png")


# plot_losses()
# plot_KS_statistics()
plot_FID_scores()
