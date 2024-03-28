# Read in the loss data from the DDPM training and plot it

import matplotlib.pyplot as plt
import pandas as pd
import torch


def plot_losses():
    # Read in the loss from DDPM training and plot it
    schedulers = ["linear", "cosine"]
    plt.figure(figsize=(6, 5))

    for scheduler in schedulers:
        loss_directory = f"./ColdDiffusion_results/UNet_results/losses_{scheduler}.csv"

        # Read in the loss data
        loss_data = pd.read_csv(loss_directory)

        # Plot the loss data)
        plt.plot(loss_data["Training Loss"], label=f"Train Loss {scheduler} schedule", marker="o",
                 markevery=5)
        plt.plot(loss_data[" Validation Loss"], label=f"Validation Loss {scheduler} schedule",
                 marker="o", markevery=5)
    plt.title("Train and Validation Loss for linear and cosine scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/ColdDiffusion_loss.png")


def plot_FID_scores():
    # Read in the FID scores from DDPM training and plot it
    schedulers = ["linear", "cosine"]

    plt.figure(figsize=(6, 5))
    for scheduler in schedulers:
        fid_directory = f"./ColdDiffusion_results/UNet_results/fid_scores_{scheduler}.csv"

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
    plt.savefig("./figures/ColdDiffusion_FID_DDPM_plots.png")


plot_losses()
plot_FID_scores()
