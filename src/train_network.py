import torch
import numpy as np
import torch.nn as nn
import sys
import os
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from original_diffusion_model import CNN, DDPM

# Temporatrily import Unet to test functionality
from fashion_MNIST_diffusion_model import UNet, ColdDiffusion

# Fix a seed for reproducibility
torch.manual_seed(1029)


def train_network(noise_schedule_choice, n_hidden, n_epoch,
                  record_metrics=True, model_type='ColdDiffusion',
                  nn_choice='CNN'):

    # Erase previous losses file
    if record_metrics:
        os.makedirs(f"./{model_type}_contents/{nn_choice}_contents/{noise_schedule_choice}_contents",
                    exist_ok=True)
        os.makedirs(f"./{model_type}_checkpoints/{nn_choice}_checkpoints/{noise_schedule_choice}_checkpoints",
                    exist_ok=True)
        os.makedirs(f"./{model_type}_results/{nn_choice}_results", exist_ok=True)
        with open(f"./{model_type}_results/{nn_choice}_results/losses_{noise_schedule_choice}.csv", "w") as f:
            f.write("Epoch,Training Loss, Validation Loss\n")

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,), (1.0))])
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    validation_dataset = MNIST("./data", train=False, download=True,
                               transform=tf)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True,
                            num_workers=4,
                            drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=128,
                                       shuffle=True, num_workers=4,
                                       drop_last=True)

    if model_type == 'ColdDiffusion':
        if nn_choice == 'CNN':
            gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden,
                     act=nn.GELU)
        else:
            gt = UNet(in_channels=1, out_channels=1, hidden_dims=n_hidden,
                      time_embeddings=32, act=nn.GELU)
        ddpm = ColdDiffusion(restore_nn=gt, noise_schedule_choice=noise_schedule_choice,
                             betas=(1e-4, 0.02), n_T=1000)
    elif model_type == 'DDPM':
        # gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden,
        #          act=nn.GELU)
        if nn_choice == 'CNN':
            gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden,
                     act=nn.GELU)
        else:
            gt = UNet(in_channels=1, out_channels=1, hidden_dims=n_hidden,
                      time_embeddings=32, act=nn.GELU)
        ddpm = DDPM(gt=gt, noise_schedule_choice=noise_schedule_choice,
                    betas=(1e-4, 0.02), n_T=1000)
    else:
        raise ValueError("Invalid model type. Choose 'ColdDiffusion' or 'DDPM'")

    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

    accelerator = Accelerator()
    ddpm, optim, dataloader, validation_dataloader = accelerator.prepare(
        ddpm, optim,
        dataloader,
        validation_dataloader
    )

    training_losses = []
    validation_losses = []

    for i in range(n_epoch):
        batch_losses = []
        ddpm.train()

        pbar = tqdm(dataloader)  # Wrap our loop with a visual progress bar
        for x, _ in pbar:
            optim.zero_grad()

            loss = ddpm(x)

            accelerator.backward(loss)

            batch_losses.append(loss.item())
            avg_loss = np.average(batch_losses[min(len(batch_losses)-100, 0):])
            pbar.set_description(f"loss: {avg_loss:.3g}")

            optim.step()

        # Save the training loss per epoch as the average of the batch losses
        training_losses.append(torch.mean(torch.tensor(batch_losses)))

        ddpm.eval()
        with torch.no_grad():
            # Compute the loss on the validation set
            batch_losses = []
            for x, _ in validation_dataloader:
                loss = ddpm(x)
                batch_losses.append(loss.item())

            # Save the validation loss per epoch as the average of the batches
            validation_losses.append(torch.mean(torch.tensor(batch_losses)))

            xh = ddpm.sample(16, (1, 28, 28), accelerator.device)

            # Add 0.5 to correct for imshow clipping at 0
            xh += 0.5
            grid = make_grid(xh, nrow=4)

            # Save samples to `./contents` directory

            # Print the current path

        if record_metrics:

            # Save the image according to the noise schedule
            save_image(grid, f"./{model_type}_contents/" +
                       f"{nn_choice}_contents/" +
                       f"{noise_schedule_choice}_contents/" +
                       f"{noise_schedule_choice}_sample_{i:04d}.png")

            # Save the model every 5 epochs
            j = i + 1
            if j % 5 == 0:
                torch.save(ddpm.state_dict(), f"./{model_type}_checkpoints/" +
                           f"{nn_choice}_checkpoints/" +
                           f"{noise_schedule_choice}_checkpoints/" +
                           f"{noise_schedule_choice}_epoch{j:03d}.pt")

            # Save the training and validation losses to a csv file
            with open(f"{model_type}_results/{nn_choice}_results/losses_{noise_schedule_choice}.csv", "a") as f:
                f.write(f"{i},{training_losses[-1]},{validation_losses[-1]}\n")
        else:
            # Save the images in a temporary directory
            save_image(grid, f"./temp_contents/sample_{i:04d}.png")

            # Save a checkpoint at the end of the training
            if i == n_epoch - 1:
                torch.save(ddpm.state_dict(), f"./temp_checkpoints/epoch{i:03d}.pt")


if __name__ == "__main__":
    noise_schedule = sys.argv[1]
    if noise_schedule not in ["linear", "cosine", "inverse", "constant"]:
        raise ValueError("Invalid noise schedule choice. Choose 'linear', 'cosine', 'inverse', or 'constant'")

    model_type = sys.argv[2]
    if model_type not in ["ColdDiffusion", "DDPM"]:
        raise ValueError("Invalid model type. Choose 'ColdDiffusion' or 'DDPM'")

    nn_choice = sys.argv[3]
    if nn_choice not in ["CNN", "UNet"]:
        raise ValueError("Invalid neural network choice. Choose 'CNN' or 'UNet'")

    if model_type == "ColdDiffusion":
        if nn_choice == "CNN":
            train_network(noise_schedule, (16, 32, 32, 16), 50, record_metrics=True,
                          model_type=model_type, nn_choice=nn_choice)
        else:
            train_network(noise_schedule, (32, 64, 128), 50, record_metrics=True,
                          model_type=model_type, nn_choice=nn_choice)
    else:
        if nn_choice == "CNN":
            train_network(noise_schedule, (16, 32, 32, 16), 50, record_metrics=True,
                          model_type=model_type, nn_choice=nn_choice)
        else:
            train_network(noise_schedule, (32, 64, 128), 50, record_metrics=True,
                          model_type=model_type, nn_choice=nn_choice)
# train_network("ddpm", (16, 32, 32, 16), 10, record_metrics=False)
