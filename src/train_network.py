import torch
import numpy as np
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from original_diffusion_model import CNN, DDPM

# Fix a seed for reproducibility
torch.manual_seed(1029)

noise_schedule_choice = 'ddpm'
# noise_schedule_choice = 'cosine'
# noise_schedule_choice = 'constant'
# noise_schedule_choice = 'inverse'
n_hidden = (16, 32, 32, 16)
# n_hidden = (32, 64, 128, 64, 32)
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),
                                                                     (1.0))])
dataset = MNIST("./data", train=True, download=True, transform=tf)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4,
                        drop_last=True)

gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden,
         act=nn.GELU)
# For testing: (16, 32, 32, 16)
# For more capacity (for example): (64, 128, 256, 128, 64)
ddpm = DDPM(gt=gt, noise_schedule_choice=noise_schedule_choice,
            betas=(1e-4, 0.02), n_T=1000)
optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

accelerator = Accelerator()
ddpm, optim, dataloader = accelerator.prepare(ddpm, optim, dataloader)

losses = []

n_epoch = 50
losses = []

for i in range(n_epoch):
    ddpm.train()

    pbar = tqdm(dataloader)  # Wrap our loop with a visual progress bar
    for x, _ in pbar:
        optim.zero_grad()

        loss = ddpm(x)

        accelerator.backward(loss)

        losses.append(loss.item())
        avg_loss = np.average(losses[min(len(losses)-100, 0):])
        pbar.set_description(f"loss: {avg_loss:.3g}")

        optim.step()

    ddpm.eval()
    with torch.no_grad():
        xh = ddpm.sample(16, (1, 28, 28), accelerator.device)
        grid = make_grid(xh, nrow=4)

        # Save samples to `./contents` directory

        # Print the current path
        # Save the image according to the noise schedule
        save_image(grid,
                   f"./{noise_schedule_choice}_contents/" +
                   f"{noise_schedule_choice}_sample_{i:04d}.png")

        # Save the model every 5 epochs
        j = i + 1
        if j % 5 == 0:
            torch.save(ddpm.state_dict(),
                       f"./{noise_schedule_choice}_checkpoints/" +
                       f"{noise_schedule_choice}_epoch{j:03d}.pt")

        # save_image(grid,
        #            "./temp_contents/" +
        #            f"sample_{i:04d}.png")

        # # Save the model every 5 epochs
        # j = i + 1
        # if j % 5 == 0:
        #     torch.save(ddpm.state_dict(),
        #                "./temp_checkpoints/" +
        #                f"epoch{j:03d}.pt")
