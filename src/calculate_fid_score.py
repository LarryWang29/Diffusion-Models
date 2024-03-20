import torch
import torch.nn as nn
from torchmetrics.image.fid import FrechetInceptionDistance
from original_diffusion_model import DDPM, CNN
from torchvision.datasets import MNIST
from torchvision import transforms

# Load the dataset
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5,), (1.0))])
dataset = MNIST(root="data", download=True, transform=tf)
n_hidden = (16, 32, 32, 16)
# n_hidden = (32, 64, 128, 64, 32)
gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden,
         act=nn.GELU)


def to_rgb(images):
    return images.repeat(1, 3, 1, 1)


def calculate_fid_score(scheduler):
    # Clear out the current file
    with open(f"fid_scores_{scheduler}.csv", "w") as f:
        f.write("Epoch,FID\n")

    # Load 1000 images from the MNIST dataset
    real_images = torch.stack([dataset[i][0] for i in range(1000)])
    real_images = real_images.to('cuda')

    for i in range(5, 105, 5):
        print(f"Calculating FID score for epoch {i}")
        # Load the model

        model = DDPM(gt=gt, noise_schedule_choice=scheduler,
                     betas=(1e-4, 0.02), n_T=1000)
        # Load the model checkpoint
        model.load_state_dict(torch.load(f"./{scheduler}_checkpoints/" +
                                         f"{scheduler}_epoch{i:03d}.pt"))

        # model.load_state_dict(torch.load("./temp_checkpoints/" +
        #                                  f"epoch{i:03d}.pt"))
        model.to('cuda')

        # Sample 1000 images from the model
        with torch.no_grad():
            fake_images = model.sample(1000, (1, 28, 28), device='cuda')

        real_images_rgb, fake_images_rgb = to_rgb(real_images), \
            to_rgb(fake_images)

        fid = FrechetInceptionDistance(normalize=True)
        fid.to('cuda')

        for j in range(10):
            fid.update(real_images_rgb[j*100:(j+1)*100], real=True)
            fid.update(fake_images_rgb[j*100:(j+1)*100], real=False)

        # Save the calculated FID score to a csv file
        with open(f"fid_scores_{scheduler}.csv", "a") as f:
            f.write(f"{i},{fid.compute().item()}\n")


calculate_fid_score("ddpm")
# calculate_fid_score("cosine")
# calculate_fid_score("constant")
# calculate_fid_score("inverse")
# calculate_fid_score("temp")
