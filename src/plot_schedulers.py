import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from schedulers import cosine_beta_schedule, ddpm_schedules, \
    inverse_linear_schedule, constant_noise_schedule

ddpm_schedule = ddpm_schedules()
cosine_schedule = cosine_beta_schedule()
inverse_schedule = inverse_linear_schedule()
constant_schedule = constant_noise_schedule()

plt.figure(figsize=(6, 5))

# Plot the alpha values for each schedule
plt.title("Alpha values for each schedule")
plt.plot(ddpm_schedule["alpha_t"].numpy(), label="DDPM")
plt.plot(cosine_schedule["alpha_t"].numpy(), label="Cosine")
plt.plot(inverse_schedule["alpha_t"].numpy(), label="Inverse")
plt.plot(constant_schedule["alpha_t"].numpy(), label="Constant")
plt.legend()
plt.tight_layout()
plt.savefig("./figures/alpha_values.png")

# Plot the beta values for each schedule
plt.figure(figsize=(6, 5))
plt.title("Beta values for each schedule")
plt.plot(ddpm_schedule["beta_t"].numpy(), label="DDPM")
plt.plot(cosine_schedule["beta_t"].numpy(), label="Cosine")
plt.plot(inverse_schedule["beta_t"].numpy(), label="Inverse")
plt.plot(constant_schedule["beta_t"].numpy(), label="Constant")
plt.legend()
plt.tight_layout()
plt.savefig("./figures/beta_values.png")

# Get an image from the MNIST dataset
mnist = MNIST(root="./data", download=True, transform=transforms.ToTensor())
image, _ = mnist[0]

fig, axs = plt.subplots(nrows=4, ncols=20, figsize=(80, 16))

for t in range(0, 1000, 50):
    image_1 = image.clone()
    image_2 = image.clone()
    image_3 = image.clone()
    image_4 = image.clone()

    # Diffuse the image
    z_t1 = torch.sqrt(inverse_schedule["alpha_t"][t]) * image_1 + \
        torch.randn_like(image_1) * (torch.sqrt(1 - (inverse_schedule["alpha_t"][t])))
    z_t2 = torch.sqrt(constant_schedule["alpha_t"][t]) * image_2 + \
        torch.randn_like(image_2) * (torch.sqrt(1 - (constant_schedule["alpha_t"][t])))
    z_t3 = torch.sqrt(ddpm_schedule["alpha_t"][t]) * image_3 + \
        torch.randn_like(image_3) * (torch.sqrt(1 - (ddpm_schedule["alpha_t"][t])))
    z_t4 = torch.sqrt(cosine_schedule["alpha_t"][t]) * image_4 + \
        torch.randn_like(image_4) * (torch.sqrt(1 - (cosine_schedule["alpha_t"][t])))

    # Hide the axis ticks
    axs[0, t // 50].axis("off")
    axs[1, t // 50].axis("off")
    axs[2, t // 50].axis("off")
    axs[3, t // 50].axis("off")

    axs[0, t // 50].imshow(z_t1[0], cmap="viridis")
    axs[1, t // 50].imshow(z_t2[0], cmap="viridis")
    axs[2, t // 50].imshow(z_t3[0], cmap="viridis")
    axs[3, t // 50].imshow(z_t4[0], cmap="viridis")
fig.suptitle('Comparison of diffused images every 50 diffusion steps', fontsize=50)
plt.savefig("./figures/diffused_images.png")
