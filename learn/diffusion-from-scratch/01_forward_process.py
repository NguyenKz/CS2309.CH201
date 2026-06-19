"""Visualize the forward diffusion process: gradually adding noise to an image."""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from diffusion.schedule import LinearSchedule
from diffusion.utils import set_seed


def main():
    set_seed(42)
    schedule = LinearSchedule(T=1000)

    # Load one MNIST image
    dataset = datasets.MNIST("data", train=True, download=True,
                             transform=transforms.ToTensor())
    x_0 = dataset[3][0].unsqueeze(0) * 2 - 1  # Scale to [-1, 1]

    # Show noising at different timesteps
    timesteps = [0, 50, 100, 200, 400, 600, 800, 999]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("Forward Diffusion Process: q(x_t | x_0)", fontsize=14)

    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t])
        x_t, _ = schedule.q_sample(x_0, t_tensor)
        img = (x_t.squeeze().numpy() + 1) / 2  # Back to [0, 1]

        ax = axes[i // 4, i % 4]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t = {t}\nᾱ = {schedule.alpha_bar[t]:.3f}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig("assets/forward_process.png", dpi=150)
    plt.close(fig)
    print("  Plot saved → assets/forward_process.png")

    # Plot alpha_bar curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(schedule.alpha_bar.numpy(), color="steelblue", linewidth=2)
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("ᾱ_t (cumulative signal retention)")
    ax.set_title("Noise Schedule: ᾱ_t over time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("assets/noise_schedule.png", dpi=150)
    plt.close(fig)
    print("  Plot saved → assets/noise_schedule.png")


if __name__ == "__main__":
    main()
