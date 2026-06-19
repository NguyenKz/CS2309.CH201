"""Train a DDPM (Denoising Diffusion Probabilistic Model) on MNIST.

Training objective: predict the noise added at each timestep.
Loss = MSE(epsilon_predicted, epsilon_actual)
"""

import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion import LinearSchedule, UNet
from diffusion.utils import set_seed, get_device


def train(num_epochs=15, batch_size=128, lr=2e-4, T=1000, seed=42):
    set_seed(seed)
    device = get_device()
    print(f"  Device: {device}")

    # Data: scale to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Model & schedule
    model = UNet(in_ch=1, base_ch=32, time_dim=128).to(device)
    schedule = LinearSchedule(T=T, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  U-Net parameters: {num_params:,}")

    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start = time.time()

        for x_0, _ in loader:
            x_0 = x_0.to(device)

            # Sample random timesteps
            t = torch.randint(0, T, (x_0.shape[0],), device=device)

            # Forward process: add noise
            noise = torch.randn_like(x_0)
            x_t, _ = schedule.q_sample(x_0, t, noise)

            # Predict noise
            noise_pred = model(x_t, t)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        elapsed = time.time() - start
        print(f"  Epoch {epoch+1}/{num_epochs}  Loss: {avg_loss:.4f}  ({elapsed:.1f}s)")

    # Save model
    torch.save(model.state_dict(), "assets/ddpm_mnist.pt")
    print("  Model saved → assets/ddpm_mnist.pt")

    # Plot loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, "o-", color="steelblue", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("DDPM Training Loss — MNIST")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("assets/training_loss.png", dpi=150)
    plt.close(fig)
    print("  Plot saved → assets/training_loss.png")

    return model, schedule


if __name__ == "__main__":
    train()
