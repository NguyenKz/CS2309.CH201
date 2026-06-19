"""Generate images using DDPM sampling (full T=1000 steps).

Reverse process: start from pure noise x_T ~ N(0, I),
iteratively denoise x_T → x_{T-1} → ... → x_0.
"""

import torch
from tqdm import tqdm

from diffusion import LinearSchedule, UNet
from diffusion.utils import set_seed, get_device, plot_grid


@torch.no_grad()
def sample_ddpm(model, schedule, n_samples=64, device="cpu"):
    """Full DDPM sampling: T denoising steps."""
    model.eval()
    x = torch.randn(n_samples, 1, 28, 28, device=device)

    for t in tqdm(range(schedule.T - 1, -1, -1), desc="DDPM sampling"):
        x = schedule.p_sample(model, x, t)

    # Clamp to [-1, 1] and scale to [0, 1]
    return (x.clamp(-1, 1) + 1) / 2


def main():
    set_seed(42)
    device = get_device()

    model = UNet(in_ch=1, base_ch=32, time_dim=128).to(device)
    model.load_state_dict(torch.load("assets/ddpm_mnist.pt", map_location=device,
                                      weights_only=True))
    schedule = LinearSchedule(T=1000, device=device)

    print("  Generating 64 samples with DDPM (1000 steps)...")
    samples = sample_ddpm(model, schedule, n_samples=64, device=device)
    plot_grid(samples, "DDPM Samples (T=1000 steps)", "assets/ddpm_samples.png")

    # Also save denoising progression
    print("  Generating denoising progression...")
    model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)
    snapshots = []
    save_at = [999, 800, 600, 400, 200, 100, 50, 0]

    for t in range(999, -1, -1):
        x = schedule.p_sample(model, x, t)
        if t in save_at:
            snapshots.append((x.clamp(-1, 1) + 1) / 2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 2.5))
    fig.suptitle("Denoising Process: x_T → x_0", fontsize=13)
    for i, (snap, t) in enumerate(zip(snapshots, save_at)):
        axes[i].imshow(snap[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"t={t}")
        axes[i].axis("off")
    fig.tight_layout()
    fig.savefig("assets/denoising_steps.png", dpi=150)
    plt.close(fig)
    print("  Plot saved → assets/denoising_steps.png")


if __name__ == "__main__":
    main()
