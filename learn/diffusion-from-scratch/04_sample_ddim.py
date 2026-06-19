"""Generate images using DDIM sampling (fast, 50 steps instead of 1000).

DDIM (Denoising Diffusion Implicit Models) enables:
  - Deterministic sampling (eta=0)
  - Fewer steps via subsequence of timesteps
  - Same trained model, just different sampling procedure
"""

import torch
from tqdm import tqdm

from diffusion import LinearSchedule, UNet
from diffusion.utils import set_seed, get_device, plot_grid


@torch.no_grad()
def sample_ddim(model, schedule, n_samples=64, num_steps=50, eta=0.0, device="cpu"):
    """DDIM sampling with configurable number of steps."""
    model.eval()
    x = torch.randn(n_samples, 1, 28, 28, device=device)

    # Create evenly-spaced timestep subsequence
    step_size = schedule.T // num_steps
    timesteps = list(range(schedule.T - 1, -1, -step_size))

    for i in tqdm(range(len(timesteps)), desc=f"DDIM sampling ({num_steps} steps)"):
        t = timesteps[i]
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        x = schedule.ddim_sample(model, x, t, t_prev, eta=eta)

    return (x.clamp(-1, 1) + 1) / 2


def main():
    set_seed(42)
    device = get_device()

    model = UNet(in_ch=1, base_ch=32, time_dim=128).to(device)
    model.load_state_dict(torch.load("assets/ddpm_mnist.pt", map_location=device,
                                      weights_only=True))
    schedule = LinearSchedule(T=1000, device=device)

    # DDIM with 50 steps (deterministic)
    print("  Generating 64 samples with DDIM (50 steps, eta=0)...")
    samples_50 = sample_ddim(model, schedule, n_samples=64, num_steps=50,
                              eta=0.0, device=device)
    plot_grid(samples_50, "DDIM Samples (50 steps, deterministic)",
              "assets/ddim_50_samples.png")

    # DDIM with 20 steps
    print("  Generating 64 samples with DDIM (20 steps, eta=0)...")
    samples_20 = sample_ddim(model, schedule, n_samples=64, num_steps=20,
                              eta=0.0, device=device)
    plot_grid(samples_20, "DDIM Samples (20 steps, deterministic)",
              "assets/ddim_20_samples.png")


if __name__ == "__main__":
    main()
