import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_grid(images, title, save_path, nrow=8):
    """Plot a grid of images. images: list/tensor of [C, H, W] or [H, W]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(images), nrow * nrow)
    fig, axes = plt.subplots(nrow, nrow, figsize=(8, 8))
    fig.suptitle(title, fontsize=14)
    for i, ax in enumerate(axes.flat):
        if i < n:
            img = images[i]
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if img.ndim == 3:
                img = img.squeeze(0) if img.shape[0] == 1 else img.transpose(1, 2, 0)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {save_path}")
