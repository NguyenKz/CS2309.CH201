"""Run the full pipeline: visualize → train → sample (DDPM + DDIM)."""

import time
import importlib


def main():
    print("=" * 55)
    print("  diffusion-from-scratch — Full Pipeline")
    print("=" * 55)

    steps = [
        ("01 Forward Process", "01_forward_process"),
        ("02 Train DDPM", "02_train_ddpm"),
        ("03 Sample DDPM", "03_sample_ddpm"),
        ("04 Sample DDIM", "04_sample_ddim"),
    ]

    for name, mod_name in steps:
        print(f"\n{'─' * 50}")
        print(f"  {name}")
        print(f"{'─' * 50}")
        start = time.time()
        mod = importlib.import_module(mod_name)
        mod.main() if hasattr(mod, "main") else mod.train()
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s")

    print(f"\n{'=' * 55}")
    print("  All plots saved to assets/")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
