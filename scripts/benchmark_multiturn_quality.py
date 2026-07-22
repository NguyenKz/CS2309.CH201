#!/usr/bin/env python3
"""Đo suy giảm do VAE roundtrip: naive toàn ảnh so với Hybrid local composite."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms.functional import pil_to_tensor

from hybrid_editing import hybrid_composite


ROOT = Path(__file__).resolve().parent.parent
MODEL_SIZE = 512


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def letterbox(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    image = image.convert("RGB")
    width, height = image.size
    scale = MODEL_SIZE / max(width, height)
    content_w = int(round(width * scale))
    content_h = int(round(height * scale))
    left = (MODEL_SIZE - content_w) // 2
    top = (MODEL_SIZE - content_h) // 2
    resized = image.resize((content_w, content_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (MODEL_SIZE, MODEL_SIZE), (127, 127, 127))
    canvas.paste(resized, (left, top))
    return canvas, (left, top, content_w, content_h)


def unletterbox(
    image: Image.Image,
    meta: tuple[int, int, int, int],
    output_size: tuple[int, int],
) -> Image.Image:
    left, top, content_w, content_h = meta
    cropped = image.crop((left, top, left + content_w, top + content_h))
    return cropped.resize(output_size, Image.Resampling.LANCZOS)


@torch.no_grad()
def vae_roundtrip(image: Image.Image, vae: AutoencoderKL, device: str) -> Image.Image:
    tensor = pil_to_tensor(image).unsqueeze(0).to(device, dtype=torch.float32) / 127.5 - 1
    latent = vae.encode(tensor).latent_dist.mode() * vae.config.scaling_factor
    decoded = vae.decode(latent / vae.config.scaling_factor).sample
    array = ((decoded[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array.astype(np.uint8))


def image_tensor(image: Image.Image, device: str, size: int | None = None) -> torch.Tensor:
    if size is not None:
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return pil_to_tensor(image).unsqueeze(0).to(device, dtype=torch.float32) / 255.0


def masked_psnr(reference: Image.Image, result: Image.Image, keep_mask: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float32) / 255.0
    out = np.asarray(result, dtype=np.float32) / 255.0
    values = (ref - out)[keep_mask]
    mse = float(np.mean(values * values))
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def peak_memory_mb(device: str) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1024**2
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT / "data/PIE-Bench-smoke/annotation_images/0_random_140/smoke_woman.jpg",
    )
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()

    device = get_device()
    original = Image.open(args.image).convert("RGB")
    width, height = original.size
    mask_arr = np.zeros((height, width), np.uint8)
    mask_arr[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 255
    edit_mask = Image.fromarray(mask_arr, mode="L")
    keep_mask = mask_arr == 0

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae").to(
        device, dtype=torch.float32
    )
    vae.eval()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = None if args.skip_lpips else LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device)

    naive = original.copy()
    hybrid = original.copy()
    rows: list[dict] = []
    started = time.perf_counter()
    for turn in range(1, args.turns + 1):
        for mode in ("naive", "hybrid"):
            current = naive if mode == "naive" else hybrid
            boxed, meta = letterbox(current)
            sync(device)
            t0 = time.perf_counter()
            reconstructed = vae_roundtrip(boxed, vae, device)
            sync(device)
            runtime = time.perf_counter() - t0
            candidate = unletterbox(reconstructed, meta, original.size)
            if mode == "naive":
                naive = candidate
                result = naive
            else:
                hybrid = hybrid_composite(
                    hybrid,
                    candidate,
                    edit_mask,
                    mode="local",
                    dilation=0,
                    blur=0,
                )
                result = hybrid
            ref_t = image_tensor(original, device)
            out_t = image_tensor(result, device)
            lpips_value = None
            if lpips is not None:
                lpips_value = float(
                    lpips(image_tensor(original, device, 256), image_tensor(result, device, 256))
                )
            rows.append(
                {
                    "turn": turn,
                    "mode": mode,
                    "outside_mask_psnr": masked_psnr(original, result, keep_mask),
                    "ssim": float(ssim(out_t, ref_t)),
                    "lpips": lpips_value,
                    "runtime_s": runtime,
                    "peak_memory_mb": peak_memory_mb(device),
                }
            )

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / "experimental_data" / f"multiturn_hybrid_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    naive.save(out_dir / "naive_turn5.png")
    hybrid.save(out_dir / "hybrid_turn5.png")
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    final_naive = next(row for row in reversed(rows) if row["mode"] == "naive")
    final_hybrid = next(row for row in reversed(rows) if row["mode"] == "hybrid")
    report = (
        "# Benchmark Hybrid Multi-turn\n\n"
        f"- Ảnh: `{args.image}` ({width}×{height})\n"
        f"- Device: `{device}`; số lượt: {args.turns}; tổng thời gian: "
        f"{time.perf_counter() - started:.2f}s\n"
        "- Thí nghiệm cô lập suy giảm VAE: không semantic edit; mask giữa ảnh chiếm 25%.\n\n"
        "## Kết quả lượt cuối\n\n"
        "| Mode | PSNR ngoài mask | SSIM | LPIPS |\n"
        "|---|---:|---:|---:|\n"
        f"| Naive | {final_naive['outside_mask_psnr']:.4f} | "
        f"{final_naive['ssim']:.6f} | {final_naive['lpips']} |\n"
        f"| Hybrid | {final_hybrid['outside_mask_psnr']:.4f} | "
        f"{final_hybrid['ssim']:.6f} | {final_hybrid['lpips']} |\n\n"
        "Hybrid giữ nguyên pixel ngoài mask; Naive đưa toàn ảnh qua VAE ở mọi lượt.\n"
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"[multiturn] report: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
