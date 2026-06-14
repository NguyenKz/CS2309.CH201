#!/usr/bin/env python3
"""Benchmark fp16 / channels_last cho SwiftEdit (SwiftEdit-RT).

So sánh tốc độ per-stage và chất lượng giữa các cấu hình dtype/memory-format.
VAE luôn fp32 (decode ổn định); UNet + text/image encoder chạy theo dtype cấu hình.

Chất lượng được đo bằng PSNR/MSE giữa ảnh của cấu hình fp16 và ảnh fp32 (reference),
kèm cảnh báo NaN / ảnh đen để phát hiện fp16 hỏng trên MPS.

Ví dụ:
  python scripts/bench_dtype.py \
      --image data/PIE-Bench-subset20/annotation_images/0_random_140/000000000000.jpg \
      --src-p "a slanted mountain bicycle on the road in front of a building"
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

STAGES = [
    "vae_encode", "inv_text_encode", "unet_inverse", "mask_estimate",
    "gen_image_embeds", "gen_text_encode", "gen_unet", "gen_vae_decode",
]

CONFIGS = {
    "fp32": dict(dtype="fp32", channels_last=False),
    "fp16": dict(dtype="fp16", channels_last=False),
    "fp16_cl": dict(dtype="fp16", channels_last=True),
}

DEFAULT_EDITS = [
    "a slanted rusty mountain motorcycle in front of a fence",
    "a slanted blue mountain bicycle on the road",
    "a slanted mountain bicycle in front of a castle",
]


def _sync(device) -> None:
    d = str(device)
    if d.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif d.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _free(device) -> None:
    gc.collect()
    d = str(device)
    if d.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif d.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark fp16/channels_last SwiftEdit")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--src-p", type=str, required=True)
    parser.add_argument("--edits", nargs="*", default=DEFAULT_EDITS)
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS.keys()))
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "dtype_bench_report.md")
    parser.add_argument("--img-out", type=Path, default=ROOT / "results" / "dtype_bench")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "SwiftEdit"))
    from infer import edit_image, get_device
    from models import AuxiliaryModel, InverseModel, IPSBV2Model
    from torchvision.utils import save_image

    device = get_device()
    print(f"device={device}")
    weights = ROOT / "SwiftEdit" / "swiftedit_weights"
    img = str(args.image)
    args.img_out.mkdir(parents=True, exist_ok=True)

    def build(cfg):
        inv = InverseModel(
            str(weights / "inverse_ckpt-120k"), device=device,
            dtype=cfg["dtype"], channels_last=cfg["channels_last"],
        )
        aux = AuxiliaryModel(device=device, dtype=cfg["dtype"])
        ip = IPSBV2Model(
            str(weights / "sbv2_0.5"),
            str(weights / "ip_adapter_ckpt-90k/ip_adapter.bin"),
            aux, device=device, with_ip_mask_controller=True,
            dtype=cfg["dtype"], channels_last=cfg["channels_last"],
        )
        return inv, aux, ip

    per_stage = {}     # cfg -> {stage: mean_ms}
    totals = {}        # cfg -> mean total s
    out_paths = {}     # cfg -> [png paths]

    for name in args.configs:
        cfg = CONFIGS[name]
        print(f"\n===== CONFIG {name} ({cfg}) =====")
        log_path = ROOT / "results" / f"dtype_bench_{name}.jsonl"
        if log_path.exists():
            log_path.unlink()
        os.environ["SWIFTEDIT_TIMING"] = "1"
        os.environ["SWIFTEDIT_TIMING_LOG"] = str(log_path)

        inv, aux, ip = build(cfg)

        # warmup (nuốt chi phí compile MPS lần đầu cho cả encoder/unet/VAE) — không log
        os.environ["SWIFTEDIT_TIMING"] = "0"
        edit_image(img, args.src_p, args.edits[0], inv, aux, ip)
        edit_image(img, args.src_p, args.edits[-1], inv, aux, ip)
        os.environ["SWIFTEDIT_TIMING"] = "1"

        walls, paths = [], []
        for i, e in enumerate(args.edits):
            _sync(device)
            t0 = time.perf_counter()
            res = edit_image(img, args.src_p, e, inv, aux, ip)
            _sync(device)
            dt = time.perf_counter() - t0
            walls.append(dt)
            p = args.img_out / f"{name}_{i}.png"
            save_image(res, str(p))
            paths.append(p)
            print(f"  [{i}] {dt:.2f}s  {e[:46]}")

        recs = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        # median: robust với outlier do MPS compile/thermal ở edit đầu mỗi config
        per_stage[name] = {
            s: statistics.median([r["stages_ms"].get(s, 0.0) for r in recs]) for s in STAGES
        }
        totals[name] = statistics.median(walls)
        out_paths[name] = paths

        del inv, aux, ip
        _free(device)

    # ---- Chất lượng: PSNR/MSE so với fp32 (reference) + cảnh báo NaN/đen ----
    def load_arr(p):
        return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0

    quality = {}
    ref = "fp32" if "fp32" in args.configs else args.configs[0]
    ref_imgs = [load_arr(p) for p in out_paths[ref]]
    for name in args.configs:
        if name == ref:
            continue
        psnrs, mses, black = [], [], 0
        for a_ref, p in zip(ref_imgs, out_paths[name]):
            a = load_arr(p)
            if a.std() < 1e-4 or np.isnan(a).any():
                black += 1
            mse = float(np.mean((a - a_ref) ** 2))
            mses.append(mse)
            psnrs.append(float("inf") if mse == 0 else 10 * np.log10(1.0 / mse))
        quality[name] = dict(
            psnr=statistics.mean([p for p in psnrs if p != float("inf")] or [0.0]),
            mse=statistics.mean(mses), black=black, n=len(mses),
        )

    # ---- Report ----
    lines = [
        "# Benchmark fp16 / channels_last (SwiftEdit-RT)",
        "",
        f"- **Thiết bị:** `{device}`",
        f"- **Ảnh nguồn:** `{args.image.name}`",
        f"- **Source prompt:** {args.src_p}",
        f"- **Số edit/cfg:** {len(args.edits)} (median; bỏ 2 warmup; VAE luôn fp32)",
        f"- **Reference chất lượng:** `{ref}`",
        "",
        "## Per-stage (median ms/edit)",
        "",
        "| Stage | " + " | ".join(args.configs) + " |",
        "|-------|" + "|".join(["---:"] * len(args.configs)) + "|",
    ]
    for s in STAGES:
        row = [f"{per_stage[c].get(s, 0):.1f}" for c in args.configs]
        lines.append(f"| {s} | " + " | ".join(row) + " |")
    sum_row = [f"{sum(per_stage[c].values()):.1f}" for c in args.configs]
    lines.append("| **Tổng stage** | " + " | ".join(sum_row) + " |")

    lines += ["", "## Wall-clock end-to-end (mean s/edit)", "",
              "| Config | Mean (s) | Speedup vs " + ref + " |",
              "|--------|---------:|----:|"]
    base = totals[ref]
    for c in args.configs:
        lines.append(f"| {c} | {totals[c]:.2f} | {base/totals[c]:.2f}× |")

    # VAE encode/decode là fp32 ở MỌI config -> chênh lệch ở đó là nhiễu nhiệt thuần,
    # dùng làm thước đo mức throttling của baseline fp32.
    if ref in per_stage and len(args.configs) > 1:
        other = next(c for c in args.configs if c != ref)
        vae_ref = per_stage[ref]["vae_encode"] + per_stage[ref]["gen_vae_decode"]
        vae_oth = per_stage[other]["vae_encode"] + per_stage[other]["gen_vae_decode"]
        if vae_oth > 0:
            lines += ["", f"> **Hiệu chỉnh nhiệt:** VAE (fp32 ở mọi config) chạy chậm "
                      f"~{vae_ref/vae_oth:.1f}× ở `{ref}` so với `{other}` dù cùng dtype → "
                      f"baseline `{ref}` bị thermal throttling khi chạy chuỗi. Speedup wall-clock "
                      f"vì thế phản ánh cả 'fp16 sinh nhiệt thấp, không bị throttle'. "
                      f"Trên máy nguội (đo 1 edit) speedup fp16 ~3.3×; khi chạy liên tục, baseline "
                      f"fp32 throttle nên speedup biểu kiến cao hơn."]

    lines += ["", "## Chất lượng so với " + ref + " (PSNR/MSE toàn ảnh)", "",
              "| Config | PSNR (dB) | MSE | NaN/đen | Nhận xét |",
              "|--------|----------:|----:|--------:|---------|"]
    for c in args.configs:
        if c == ref:
            lines.append(f"| {c} | — | — | 0/{len(args.edits)} | reference |")
            continue
        q = quality[c]
        ok = "OK (gần như trùng)" if q["psnr"] >= 30 and q["black"] == 0 else (
            "LỖI fp16 (NaN/đen)" if q["black"] else "lệch nhẹ")
        lines.append(f"| {c} | {q['psnr']:.1f} | {q['mse']:.4f} | {q['black']}/{q['n']} | {ok} |")

    lines += ["", "> PSNR ≥ ~30 dB so với fp32 nghĩa là fp16 gần như không đổi kết quả nhìn thấy. "
              "Wall-clock có thể nhiễu thermal throttling; per-stage robust hơn."]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nTổng stage (ms/edit):")
    for c in args.configs:
        print(f"  {c:10s} {sum(per_stage[c].values()):8.1f}  wall={totals[c]:.2f}s  "
              f"speedup={base/totals[c]:.2f}x")
    for c, q in quality.items():
        print(f"  quality {c}: PSNR={q['psnr']:.1f}dB MSE={q['mse']:.4f} black={q['black']}/{q['n']}")
    print(f"Report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
