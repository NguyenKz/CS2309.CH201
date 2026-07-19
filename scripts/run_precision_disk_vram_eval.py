#!/usr/bin/env python3
"""Đánh giá disk + memory + chất lượng: fp32 vs fp16_disk (Phase A) và fp4 (Phase B / Colab).

Xuất thư mục experimental_data/precision_disk_vram_<date>/ gồm:
  disk.csv, memory.csv, quality.csv, quality_summary.csv,
  run_meta.json, report.md, edited_images/, COMPARE_WITH_PREV.md,
  bundle.zip (tải về để so báo cáo)

Ví dụ Mac (Phase A):
  python scripts/run_precision_disk_vram_eval.py \\
      --configs baseline_fp32,fp16_disk --n-images 4 --edits-per-image 3

Colab T4 (Phase B — thêm fp4):
  python scripts/run_precision_disk_vram_eval.py \\
      --configs baseline_fp32,fp16_disk,fp4_from_fp16 \\
      --weights-fp16 /content/drive/MyDrive/.../swiftedit_weights_fp16
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

# Fallback CHỈ khi không có mapping_file (ảnh xe đạp demo).
BICYCLE_SRC = "a slanted mountain bicycle on the road in front of a building"
BICYCLE_EDITS = [
    "a slanted rusty mountain motorcycle in front of a fence",
    "a slanted blue mountain bicycle on the road in front of a building",
    "a slanted mountain bicycle on the road in front of a castle",
]

# Khớp naming báo cáo cũ 2026-06-17 để người đọc dễ map cột.
PREV_BENCH_HINT = {
    "baseline_fp32": "baseline_fp32 (quality_speed_bench_2026-06-17)",
    "fp16_disk": "gần improved_fp16_cache nhưng weights fp16 trên disk + torch_dtype load",
    "fp4_from_fp16": "gần improved_fp4_cache (runtime bitsandbytes sau load fp16)",
}


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _fmt_mb(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.0f}"


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _free(device: str) -> None:
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _reset_peak(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _peak_alloc_mb(device: str) -> float | None:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    if device.startswith("mps") and torch.backends.mps.is_available():
        # MPS không có max_memory_allocated ổn định như CUDA — dùng current.
        try:
            return torch.mps.current_allocated_memory() / (1024**2)
        except Exception:
            return None
    return None


def _driver_used_mb(device: str) -> float | None:
    if device.startswith("cuda") and torch.cuda.is_available():
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip().splitlines()[0])
    if device.startswith("mps") and torch.backends.mps.is_available():
        try:
            return torch.mps.driver_allocated_memory() / (1024**2)
        except Exception:
            return None
    return None


def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _jobs_from_mapping(bench_root: Path, n_images: int) -> list[tuple[Path, str, str]]:
    """Mỗi entry mapping → (image, original_prompt, editing_prompt)."""
    mapping_path = bench_root / "mapping_file.json"
    if not mapping_path.is_file():
        return []
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    jobs: list[tuple[Path, str, str]] = []
    for _key, meta in data.items():
        rel = meta.get("image_path") or ""
        src_p = (meta.get("original_prompt") or "").strip()
        edit_p = (meta.get("editing_prompt") or "").strip()
        if not rel or not src_p or not edit_p:
            continue
        img = bench_root / "annotation_images" / rel
        if not img.is_file():
            # một số layout để path phẳng
            img = bench_root / rel
        if not img.is_file():
            continue
        jobs.append((img, src_p, edit_p))
        if len(jobs) >= n_images:
            break
    return jobs


def _pick_jobs(n_images: int, edits_per_image: int) -> list[tuple[Path, str, str]]:
    """Danh sách (path, source_prompt, edit_prompt) — prompt KHỚP ảnh.

    Ưu tiên mapping PIE-Bench-smoke / subset20 (tránh gán prompt xe đạp lên ảnh người).
    Fallback: imgs_demo/02.jpg + prompt bicycle.
    """
    for base in (
        ROOT / "data" / "PIE-Bench-smoke",
        ROOT / "data" / "PIE-Bench-subset20",
        ROOT / "data" / "PIE-Bench",
    ):
        jobs = _jobs_from_mapping(base, n_images)
        if jobs:
            if edits_per_image > 1:
                print(
                    f"Dùng mapping {base.name}: mỗi ảnh 1 edit (original→editing trong JSON). "
                    f"--edits-per-image={edits_per_image} bị bỏ qua để tránh prompt sai ngữ cảnh."
                )
            print(f"Jobs từ mapping ({base}):")
            for p, s, e in jobs:
                print(f"  {p.name}: '{s}' → '{e}'")
            return jobs

    demo = ROOT / "SwiftEdit" / "assets" / "imgs_demo" / "02.jpg"
    if demo.is_file():
        jobs = []
        for edit in BICYCLE_EDITS[: max(1, edits_per_image)]:
            jobs.append((demo, BICYCLE_SRC, edit))
        print(f"Fallback imgs_demo: {demo.name} + bicycle prompts ({len(jobs)} edits)")
        return jobs
    raise FileNotFoundError("Không tìm thấy PIE-Bench mapping hoặc imgs_demo/02.jpg")


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.dim() == 4:
        t = t[-1]
    arr = t.clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def _psnr_mse(a: Image.Image, b: Image.Image) -> tuple[float, float]:
    xa = np.asarray(a.convert("RGB").resize((512, 512)), dtype=np.float64) / 255.0
    xb = np.asarray(b.convert("RGB").resize((512, 512)), dtype=np.float64) / 255.0
    mse = float(np.mean((xa - xb) ** 2))
    if mse <= 1e-12:
        return 99.0, mse
    psnr = 10.0 * np.log10(1.0 / mse)
    return psnr, mse


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _zip_bundle(out_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            if p.is_file() and p.name != zip_path.name:
                zf.write(p, arcname=str(p.relative_to(out_dir)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Precision disk/memory/quality eval")
    parser.add_argument(
        "--configs",
        type=str,
        default="baseline_fp32,fp16_disk",
        help="CSV: baseline_fp32,fp16_disk[,fp4_from_fp16]",
    )
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--edits-per-image", type=int, default=3)
    parser.add_argument(
        "--weights-fp32",
        type=Path,
        default=ROOT / "SwiftEdit" / "swiftedit_weights",
    )
    parser.add_argument(
        "--weights-fp16",
        type=Path,
        default=ROOT / "SwiftEdit" / "swiftedit_weights_fp16",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Thư mục output (mặc định experimental_data/precision_disk_vram_<stamp>)",
    )
    parser.add_argument(
        "--src-p",
        type=str,
        default=None,
        help="Override source prompt cho mọi job (mặc định: lấy từ mapping)",
    )
    parser.add_argument(
        "--edit-p",
        type=str,
        default=None,
        help="Override edit prompt cho mọi job (mặc định: lấy từ mapping)",
    )
    args = parser.parse_args()

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    # baseline_fp32 luôn chạy trước để làm reference PSNR
    if "baseline_fp32" in config_names:
        config_names = ["baseline_fp32"] + [c for c in config_names if c != "baseline_fp32"]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = args.out or (ROOT / "experimental_data" / f"precision_disk_vram_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "edited_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT / "SwiftEdit"))
    from infer import edit_image, get_device
    from models import AuxiliaryModel, InverseModel, IPSBV2Model

    device = get_device()
    print(f"device={device} configs={config_names}")
    print(f"out={out_dir}")

    # --- disk ---
    disk_rows = []
    for label, path in [
        ("swiftedit_weights_fp32", args.weights_fp32),
        ("swiftedit_weights_fp16", args.weights_fp16),
    ]:
        b = _dir_bytes(path)
        disk_rows.append(
            {
                "label": label,
                "path": str(path),
                "bytes": b,
                "gib": round(b / (1024**3), 4),
                "exists": path.exists(),
            }
        )
    _write_csv(out_dir / "disk.csv", disk_rows)
    fp32_b = next(r["bytes"] for r in disk_rows if r["label"].endswith("fp32"))
    fp16_b = next(r["bytes"] for r in disk_rows if r["label"].endswith("fp16"))
    disk_save_pct = (1 - fp16_b / fp32_b) * 100 if fp32_b and fp16_b else None
    print(
        f"disk fp32={fp32_b/1e9:.2f}GB fp16={fp16_b/1e9:.2f}GB "
        f"save={disk_save_pct:.1f}%" if disk_save_pct is not None else "disk: fp16 tree thiếu?"
    )

    if "fp16_disk" in config_names or "fp4_from_fp16" in config_names:
        if not (args.weights_fp16 / "sbv2_0.5").is_dir():
            print(
                f"Thiếu weights fp16 tại {args.weights_fp16} — chạy:\n"
                f"  python scripts/convert_weights_fp16.py",
                file=sys.stderr,
            )
            return 1

    jobs = _pick_jobs(args.n_images, args.edits_per_image)
    if args.src_p or args.edit_p:
        jobs = [
            (p, args.src_p or s, args.edit_p or e) for p, s, e in jobs
        ]

    cfg_map = {
        "baseline_fp32": dict(
            weights=args.weights_fp32,
            dtype="fp32",
            channels_last=False,
            quant=None,
        ),
        "fp16_disk": dict(
            weights=args.weights_fp16,
            dtype="fp16",
            channels_last=device.startswith("cuda"),
            quant=None,
        ),
        "fp4_from_fp16": dict(
            weights=args.weights_fp16,
            dtype="fp16",
            channels_last=True,
            quant="fp4",
        ),
    }

    memory_rows: list[dict] = []
    quality_rows: list[dict] = []
    ref_images: dict[tuple[int, int], Image.Image] = {}

    for cname in config_names:
        if cname not in cfg_map:
            print(f"Bỏ qua config lạ: {cname}", file=sys.stderr)
            continue
        if cname == "fp4_from_fp16" and not device.startswith("cuda"):
            print(
                "SKIP fp4_from_fp16 — cần CUDA (Phase B Colab T4). bitsandbytes không dùng MPS.",
                file=sys.stderr,
            )
            memory_rows.append(
                {
                    "config": cname,
                    "phase": "skipped",
                    "device": device,
                    "peak_alloc_mb": None,
                    "driver_used_mb": None,
                    "note": "fp4 requires CUDA",
                }
            )
            continue

        cfg = cfg_map[cname]
        wroot: Path = cfg["weights"]
        print(f"\n=== load {cname} weights={wroot} dtype={cfg['dtype']} quant={cfg['quant']} ===")
        _free(device)
        _reset_peak(device)
        t_load0 = time.perf_counter()

        inv = InverseModel(
            str(wroot / "inverse_ckpt-120k"),
            device=device,
            dtype=cfg["dtype"],
            channels_last=cfg["channels_last"],
            quant=cfg["quant"],
        )
        aux = AuxiliaryModel(device=device, dtype=cfg["dtype"])
        ip = IPSBV2Model(
            str(wroot / "sbv2_0.5"),
            str(wroot / "ip_adapter_ckpt-90k" / "ip_adapter.bin"),
            aux,
            device=device,
            with_ip_mask_controller=True,
            dtype=cfg["dtype"],
            channels_last=cfg["channels_last"],
            quant=cfg["quant"],
        )
        _sync(device)
        load_s = time.perf_counter() - t_load0
        peak_load = _peak_alloc_mb(device)
        driver_load = _driver_used_mb(device)
        memory_rows.append(
            {
                "config": cname,
                "phase": "after_load",
                "device": device,
                "load_seconds": round(load_s, 2),
                "peak_alloc_mb": round(peak_load, 2) if peak_load is not None else None,
                "driver_used_mb": round(driver_load, 2) if driver_load is not None else None,
                "note": PREV_BENCH_HINT.get(cname, ""),
            }
        )
        print(f"  load {load_s:.1f}s peak_alloc_mb={peak_load} driver_mb={driver_load}")

        # warmup 1 edit
        w_img, w_src, w_edit = jobs[0]
        _ = edit_image(
            str(w_img),
            w_src,
            w_edit,
            inv,
            aux,
            ip,
            scale_edit=0.2,
            scale_non_edit=1.0,
            mask_threshold=0.5,
        )
        _sync(device)
        peak_warm = _peak_alloc_mb(device)
        driver_warm = _driver_used_mb(device)
        memory_rows.append(
            {
                "config": cname,
                "phase": "after_warmup_edit",
                "device": device,
                "load_seconds": None,
                "peak_alloc_mb": round(peak_warm, 2) if peak_warm is not None else None,
                "driver_used_mb": round(driver_warm, 2) if driver_warm is not None else None,
                "note": "steady-ish after 1 edit",
            }
        )

        for ji, (ipath, src_p, edit_p) in enumerate(jobs):
            _reset_peak(device)
            t0 = time.perf_counter()
            res = edit_image(
                str(ipath),
                src_p,
                edit_p,
                inv,
                aux,
                ip,
                scale_edit=0.2,
                scale_non_edit=1.0,
                mask_threshold=0.5,
            )
            _sync(device)
            dt = time.perf_counter() - t0
            peak_ed = _peak_alloc_mb(device)
            pil = _tensor_to_pil(res)
            out_png = img_dir / cname / f"job{ji}_{ipath.stem}.png"
            out_png.parent.mkdir(parents=True, exist_ok=True)
            pil.save(out_png)

            key = (ji, 0)
            if cname == "baseline_fp32":
                ref_images[key] = pil.copy()
                psnr, mse = 99.0, 0.0
            else:
                ref = ref_images.get(key)
                if ref is None:
                    psnr, mse = float("nan"), float("nan")
                else:
                    psnr, mse = _psnr_mse(pil, ref)

            quality_rows.append(
                {
                    "config": cname,
                    "image_idx": ji,
                    "edit_idx": 0,
                    "image": str(ipath),
                    "src_prompt": src_p,
                    "edit_prompt": edit_p,
                    "seconds": round(dt, 3),
                    "psnr_vs_fp32": round(psnr, 4) if psnr == psnr else None,
                    "mse_vs_fp32": mse,
                    "peak_alloc_mb": round(peak_ed, 2) if peak_ed is not None else None,
                    "out_png": str(out_png.relative_to(out_dir)),
                }
            )
            print(
                f"  [{cname}] job{ji} {ipath.name}: {dt:.2f}s "
                f"PSNR_vs_fp32={psnr:.2f} | '{src_p}' → '{edit_p}'"
                if psnr == psnr
                else f"  [{cname}] job{ji} {ipath.name}: {dt:.2f}s"
            )

        del inv, aux, ip
        _free(device)

    # Ensure baseline first for refs — if user ran only fp16, warn
    if "baseline_fp32" not in config_names and quality_rows:
        print(
            "Cảnh báo: không có baseline_fp32 trong --configs → cột psnr_vs_fp32 có thể NaN.",
            file=sys.stderr,
        )

    _write_csv(out_dir / "memory.csv", memory_rows)
    _write_csv(out_dir / "quality.csv", quality_rows)

    # summary per config
    summary_rows = []
    for cname in config_names:
        qs = [r for r in quality_rows if r["config"] == cname]
        ms = [r for r in memory_rows if r["config"] == cname and r["phase"] == "after_load"]
        if not qs and not ms:
            continue
        psnrs = [r["psnr_vs_fp32"] for r in qs if r["psnr_vs_fp32"] is not None]
        secs = [r["seconds"] for r in qs]
        summary_rows.append(
            {
                "config": cname,
                "n_edits": len(qs),
                "seconds_mean": round(statistics.mean(secs), 3) if secs else None,
                "psnr_vs_fp32_mean": round(statistics.mean(psnrs), 3) if psnrs else None,
                "psnr_vs_fp32_min": round(min(psnrs), 3) if psnrs else None,
                "peak_alloc_mb_after_load": ms[0]["peak_alloc_mb"] if ms else None,
                "driver_used_mb_after_load": ms[0]["driver_used_mb"] if ms else None,
                "map_to_prev_bench": PREV_BENCH_HINT.get(cname, ""),
            }
        )
    _write_csv(out_dir / "quality_summary.csv", summary_rows)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "A_mac" if not device.startswith("cuda") else "B_or_cuda",
        "device": device,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git_commit": _git_commit(),
        "configs": config_names,
        "n_images": len(jobs),
        "edits_per_image": 1,
        "n_jobs": len(jobs),
        "jobs": [{"image": str(p), "src": s, "edit": e} for p, s, e in jobs],
        "weights_fp32": str(args.weights_fp32),
        "weights_fp16": str(args.weights_fp16),
        "disk_fp32_bytes": fp32_b,
        "disk_fp16_bytes": fp16_b,
        "disk_save_pct": disk_save_pct,
        "out_dir": str(out_dir),
        "compare_with": "experimental_data/quality_speed_bench_2026-06-17/",
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # report.md
    lines = [
        "# Precision disk / memory / quality",
        "",
        f"- Timestamp (UTC): `{meta['timestamp_utc']}`",
        f"- Device: `{device}` | torch `{meta['torch']}` | git `{meta['git_commit']}`",
        f"- Configs: {', '.join(config_names)}",
        f"- Jobs: {len(jobs)} (prompt theo mapping PIE-Bench)",
        "",
        "## Disk (UNet + IP tree)",
        "",
        "| Label | GiB | Exists |",
        "|---|---:|:---:|",
    ]
    for r in disk_rows:
        lines.append(f"| {r['label']} | {r['gib']} | {r['exists']} |")
    if disk_save_pct is not None:
        lines.append(f"| **tiết kiệm fp16 vs fp32** | **{disk_save_pct:.1f}%** | |")
    lines += [
        "",
        "## Memory after load",
        "",
        "| Config | peak_alloc_mb | driver_used_mb | load_s |",
        "|---|---:|---:|---:|",
    ]
    for r in memory_rows:
        if r["phase"] != "after_load":
            continue
        lines.append(
            f"| {r['config']} | {_fmt_mb(r['peak_alloc_mb'])} | "
            f"{_fmt_mb(r['driver_used_mb'])} | {r.get('load_seconds') or '—'} |"
        )
    lines += [
        "",
        "## Quality vs baseline_fp32 (cùng ảnh + prompt)",
        "",
        "| Config | n | seconds_mean | PSNR↑ mean | PSNR min | map → bench cũ |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['config']} | {r['n_edits']} | {r['seconds_mean']} | "
            f"{r['psnr_vs_fp32_mean']} | {r['psnr_vs_fp32_min']} | {r['map_to_prev_bench']} |"
        )
    lines += [
        "",
        "## File xuất để tải về / so báo cáo",
        "",
        "- `disk.csv`, `memory.csv`, `quality.csv`, `quality_summary.csv`",
        "- `run_meta.json`, `report.md`, `COMPARE_WITH_PREV.md`",
        "- `edited_images/<config>/`",
        "- `bundle.zip` — nén toàn bộ thư mục này",
        "",
        "So với chạy cũ: mở `experimental_data/quality_speed_bench_2026-06-17/report.md`.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    compare = [
        "# So sánh với quality_speed_bench_2026-06-17",
        "",
        "Bench cũ đo **runtime cast/quant** (disk vẫn fp32). Run này đo thêm **disk fp16** + load `torch_dtype`.",
        "",
        "| Run này | Bench 2026-06-17 | Ghi chú |",
        "|---|---|---|",
        "| baseline_fp32 | baseline_fp32 | reference ảnh |",
        "| fp16_disk | improved_fp16_cache | cùng dtype chạy; khác: weights trên disk + cache (bench cũ có EditCache) |",
        "| fp4_from_fp16 | improved_fp4_cache | chỉ CUDA; skip trên Mac |",
        "",
        "## Số liệu tham chiếu bench cũ (Colab T4, 600 edits)",
        "",
        "| Config | Speedup | VRAM MB | PSNR vs fp32 |",
        "|---|---:|---:|---:|",
        "| improved_fp16_cache | 1.70× | 8446 | 48.56 |",
        "| improved_fp4_cache | 1.68× | 7515 | 21.67 |",
        "",
        "## Số liệu run này (xem quality_summary.csv)",
        "",
    ]
    for r in summary_rows:
        compare.append(
            f"- **{r['config']}**: PSNR_vs_fp32_mean={r['psnr_vs_fp32_mean']}, "
            f"peak_load_mb={r['peak_alloc_mb_after_load']}, n={r['n_edits']}"
        )
    compare += [
        "",
        f"Disk save fp16 vs fp32: **{disk_save_pct:.1f}%**" if disk_save_pct else "",
        "",
        "Khi viết báo cáo: nêu rõ Phase A = Mac/MPS (memory khác CUDA); Phase B = T4 cho VRAM + fp4.",
        "",
    ]
    (out_dir / "COMPARE_WITH_PREV.md").write_text("\n".join(compare) + "\n", encoding="utf-8")

    zip_path = out_dir / "bundle.zip"
    _zip_bundle(out_dir, zip_path)
    print(f"\nDone. Report: {out_dir / 'report.md'}")
    print(f"Download bundle: {zip_path}")
    return 0


if __name__ == "__main__":
    # Cho phép override weights qua env (Colab Drive symlink)
    if os.environ.get("SWIFTEDIT_WEIGHTS_FP16"):
        # argparse default đã set; patch via env trước parse bằng re-exec không cần —
        # truyền khi gọi CLI. Giữ env cho setup script.
        pass
    raise SystemExit(main())
