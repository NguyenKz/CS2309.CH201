#!/usr/bin/env python3
"""Gộp nhiều bundle precision_run_* → báo cáo cuối (local, không chạy model).

Yêu cầu cùng `jobs_hash` (cùng jobs_june17 subset). Baseline chất lượng = ảnh
`edited_images/baseline_fp32/` từ run có fp32 (có thể run riêng).

Ví dụ:
  # Thả các bundle đã giải nén vào experimental_data/
  python scripts/compare_precision_runs.py

  python scripts/compare_precision_runs.py \\
      --runs experimental_data/precision_run_..._baseline_fp32 \\
             experimental_data/precision_run_..._fp16_disk \\
      --out experimental_data/PRECISION_FINAL_REPORT.md
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from precision_catalog import ALL_CANONICAL, CONFIGS  # noqa: E402


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _psnr(a: Path, b: Path) -> float | None:
    if not a.is_file() or not b.is_file():
        return None
    xa = np.asarray(Image.open(a).convert("RGB").resize((512, 512)), dtype=np.float64) / 255.0
    xb = np.asarray(Image.open(b).convert("RGB").resize((512, 512)), dtype=np.float64) / 255.0
    mse = float(np.mean((xa - xb) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _discover_runs(exp: Path, explicit: list[Path] | None) -> list[Path]:
    if explicit:
        return [p.resolve() for p in explicit if p.is_dir()]
    runs = sorted(exp.glob("precision_run_*"))
    # giữ tương thích thư mục cũ
    runs += sorted(exp.glob("precision_disk_vram_*"))
    return [p for p in runs if p.is_dir() and (p / "run_meta.json").is_file()]


def _load_run(path: Path) -> dict:
    meta = _read_json(path / "run_meta.json")
    inv = _read_json(path / "inventory.json")
    summary = _read_csv(path / "quality_summary.csv")
    quality = _read_csv(path / "quality.csv")
    disk = _read_csv(path / "disk.csv")
    memory = _read_csv(path / "memory.csv")
    return {
        "path": path,
        "meta": meta,
        "inventory": inv,
        "summary": summary,
        "quality": quality,
        "disk": disk,
        "memory": memory,
        "jobs_hash": meta.get("jobs_hash") or inv.get("jobs_hash"),
    }


def _fp32_image_index(runs: list[dict]) -> dict[str, Path]:
    """job_id → png path baseline_fp32 (ưu tiên run mới nhất có ảnh)."""
    index: dict[str, Path] = {}
    for run in runs:
        d = run["path"] / "edited_images" / "baseline_fp32"
        if not d.is_dir():
            continue
        for png in d.glob("*.png"):
            index[png.stem] = png
    return index


def _recompute_psnr_vs_fp32(run: dict, fp32_imgs: dict[str, Path]) -> dict[str, float]:
    """config → mean PSNR vs fp32 images (cross-run)."""
    out: dict[str, list[float]] = {}
    qdir = run["path"] / "edited_images"
    if not qdir.is_dir() or not fp32_imgs:
        return {}
    for cfg_dir in qdir.iterdir():
        if not cfg_dir.is_dir() or cfg_dir.name == "baseline_fp32":
            continue
        vals = []
        for png in cfg_dir.glob("*.png"):
            ref = fp32_imgs.get(png.stem)
            if ref is None:
                continue
            v = _psnr(png, ref)
            if v is not None:
                vals.append(v)
        if vals:
            out[cfg_dir.name] = statistics.mean(vals)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="*",
        type=Path,
        default=None,
        help="Thư mục run cụ thể; mặc định quét experimental_data/precision_run_*",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experimental_data" / "PRECISION_FINAL_REPORT.md",
    )
    parser.add_argument(
        "--require-same-hash",
        action="store_true",
        default=True,
        help="Chỉ gộp run cùng jobs_hash (mặc định bật)",
    )
    parser.add_argument(
        "--allow-mixed-hash",
        action="store_true",
        help="Cho phép gộp khác jobs_hash (không khuyến nghị)",
    )
    args = parser.parse_args()

    exp = ROOT / "experimental_data"
    run_paths = _discover_runs(exp, args.runs)
    if not run_paths:
        print("Không thấy precision_run_*. Giải nén bundle.zip vào experimental_data/ trước.")
        return 1

    runs = [_load_run(p) for p in run_paths]
    hashes = {r["jobs_hash"] for r in runs if r["jobs_hash"]}
    primary_hash = None
    if hashes:
        # chọn hash có nhiều run nhất
        from collections import Counter

        primary_hash = Counter(r["jobs_hash"] for r in runs if r["jobs_hash"]).most_common(1)[0][0]
    if not args.allow_mixed_hash and primary_hash:
        runs = [r for r in runs if r["jobs_hash"] == primary_hash]
        skipped = [p.name for p in run_paths if _load_run(p)["jobs_hash"] != primary_hash]
        if skipped:
            print("Bỏ run khác jobs_hash:", ", ".join(skipped))

    fp32_imgs = _fp32_image_index(runs)

    # Merged: config → latest summary (stamp trong tên thư mục tăng dần)
    by_config: dict[str, dict] = {}
    for run in runs:
        psnr_map = _recompute_psnr_vs_fp32(run, fp32_imgs)
        for row in run["summary"]:
            cfg = row.get("config")
            if not cfg:
                continue
            psnr_cross = psnr_map.get(cfg)
            in_run = row.get("psnr_vs_fp32_mean") not in (None, "")
            entry = {
                "config": cfg,
                "source": run["path"].name,
                "jobs_hash": run["jobs_hash"],
                "device": run["meta"].get("device"),
                "n": row.get("n_edits"),
                "seconds_mean": row.get("seconds_mean"),
                "seconds_cache_hit_mean": row.get("seconds_cache_hit_mean"),
                "seconds_cache_miss_mean": row.get("seconds_cache_miss_mean"),
                "vram_mb": row.get("peak_alloc_mb_after_load"),
                "psnr_vs_fp32": row.get("psnr_vs_fp32_mean") if in_run else psnr_cross,
                "psnr_source": (
                    "in-run"
                    if in_run
                    else ("cross-run images" if psnr_cross is not None else "missing")
                ),
                "label": CONFIGS.get(cfg, {}).get("label", cfg),
            }
            by_config[cfg] = entry

    # disk note
    disk_note = {}
    for run in runs:
        for d in run["disk"]:
            disk_note[d.get("label")] = f"{d.get('gib')} GiB"

    lines = [
        "# Báo cáo precision cuối (hợp nhất)",
        "",
        f"- Sinh bởi `scripts/compare_precision_runs.py`",
        f"- `jobs_hash` dùng: `{primary_hash}`",
        f"- Số run gộp: **{len(runs)}**",
        f"- Ảnh FP32 làm baseline PSNR: **{len(fp32_imgs)}** job"
        + (" (thiếu — chạy `--configs fp32` trước)" if not fp32_imgs else ""),
        "",
        "## Checklist coverage",
        "",
        "| Config | Có data? | Nguồn run |",
        "|---|:---:|---|",
    ]
    for cfg in ALL_CANONICAL:
        if cfg in by_config:
            lines.append(f"| `{cfg}` | YES | {by_config[cfg]['source']} |")
        else:
            lines.append(f"| `{cfg}` | NO | — |")

    lines += [
        "",
        "## Hiệu năng / resource / chất lượng (vs FP32)",
        "",
        "| Config | n | s/edit | cache hit | VRAM peak MB | PSNR↑ vs FP32 | PSNR nguồn | Disk tree |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]

    # baseline speed for speedup
    base_s = None
    if "baseline_fp32" in by_config and by_config["baseline_fp32"].get("seconds_mean"):
        try:
            base_s = float(by_config["baseline_fp32"]["seconds_mean"])
        except (TypeError, ValueError):
            base_s = None

    for cfg in ALL_CANONICAL:
        r = by_config.get(cfg)
        if not r:
            continue
        disk = (
            disk_note.get("swiftedit_weights_fp16", "—")
            if CONFIGS.get(cfg, {}).get("needs_fp16_disk")
            else disk_note.get("swiftedit_weights_fp32", "—")
        )
        lines.append(
            f"| `{cfg}` | {r['n']} | {r['seconds_mean']} | {r['seconds_cache_hit_mean']} | "
            f"{r['vram_mb']} | {r['psnr_vs_fp32']} | {r['psnr_source']} | {disk} |"
        )

    if base_s:
        lines += ["", "## Speedup vs baseline_fp32", ""]
        for cfg in ALL_CANONICAL:
            r = by_config.get(cfg)
            if not r or not r.get("seconds_mean"):
                continue
            try:
                s = float(r["seconds_mean"])
                sp = base_s / s if s else None
            except (TypeError, ValueError):
                sp = None
            if sp:
                lines.append(f"- `{cfg}`: **{sp:.2f}×**")

    missing = [c for c in ALL_CANONICAL if c not in by_config]
    lines += [
        "",
        "## Còn thiếu",
        "",
    ]
    if missing:
        for m in missing:
            lines.append(f"- `{m}` — chạy Colab `--configs {m}` (cùng MAX_JOBS / jobs_june17), tải bundle về")
    else:
        lines.append("- Đủ 6 config trong catalog hiện tại.")

    lines += [
        "",
        "## Cách bổ sung data",
        "",
        "1. Notebook: chọn **một** (hoặc vài) config trong `EVAL_CONFIGS`.",
        "2. Cell ensure weights (tự convert nếu `*_weight`).",
        "3. Eval → tải `bundle.zip` → giải nén vào `experimental_data/`.",
        "4. Chạy lại script này.",
        "",
        "Không dùng số June17 cũ nếu bạn không tin — chỉ dùng các `precision_run_*` mới cùng `jobs_hash`.",
        "",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", args.out)
    print("Configs covered:", ", ".join(by_config) or "(none)")
    if missing:
        print("Missing:", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
