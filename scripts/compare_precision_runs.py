#!/usr/bin/env python3
"""Gộp số liệu các run precision → bảng markdown cho báo cáo (chạy local).

Không chạy lại model. Đọc:
  - experimental_data/quality_speed_bench_2026-06-17/report.md (số tay / TL;DR)
  - experimental_data/precision_disk_vram_*/quality_summary.csv, disk.csv, memory.csv

Ví dụ:
  python scripts/compare_precision_runs.py
  python scripts/compare_precision_runs.py --out experimental_data/PRECISION_COMPARISON.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Số liệu đã công bố trong report 2026-06-17 (không parse lại 600 dòng CSV nặng).
JUNE17 = [
    {
        "case": "1 Full FP32",
        "config": "baseline_fp32",
        "source": "quality_speed_bench_2026-06-17",
        "device": "T4",
        "n": 600,
        "seconds_mean": 2.91,
        "vram_mb": 14596,
        "psnr_vs_fp32": None,
        "disk_note": "fp32 ~10GB",
        "status": "DONE",
    },
    {
        "case": "2 FP16 compute (+cache)",
        "config": "improved_fp16_cache",
        "source": "quality_speed_bench_2026-06-17",
        "device": "T4",
        "n": 600,
        "seconds_mean": 1.71,
        "vram_mb": 8446,
        "psnr_vs_fp32": 48.56,
        "disk_note": "disk vẫn fp32",
        "status": "DONE",
    },
    {
        "case": "3 FP8 runtime",
        "config": "improved_fp8_cache",
        "source": "quality_speed_bench_2026-06-17",
        "device": "T4",
        "n": 600,
        "seconds_mean": 1.52,
        "vram_mb": 7819,
        "psnr_vs_fp32": 6.01,
        "disk_note": "disk vẫn fp32",
        "status": "DONE (fail chất lượng)",
    },
    {
        "case": "4 FP4 runtime",
        "config": "improved_fp4_cache",
        "source": "quality_speed_bench_2026-06-17",
        "device": "T4",
        "n": 600,
        "seconds_mean": 1.74,
        "vram_mb": 7515,
        "psnr_vs_fp32": 21.67,
        "disk_note": "disk vẫn fp32",
        "status": "DONE",
    },
]


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_precision_runs(exp: Path) -> list[dict]:
    rows = []
    for d in sorted(exp.glob("precision_disk_vram_*")):
        if not d.is_dir() or d.name.endswith("_promptfix") is False and "PROMPT" in d.name:
            pass
        summary = _read_csv(d / "quality_summary.csv")
        disk = _read_csv(d / "disk.csv")
        disk_note = ""
        for r in disk:
            if "fp16" in r.get("label", ""):
                disk_note = f"fp16 tree {r.get('gib')} GiB"
        for r in summary:
            rows.append(
                {
                    "case": f"disk-run {d.name} / {r.get('config')}",
                    "config": r.get("config"),
                    "source": d.name,
                    "device": "see run_meta",
                    "n": r.get("n_edits"),
                    "seconds_mean": r.get("seconds_mean"),
                    "vram_mb": r.get("peak_alloc_mb_after_load"),
                    "psnr_vs_fp32": r.get("psnr_vs_ref_mean") or r.get("psnr_vs_fp32_mean"),
                    "disk_note": disk_note,
                    "status": "DONE" if summary else "EMPTY",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experimental_data" / "PRECISION_COMPARISON.md",
    )
    args = parser.parse_args()

    exp = ROOT / "experimental_data"
    all_rows = list(JUNE17) + _load_precision_runs(exp)

    lines = [
        "# Bảng so sánh precision (tổng hợp)",
        "",
        "Sinh bởi `scripts/compare_precision_runs.py`. Case 1–4 từ bench T4 2026-06-17;",
        "các dòng `precision_disk_vram_*` từ Phase A/B (disk fp16 / fp4_from_fp16).",
        "",
        "Chi tiết kiểm kê: [`PRECISION_CASES.md`](./PRECISION_CASES.md).",
        "",
        "| Case | Config | Nguồn | n | s/edit | VRAM/peak MB | PSNR | Disk | Status |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in all_rows:
        lines.append(
            f"| {r['case']} | `{r['config']}` | {r['source']} | {r['n']} | "
            f"{r['seconds_mean']} | {r['vram_mb']} | {r['psnr_vs_fp32']} | "
            f"{r['disk_note']} | {r['status']} |"
        )
    lines += [
        "",
        "## Thiếu gì?",
        "",
        "- Case **FP16 disk trên T4** với quy mô gần bench cũ (hoặc subset rõ ràng).",
        "- Case **FP4 from fp16 disk** trên T4.",
        "- Sau khi có `bundle.zip` Colab: thả vào `experimental_data/` rồi chạy lại script này.",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
