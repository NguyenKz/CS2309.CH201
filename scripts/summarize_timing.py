#!/usr/bin/env python3
"""Tổng hợp results/timing*.log (JSONL) → báo cáo markdown + CSV.

Ví dụ:
  python scripts/summarize_timing.py
  python scripts/summarize_timing.py --log results/timing_bench20.log --out results/timing_report_20.md
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Stack model của SwiftEdit (khớp SwiftEdit/models.py + infer.py)
MODEL_STACK = [
    ("Inverse UNet", "swiftedit_weights/inverse_ckpt-120k (subfolder unet_ema)", "base: stabilityai/sd-turbo, fp32"),
    ("Generation UNet (SwiftBrush v2)", "swiftedit_weights/sbv2_0.5", "one-step generator"),
    ("IP-Adapter", "swiftedit_weights/ip_adapter_ckpt-90k/ip_adapter.bin", "image prompt adapter"),
    ("VAE / Text encoder / Tokenizer", "Manojb/stable-diffusion-2-1-base", "SD 2.1 base (mirror)"),
    ("CLIP image encoder", "h94/IP-Adapter (models/image_encoder)", "CLIP ViT-H"),
]


def collect_env(device: str) -> dict:
    """Thu thập thông tin thiết bị + thư viện để ghi vào báo cáo (tái lập)."""
    info = {
        "device": device,
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": "",
        "ram_gb": "",
        "torch": "",
        "transformers": "",
        "diffusers": "",
    }
    if platform.system() == "Darwin":
        try:
            info["cpu"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            info["ram_gb"] = str(mem // (1024**3))
            info["os"] = "macOS " + subprocess.check_output(
                ["sw_vers", "-productVersion"], text=True
            ).strip()
        except (subprocess.SubprocessError, ValueError, OSError):
            pass
    for mod in ("torch", "transformers", "diffusers"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            info[mod] = "n/a"
    return info

STAGE_ORDER = [
    "vae_encode",
    "inv_text_encode",
    "unet_inverse",
    "mask_estimate",
    "gen_image_embeds",
    "gen_text_encode",
    "gen_unet",
    "gen_vae_decode",
    "gen_vae_decode_noise",
]

STAGE_LABELS = {
    "vae_encode": "VAE encode (ảnh → latent)",
    "inv_text_encode": "Text encoder (inverse, src+edit)",
    "unet_inverse": "UNet inverse (ước lượng noise)",
    "mask_estimate": "Ước lượng mask chỉnh sửa",
    "gen_image_embeds": "IP-Adapter image embeds",
    "gen_text_encode": "Text encoder (generation)",
    "gen_unet": "UNet 1-step (sinh ảnh)",
    "gen_vae_decode": "VAE decode (latent → ảnh)",
    "gen_vae_decode_noise": "VAE decode noise (tuỳ chọn)",
}


def load_records(log_path: Path) -> list[dict]:
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "p50": 0, "p95": 0}
    values = sorted(values)
    n = len(values)
    p50 = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
    p95_idx = min(n - 1, int(0.95 * n))
    return {
        "n": n,
        "mean": round(statistics.mean(values), 2),
        "std": round(statistics.stdev(values), 2) if n > 1 else 0.0,
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "p50": round(p50, 2),
        "p95": round(values[p95_idx], 2),
    }


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"n_samples": 0, "device": "", "stages": {}, "total": _stats([])}

    device = records[0].get("device", "")
    totals = [r["total_ms"] for r in records if "total_ms" in r]

    stage_keys = set()
    for r in records:
        stage_keys.update(r.get("stages_ms", {}).keys())
    ordered = [s for s in STAGE_ORDER if s in stage_keys]
    ordered += sorted(stage_keys - set(ordered))

    stages = {}
    for stage in ordered:
        vals = [r["stages_ms"][stage] for r in records if stage in r.get("stages_ms", {})]
        stages[stage] = _stats(vals)

    return {
        "n_samples": len(records),
        "device": device,
        "stages": stages,
        "total": _stats(totals),
    }


QUALITY_FIELDS = [
    ("clip_whole", "CLIP-Whole (toàn ảnh ↔ edit prompt)", "cao ↑"),
    ("clip_edited", "CLIP-Edited (vùng sửa ↔ edit prompt)", "cao ↑"),
    ("psnr_unedit", "PSNR vùng nền (giữ nền)", "cao ↑"),
    ("mse_unedit", "MSE vùng nền (giữ nền)", "thấp ↓"),
]


def load_quality(metrics_csv: Path, n_recent: int) -> dict | None:
    """Đọc N dòng cuối của metrics.csv (cùng đợt chạy) và tổng hợp."""
    if not metrics_csv.is_file():
        return None
    with metrics_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    rows = rows[-n_recent:]
    out = {"n_rows": len(rows), "fields": {}}
    for key, _, _ in QUALITY_FIELDS:
        vals = []
        for r in rows:
            v = r.get(key, "")
            if v not in ("", None):
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        out["fields"][key] = _stats(vals)
    return out


def write_csv(summary: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "label", "mean_ms", "std_ms", "min_ms", "max_ms", "p50_ms", "p95_ms", "n"])
        total = summary["total"]
        w.writerow(["total", "Tổng edit_image", total["mean"], total["std"], total["min"], total["max"], total["p50"], total["p95"], total["n"]])
        for stage, st in summary["stages"].items():
            label = STAGE_LABELS.get(stage, stage)
            w.writerow([stage, label, st["mean"], st["std"], st["min"], st["max"], st["p50"], st["p95"], st["n"]])


def write_markdown(
    summary: dict,
    records: list[dict],
    md_path: Path,
    log_path: Path,
    quality: dict | None = None,
) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n = summary["n_samples"]
    device = summary["device"]
    total = summary["total"]
    env = collect_env(device)
    n_warm = 2 if n > 4 else 0
    steady = _stats([r["total_ms"] for r in records[n_warm:] if "total_ms" in r])

    lines = [
        "# Báo cáo thời gian inference SwiftEdit",
        "",
        f"- **Ngày đo:** {now}",
        f"- **Số mẫu:** {n}",
        f"- **Nguồn log:** `{log_path}`",
        f"- **Pipeline:** một lần gọi `edit_image()` (512×512, 1-step SwiftEdit)",
        "",
        "## Môi trường đo",
        "",
        "| Mục | Giá trị |",
        "|-----|---------|",
        f"| Thiết bị tính toán | `{env['device']}` (Metal Performance Shaders) |",
        f"| Máy | {env['cpu'] or env['machine']} |",
        f"| RAM | {env['ram_gb']} GB |" if env["ram_gb"] else "| RAM | n/a |",
        f"| Hệ điều hành | {env['os']} |",
        f"| Python | {env['python']} |",
        f"| PyTorch | {env['torch']} |",
        f"| transformers | {env['transformers']} |",
        f"| diffusers | {env['diffusers']} |",
        "",
        "### Model sử dụng (SwiftEdit)",
        "",
        "| Thành phần | Checkpoint / nguồn | Ghi chú |",
        "|-----------|--------------------|---------|",
    ]
    for name, ckpt, note in MODEL_STACK:
        lines.append(f"| {name} | `{ckpt}` | {note} |")
    lines += [
        "",
        "## Tổng quan",
        "",
        f"| Chỉ số | Giá trị (ms) | Giá trị (s) |",
        f"|--------|-------------:|------------:|",
        f"| Trung bình | {total['mean']} | {round(total['mean']/1000, 2)} |",
        f"| Độ lệch chuẩn | {total['std']} | — |",
        f"| Min / Max | {total['min']} / {total['max']} | {round(total['min']/1000, 2)} / {round(total['max']/1000, 2)} |",
        f"| P50 / P95 | {total['p50']} / {total['p95']} | {round(total['p50']/1000, 2)} / {round(total['p95']/1000, 2)} |",
        "",
        f"**Throughput ước lượng:** ~{round(1000 / total['mean'], 3)} ảnh/giây (chỉ inference, không tính load model).",
        "",
        f"> **Warmup (MPS):** {n_warm} mẫu đầu thường nhanh/chậm bất thường do biên dịch kernel lần đầu. "
        f"Steady-state ({steady['n']} mẫu còn lại): trung bình **{round(steady['mean']/1000, 2)} s**, P50 **{round(steady['p50']/1000, 2)} s**.",
        "",
        "## Thời gian từng công đoạn (ms)",
        "",
        "| Công đoạn | Mô tả | Mean | Std | % tổng | Min | Max | P50 |",
        "|-----------|-------|-----:|----:|-------:|----:|----:|----:|",
    ]

    mean_total = total["mean"] or 1.0
    for stage in [s for s in STAGE_ORDER if s in summary["stages"]] + [
        s for s in summary["stages"] if s not in STAGE_ORDER
    ]:
        st = summary["stages"][stage]
        pct = round(100 * st["mean"] / mean_total, 1)
        label = STAGE_LABELS.get(stage, stage)
        lines.append(
            f"| `{stage}` | {label} | {st['mean']} | {st['std']} | {pct}% | {st['min']} | {st['max']} | {st['p50']} |"
        )

    # Nhóm UNet
    unet_mean = sum(summary["stages"].get(s, {}).get("mean", 0) for s in ("unet_inverse", "gen_unet"))
    unet_pct = round(100 * unet_mean / mean_total, 1)
    lines.extend([
        "",
        "## Phân tích nhanh",
        "",
        f"- **Hai lần UNet forward** (`unet_inverse` + `gen_unet`): ~{unet_mean:.0f} ms/trung bình (~{unet_pct}% tổng thời gian).",
        "- **VAE** (`vae_encode` + `gen_vae_decode`): encode + decode latent.",
        "- **Text/IP embeds**: hai lần text encoder + IP-Adapter image encoder.",
        "- **Mask estimate**: nhẹ so với UNet.",
        "",
    ])

    if quality and quality.get("fields"):
        lines.extend([
            "## Chất lượng chỉnh sửa (PIE-Bench metrics)",
            "",
            f"Tính trên {quality['n_rows']} mẫu cùng đợt (nguồn: `results/piebench/metrics.csv`). "
            "PSNR/MSE chỉ tính trên các mẫu có vùng nền (mask không phủ toàn ảnh).",
            "",
            "| Metric | Ý nghĩa | Mean | Min | Max | Số mẫu |",
            "|--------|---------|-----:|----:|----:|------:|",
        ])
        for key, label, direction in QUALITY_FIELDS:
            st = quality["fields"].get(key)
            if not st or st["n"] == 0:
                lines.append(f"| `{key}` | {label} ({direction}) | n/a | — | — | 0 |")
            else:
                lines.append(
                    f"| `{key}` | {label} ({direction}) | {st['mean']} | {st['min']} | {st['max']} | {st['n']} |"
                )
        lines.append("")

    lines.extend([
        "## Chi tiết từng mẫu",
        "",
        "| # | Label | Total (ms) | unet_inverse | gen_unet |",
        "|--:|-------|----------:|-------------:|---------:|",
    ])

    for i, r in enumerate(records, 1):
        st = r.get("stages_ms", {})
        lines.append(
            f"| {i} | {r.get('label', '')} | {r.get('total_ms', '')} | "
            f"{st.get('unet_inverse', '')} | {st.get('gen_unet', '')} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tổng hợp timing log SwiftEdit")
    parser.add_argument("--log", type=Path, default=ROOT / "results" / "timing.log")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "timing_report.md")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=ROOT / "results" / "piebench" / "metrics.csv",
        help="CSV chất lượng PIE-Bench; lấy N dòng cuối theo số mẫu timing",
    )
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"Không thấy log: {args.log}", file=__import__("sys").stderr)
        return 1

    records = load_records(args.log)
    summary = summarize(records)
    quality = load_quality(args.metrics_csv, summary["n_samples"])
    csv_path = args.csv or args.out.with_suffix(".csv")
    write_csv(summary, csv_path)
    write_markdown(summary, records, args.out, args.log, quality=quality)

    print(f"[timing] {summary['n_samples']} mẫu → {args.out}")
    print(f"[timing] CSV → {csv_path}")
    print(f"[timing] mean total: {summary['total']['mean']} ms ({round(summary['total']['mean']/1000, 2)} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
