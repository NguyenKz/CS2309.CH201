#!/usr/bin/env python3
"""Xuất UNet + IP-Adapter SwiftEdit sang checkpoint fp16 trên disk.

Giảm dung lượng ~50% so với fp32 safetensors/bin gốc. VAE/HF base không đụng.

Ví dụ:
  python scripts/convert_weights_fp16.py
  python scripts/convert_weights_fp16.py \\
      --src SwiftEdit/swiftedit_weights \\
      --dst SwiftEdit/swiftedit_weights_fp16
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel

ROOT = Path(__file__).resolve().parent.parent


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _fmt_gb(n: int) -> str:
    return f"{n / (1024**3):.3f} GB"


def _convert_unet(src_dir: Path, dst_dir: Path, *, subfolder: str | None) -> dict:
    t0 = time.perf_counter()
    kwargs = {}
    if subfolder:
        kwargs["subfolder"] = subfolder
    print(f"[unet] load {src_dir}" + (f"/{subfolder}" if subfolder else ""))
    unet = UNet2DConditionModel.from_pretrained(str(src_dir), **kwargs)
    unet = unet.to(dtype=torch.float16)
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"[unet] save → {dst_dir}")
    unet.save_pretrained(str(dst_dir), safe_serialization=True)
    del unet
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    src_b = _dir_bytes(src_dir / subfolder) if subfolder else _dir_bytes(src_dir)
    dst_b = _dir_bytes(dst_dir)
    return {
        "src": str(src_dir / subfolder if subfolder else src_dir),
        "dst": str(dst_dir),
        "src_bytes": src_b,
        "dst_bytes": dst_b,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def _convert_ip_bin(src_bin: Path, dst_bin: Path) -> dict:
    t0 = time.perf_counter()
    print(f"[ip] load {src_bin}")
    sd = torch.load(src_bin, map_location="cpu", weights_only=True)
    out = {}
    for k, v in sd.items():
        if torch.is_floating_point(v):
            out[k] = v.half().contiguous()
        else:
            out[k] = v
    dst_bin.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ip] save → {dst_bin}")
    torch.save(out, dst_bin)
    src_b = src_bin.stat().st_size
    dst_b = dst_bin.stat().st_size
    return {
        "src": str(src_bin),
        "dst": str(dst_bin),
        "src_bytes": src_b,
        "dst_bytes": dst_b,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert SwiftEdit UNet+IP → fp16 on disk")
    parser.add_argument(
        "--src",
        type=Path,
        default=ROOT / "SwiftEdit" / "swiftedit_weights",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=ROOT / "SwiftEdit" / "swiftedit_weights_fp16",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Xóa dst nếu đã tồn tại rồi convert lại",
    )
    args = parser.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    inv_src = src / "inverse_ckpt-120k"
    sbv2_src = src / "sbv2_0.5"
    ip_src = src / "ip_adapter_ckpt-90k" / "ip_adapter.bin"

    for p, label in [
        (inv_src / "unet_ema", "inverse unet_ema"),
        (sbv2_src, "sbv2_0.5"),
        (ip_src, "ip_adapter.bin"),
    ]:
        if not p.exists():
            print(f"Thiếu {label}: {p}", file=sys.stderr)
            return 1

    if dst.exists():
        if args.force:
            print(f"Xóa dst cũ: {dst}")
            shutil.rmtree(dst)
        else:
            marker = dst / "CONVERT_META.json"
            if marker.is_file() and (dst / "sbv2_0.5").is_dir():
                print(f"Đã có {dst} — bỏ qua (dùng --force để convert lại).")
                return 0

    rows = []
    # inverse: lưu cùng layout inverse_ckpt-120k/unet_ema/
    rows.append(
        _convert_unet(
            inv_src,
            dst / "inverse_ckpt-120k" / "unet_ema",
            subfolder="unet_ema",
        )
    )
    # copy config cấp inverse nếu có
    inv_cfg = inv_src / "config.json"
    if inv_cfg.is_file():
        (dst / "inverse_ckpt-120k").mkdir(parents=True, exist_ok=True)
        shutil.copy2(inv_cfg, dst / "inverse_ckpt-120k" / "config.json")

    rows.append(_convert_unet(sbv2_src, dst / "sbv2_0.5", subfolder=None))
    rows.append(
        _convert_ip_bin(ip_src, dst / "ip_adapter_ckpt-90k" / "ip_adapter.bin")
    )

    src_total = sum(r["src_bytes"] for r in rows)
    dst_total = sum(r["dst_bytes"] for r in rows)
    meta = {
        "dtype": "float16",
        "components": rows,
        "src_total_bytes": src_total,
        "dst_total_bytes": dst_total,
        "ratio": round(dst_total / src_total, 4) if src_total else None,
        "src_root": str(src),
        "dst_root": str(dst),
    }
    (dst / "CONVERT_META.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\n=== Disk (UNet + IP) ===")
    print(f"{'component':<40} {'src':>10} {'dst':>10} {'ratio':>8}")
    for r in rows:
        name = Path(r["src"]).name
        ratio = r["dst_bytes"] / r["src_bytes"] if r["src_bytes"] else 0
        print(
            f"{name:<40} {_fmt_gb(r['src_bytes']):>10} {_fmt_gb(r['dst_bytes']):>10} "
            f"{ratio:>7.1%}"
        )
    print(
        f"{'TOTAL':<40} {_fmt_gb(src_total):>10} {_fmt_gb(dst_total):>10} "
        f"{(dst_total / src_total) if src_total else 0:>7.1%}"
    )
    print(f"\nMeta: {dst / 'CONVERT_META.json'}")
    print("Upload Colab Drive: nén thư mục dst rồi upload, hoặc sync folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
