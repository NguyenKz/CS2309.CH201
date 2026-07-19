#!/usr/bin/env python3
"""Kiểm tra / tạo weights fp16 trên disk khi config cần (fp16_disk, fp4_from_fp16).

Không đụng configs chỉ dùng disk fp32 (baseline / improved_*_cache).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from precision_catalog import needs_fp16_disk, resolve_config_names  # noqa: E402


def _fp16_ready(weights_fp16: Path) -> bool:
    need = [
        weights_fp16 / "sbv2_0.5",
        weights_fp16 / "inverse_ckpt-120k" / "unet_ema",
        weights_fp16 / "ip_adapter_ckpt-90k" / "ip_adapter.bin",
    ]
    return all(p.exists() for p in need)


def ensure_fp16_weights(
    *,
    configs: list[str],
    weights_fp32: Path,
    weights_fp16: Path,
) -> int:
    if not needs_fp16_disk(configs):
        print("Không cần fp16 disk cho configs:", ", ".join(configs))
        return 0
    if _fp16_ready(weights_fp16):
        print(f"fp16 disk OK: {weights_fp16}")
        return 0
    if not (weights_fp32 / "sbv2_0.5").is_dir():
        print(
            f"Thiếu fp32 tại {weights_fp32} — không convert được.",
            file=sys.stderr,
        )
        return 1
    print(f"Chưa có fp16 disk → convert {weights_fp32} → {weights_fp16}")
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "convert_weights_fp16.py"),
            "--src",
            str(weights_fp32),
            "--dst",
            str(weights_fp16),
        ],
        cwd=ROOT,
    )
    if r.returncode != 0:
        return r.returncode
    if not _fp16_ready(weights_fp16):
        print("Convert xong nhưng vẫn thiếu file bắt buộc.", file=sys.stderr)
        return 1
    print("fp16 disk sẵn sàng.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        required=True,
        help="CSV configs / alias (vd: fp16_weight,fp4_weight)",
    )
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
    args = parser.parse_args()
    names = resolve_config_names(args.configs)
    return ensure_fp16_weights(
        configs=names,
        weights_fp32=args.weights_fp32,
        weights_fp16=args.weights_fp16,
    )


if __name__ == "__main__":
    raise SystemExit(main())
