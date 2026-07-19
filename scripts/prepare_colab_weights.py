#!/usr/bin/env python3
"""Chuẩn bị weights trên Colab — ưu tiên Drive, tránh tải Qualcomm 5 part rồi xóa.

Chiến lược (theo --configs):
  1. Chỉ *_weight / fp16_disk / fp4_from_fp16 → cần cây fp16.
     Ưu tiên: Drive fp16 → symlink. Không tải fp32 Qualcomm.
  2. Có fp32 / fp16 / fp8 / fp4 (compute) → cần cây fp32.
     Ưu tiên: Drive fp32 → symlink; không thì tải stream (curl|tar, không giữ .part/.tar).
  3. Cần fp16 mà chưa có + đã có fp32 → convert_weights_fp16 (một lần).

Ví dụ:
  python scripts/prepare_colab_weights.py --configs fp16_weight,fp4_weight \\
      --drive-fp16 /content/drive/MyDrive/CS2309/swiftedit_weights_fp16

  python scripts/prepare_colab_weights.py --configs fp32,fp16 \\
      --drive-fp32 /content/drive/MyDrive/CS2309/swiftedit_weights
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from precision_catalog import CONFIGS, needs_fp16_disk, resolve_config_names  # noqa: E402

DEFAULT_DRIVE_FP16 = Path("/content/drive/MyDrive/CS2309/swiftedit_weights_fp16")
DEFAULT_DRIVE_FP32 = Path("/content/drive/MyDrive/CS2309/swiftedit_weights")


def _tree_ok(path: Path) -> bool:
    return (path / "sbv2_0.5").is_dir() and (
        path / "ip_adapter_ckpt-90k" / "ip_adapter.bin"
    ).is_file() and (path / "inverse_ckpt-120k").exists()


def _symlink_or_ok(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() == target.resolve() and _tree_ok(link):
            print(f"OK đã gắn: {link} → {target}")
            return
        if link.is_symlink():
            link.unlink()
        elif link.is_dir():
            # Không xóa cây nặng bằng tay — báo lỗi
            raise RuntimeError(
                f"{link} đã là thư mục thật (không phải symlink). "
                "Xóa/đổi tên rồi chạy lại, hoặc trỏ --local-fp16/--local-fp32 khác."
            )
        else:
            link.unlink()
    link.symlink_to(target, target_is_directory=True)
    print(f"symlink {link} → {target}")


def _rm_qualcomm_leftovers(work: Path) -> list[str]:
    removed = []
    for p in list(work.glob("swiftedit_weights.tar.gz*")):
        print(f"rm leftover {p.name} ({p.stat().st_size // (1024**2)} MB)")
        p.unlink(missing_ok=True)
        removed.append(p.name)
    return removed


def _link_drive(drive: Path, local: Path, label: str) -> bool:
    if not drive.is_dir():
        print(f"[{label}] Drive chưa có: {drive}")
        return False
    if not _tree_ok(drive):
        print(f"[{label}] Drive layout thiếu: {drive}")
        return False
    _symlink_or_ok(local, drive)
    return _tree_ok(local)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        required=True,
        help="CSV alias/key (giống eval), quyết định cần fp32 và/hoặc fp16 disk",
    )
    parser.add_argument("--drive-fp16", type=Path, default=DEFAULT_DRIVE_FP16)
    parser.add_argument("--drive-fp32", type=Path, default=DEFAULT_DRIVE_FP32)
    parser.add_argument(
        "--local-fp32",
        type=Path,
        default=ROOT / "SwiftEdit" / "swiftedit_weights",
    )
    parser.add_argument(
        "--local-fp16",
        type=Path,
        default=ROOT / "SwiftEdit" / "swiftedit_weights_fp16",
    )
    parser.add_argument(
        "--allow-qualcomm-download",
        action="store_true",
        default=False,
        help="Cho phép tải Qualcomm tại prepare (mặc định TẮT — để setup_colab tải 1 lần)",
    )
    parser.add_argument(
        "--no-qualcomm-download",
        action="store_true",
        help="(tương thích) luôn cấm tải ở prepare",
    )
    parser.add_argument(
        "--allow-convert",
        action="store_true",
        default=True,
        help="Nếu cần fp16 mà chưa có nhưng có fp32 → convert (chậm, tốn RAM)",
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Không convert fp32→fp16 trên Colab (khuyên: convert Mac + upload Drive)",
    )
    args = parser.parse_args()

    names = resolve_config_names(args.configs)
    need_fp16 = needs_fp16_disk(names)
    need_fp32 = any(CONFIGS[n]["weights"] == "fp32" for n in names)
    allow_dl = bool(args.allow_qualcomm_download) and not args.no_qualcomm_download
    allow_convert = args.allow_convert and not args.no_convert

    print("configs:", ", ".join(names))
    print(f"need_fp32={need_fp32} need_fp16_disk={need_fp16}")

    work = ROOT / "SwiftEdit"
    _rm_qualcomm_leftovers(work)

    # --- fp16 disk (Drive trước) ---
    if need_fp16:
        if _tree_ok(args.local_fp16):
            print(f"fp16 local OK: {args.local_fp16}")
        elif _link_drive(args.drive_fp16, args.local_fp16, "fp16"):
            print("fp16 từ Drive — không cần tải Qualcomm cho *_weight")
        else:
            print("fp16 chưa sẵn sàng trên local/Drive.")

    # --- fp32 (Drive trước, Qualcomm chỉ khi thật sự cần) ---
    fp32_ready = _tree_ok(args.local_fp32)
    if need_fp32 or (need_fp16 and not _tree_ok(args.local_fp16) and allow_convert):
        if fp32_ready:
            print(f"fp32 local OK: {args.local_fp32}")
        elif _link_drive(args.drive_fp32, args.local_fp32, "fp32"):
            print("fp32 từ Drive — bỏ qua tải Qualcomm")
            fp32_ready = True
        elif need_fp32 or (need_fp16 and not _tree_ok(args.local_fp16)):
            if not allow_dl:
                print(
                    "Thiếu fp32 tại local/Drive.\n"
                    "→ Chạy lại cell setup với SKIP_QUALCOMM_IN_SETUP=False "
                    "(tải 1 mạch trong setup — nhanh hơn tách sang prepare).\n"
                    "Hoặc upload Drive fp32, hoặc thêm --allow-qualcomm-download.",
                    file=sys.stderr,
                )
                return 1
            print(
                "Tải Qualcomm fp32 (stream HTTP→tar, log % / MB/s / ETA từng part)...",
                flush=True,
            )
            r = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts" / "download_swiftedit_weights.py"),
                    "--stream",
                    "--clean",
                ],
                cwd=ROOT,
            )
            if r.returncode != 0:
                return r.returncode
            fp32_ready = _tree_ok(args.local_fp32)
    elif not need_fp32:
        print("Không cần fp32 cho configs này → SKIP tải Qualcomm (~10GB+).")

    # --- convert nếu vẫn thiếu fp16 ---
    if need_fp16 and not _tree_ok(args.local_fp16):
        if not fp32_ready:
            print("Thiếu cả fp16 và fp32 — không convert được.", file=sys.stderr)
            return 1
        if not allow_convert:
            print(
                "Thiếu fp16. Upload Drive fp16 (khuyên) hoặc bỏ --no-convert.",
                file=sys.stderr,
            )
            return 1
        print("Convert fp32 → fp16 trên máy này (lâu / tốn RAM). Khuyên convert Mac + Drive.")
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "convert_weights_fp16.py"),
                "--src",
                str(args.local_fp32),
                "--dst",
                str(args.local_fp16),
            ],
            cwd=ROOT,
        )
        if r.returncode != 0:
            return r.returncode

    # --- kiểm tra cuối ---
    ok = True
    if need_fp32 and not _tree_ok(args.local_fp32):
        print("FAIL: vẫn thiếu fp32", file=sys.stderr)
        ok = False
    if need_fp16 and not _tree_ok(args.local_fp16):
        print("FAIL: vẫn thiếu fp16 disk", file=sys.stderr)
        ok = False
    if ok:
        print("prepare_colab_weights: OK")
        if _tree_ok(args.local_fp16):
            os.environ["SWIFTEDIT_WEIGHTS_FP16"] = str(args.local_fp16)
        if _tree_ok(args.local_fp32):
            os.environ["SWIFTEDIT_WEIGHTS_FP32"] = str(args.local_fp32)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(e, file=sys.stderr)
        raise SystemExit(1)
