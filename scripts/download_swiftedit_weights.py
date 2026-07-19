#!/usr/bin/env python3
"""Tải swiftedit_weights — ưu tiên tốc độ (curl như ban đầu).

Mặc định --parts: curl từng part (có resume + progress bar) → cat|tar → xóa .part.
Không tạo .tar.gz trung gian (đỡ nhân đôi disk). Không dùng urllib (chậm hơn curl).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = "https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0"
PARTS = ("aa", "ab", "ac", "ad", "ae")
CURL = [
    "curl",
    "-fL",
    "--retry",
    "10",
    "--retry-delay",
    "5",
    "--connect-timeout",
    "60",
]


def _tree_ok(weights: Path) -> bool:
    return (weights / "inverse_ckpt-120k").is_dir() and (weights / "sbv2_0.5").is_dir()


def _clean_archives(work: Path) -> None:
    for p in list(work.glob("swiftedit_weights.tar.gz*")):
        mb = p.stat().st_size // (1024**2)
        print(f"clean: rm {p.name} ({mb} MB)", flush=True)
        p.unlink(missing_ok=True)


def _free_gb(path: Path) -> float | None:
    try:
        return shutil.disk_usage(path).free / (1024**3)
    except Exception:
        return None


def download_with_curl(work: Path) -> None:
    """curl 5 part (nhanh) → cat|tar → xóa part. Giống flow cũ, bỏ bước ghi .tar.gz."""
    free = _free_gb(work)
    print("Tải Qualcomm bằng curl (1 lần, 5 part) → extract → xóa .part", flush=True)
    if free is not None:
        print(f"disk trống: ~{free:.1f} GB (cần ≥~20GB tạm cho 5 part + extract)", flush=True)

    part_files: list[Path] = []
    for i, part in enumerate(PARTS, start=1):
        fname = f"swiftedit_weights.tar.gz.part-{part}"
        dest = work / fname
        url = f"{BASE}/{fname}"
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(
                f"[{i}/5] part-{part}  resume/skip "
                f"({dest.stat().st_size // (1024**2)} MB đã có)",
                flush=True,
            )
        else:
            print(f"\n[{i}/5] curl part-{part} ...", flush=True)
            subprocess.run(
                [*CURL, "-C", "-", "--progress-bar", "-o", str(dest), url],
                check=True,
                cwd=work,
            )
            print(flush=True)
        part_files.append(dest)

    print("\ncat | tar extract (không tạo .tar.gz trung gian)...", flush=True)
    cat = subprocess.Popen(
        ["cat", *[str(p) for p in part_files]],
        stdout=subprocess.PIPE,
        cwd=work,
    )
    tar = subprocess.run(["tar", "zxf", "-"], stdin=cat.stdout, cwd=work)
    if cat.stdout:
        cat.stdout.close()
    cat.wait()
    if tar.returncode != 0:
        raise RuntimeError(f"tar extract failed (exit {tar.returncode})")

    for p in part_files:
        print(f"clean {p.name}", flush=True)
        p.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="(alias) cùng --parts: curl nhanh, không urllib",
    )
    parser.add_argument(
        "--parts",
        action="store_true",
        default=True,
        help="curl từng part rồi cat|tar (mặc định, nhanh)",
    )
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    work = root / "SwiftEdit"
    work.mkdir(parents=True, exist_ok=True)
    weights = work / "swiftedit_weights"

    if _tree_ok(weights):
        print("swiftedit_weights: đã có, bỏ qua tải.", flush=True)
        if args.clean and not args.no_clean:
            _clean_archives(work)
        return 0

    if args.clean and not args.no_clean:
        _clean_archives(work)

    t0 = time.perf_counter()
    try:
        download_with_curl(work)
    except Exception as e:
        print(f"\nLỗi tải weights: {e}", file=sys.stderr, flush=True)
        return 1

    if args.clean and not args.no_clean:
        _clean_archives(work)

    if not _tree_ok(weights):
        print("FAIL: sau extract vẫn thiếu layout.", file=sys.stderr)
        return 1

    du = subprocess.run(["du", "-sh", str(weights)], capture_output=True, text=True)
    print(
        f"done: {du.stdout.strip() or weights}  ({(time.perf_counter() - t0) / 60:.1f} phút)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
