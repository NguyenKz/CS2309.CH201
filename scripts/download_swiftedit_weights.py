#!/usr/bin/env python3
"""Tải swiftedit_weights từ GitHub Releases (Colab-friendly, có retry)."""
from __future__ import annotations

import subprocess
import sys
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
    "-C",
    "-",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    work = root / "SwiftEdit"
    work.mkdir(parents=True, exist_ok=True)

    if (work / "swiftedit_weights" / "inverse_ckpt-120k").is_dir():
        print("swiftedit_weights: đã có, bỏ qua.")
        return

    for part in PARTS:
        fname = f"swiftedit_weights.tar.gz.part-{part}"
        dest = work / fname
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"skip {fname} ({dest.stat().st_size // 1_000_000} MB)")
            continue
        url = f"{BASE}/{fname}"
        print(f"download {fname} ...")
        subprocess.run([*CURL, "-o", str(dest), url], check=True, cwd=work)

    parts = [work / f"swiftedit_weights.tar.gz.part-{p}" for p in PARTS]
    out_tar = work / "swiftedit_weights.tar.gz"
    print("ghép archive...")
    with open(out_tar, "wb") as out:
        for p in parts:
            out.write(p.read_bytes())

    print("giải nén...")
    subprocess.run(["tar", "zxf", str(out_tar)], check=True, cwd=work)
    size = (work / "swiftedit_weights").exists()
    print("done:", work / "swiftedit_weights", "OK" if size else "FAIL")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print("Lỗi tải weights:", e, file=sys.stderr)
        print("Thử: chạy lại cell / kiểm tra dung lượng /content còn ~12GB trống.", file=sys.stderr)
        sys.exit(e.returncode or 1)
