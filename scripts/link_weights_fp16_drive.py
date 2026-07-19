#!/usr/bin/env python3
"""Gắn / kiểm tra SwiftEdit weights fp16 từ Google Drive trên Colab.

Mặc định Drive path (đổi nếu bạn upload khác chỗ):
  /content/drive/MyDrive/CS2309/swiftedit_weights_fp16

Tạo symlink:
  <repo>/SwiftEdit/swiftedit_weights_fp16 → thư mục trên Drive

Ví dụ trên Colab:
  from google.colab import drive
  drive.mount('/content/drive')
  !python scripts/link_weights_fp16_drive.py --drive-dir /content/drive/MyDrive/CS2309/swiftedit_weights_fp16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DRIVE = Path("/content/drive/MyDrive/CS2309/swiftedit_weights_fp16")
LOCAL_LINK = ROOT / "SwiftEdit" / "swiftedit_weights_fp16"


def _ok_tree(path: Path) -> bool:
    return (path / "sbv2_0.5").is_dir() and (
        path / "ip_adapter_ckpt-90k" / "ip_adapter.bin"
    ).is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description="Link fp16 weights from Google Drive")
    parser.add_argument("--drive-dir", type=Path, default=DEFAULT_DRIVE)
    parser.add_argument(
        "--link",
        type=Path,
        default=LOCAL_LINK,
        help="Đích symlink trong repo",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy thay vì symlink (chậm hơn, dùng khi symlink lỗi)",
    )
    args = parser.parse_args()

    drive_dir: Path = args.drive_dir
    link: Path = args.link

    if not drive_dir.exists():
        print(
            f"Không thấy Drive dir: {drive_dir}\n"
            "1) Mount Drive\n"
            "2) Upload thư mục swiftedit_weights_fp16 (sau convert trên Mac)\n"
            "   Gợi ý: nén tar.gz trên Mac rồi upload, giải nén trên Drive hoặc /content\n"
            f"3) Chạy lại với --drive-dir <path>",
            file=sys.stderr,
        )
        return 1

    if not _ok_tree(drive_dir):
        print(
            f"Drive dir thiếu layout (cần sbv2_0.5/ và ip_adapter_ckpt-90k/ip_adapter.bin): {drive_dir}",
            file=sys.stderr,
        )
        return 1

    if link.exists() or link.is_symlink():
        if link.is_symlink() or link.is_dir():
            if link.resolve() == drive_dir.resolve() and _ok_tree(link):
                print(f"OK đã gắn: {link} → {drive_dir}")
                os.environ["SWIFTEDIT_WEIGHTS_FP16"] = str(link)
                print(f"export SWIFTEDIT_WEIGHTS_FP16={link}")
                return 0
            print(f"Gỡ link/dir cũ: {link}")
            if link.is_symlink():
                link.unlink()
            else:
                # không xóa cây weights thật trong repo
                print(
                    "Cảnh báo: --link là thư mục thật (không phải symlink). "
                    "Đổi --link hoặc xóa thủ công rồi chạy lại.",
                    file=sys.stderr,
                )
                return 1

    link.parent.mkdir(parents=True, exist_ok=True)
    if args.copy:
        import shutil

        print(f"Copy {drive_dir} → {link} ...")
        shutil.copytree(drive_dir, link)
    else:
        link.symlink_to(drive_dir, target_is_directory=True)
        print(f"Symlink {link} → {drive_dir}")

    if not _ok_tree(link):
        print("Link xong nhưng tree không hợp lệ", file=sys.stderr)
        return 1

    os.environ["SWIFTEDIT_WEIGHTS_FP16"] = str(link)
    print(f"OK. Dùng: --weights-fp16 {link}")
    print(f"export SWIFTEDIT_WEIGHTS_FP16={link}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
