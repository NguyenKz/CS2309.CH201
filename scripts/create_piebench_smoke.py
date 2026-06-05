#!/usr/bin/env python3
"""Tạo subset PIE-Bench mini từ imgs_demo — test pipeline khi chưa tải form."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMOKE_DIR = ROOT / "data" / "PIE-Bench-smoke"
SWIFTEDIT = ROOT / "SwiftEdit"
DEMO = SWIFTEDIT / "assets" / "imgs_demo"

# mask RLE (PnP format): toàn ảnh 512×512 = 262144 pixel
FULL_MASK_512 = [0, 262144]

SAMPLES = {
    "smoke_dog_001": {
        "image_path": "0_random_140/smoke_dog.jpg",
        "original_prompt": "dog",
        "editing_prompt": "dog with mouth opened",
        "editing_instruction": "Open the dog's mouth",
        "editing_type_id": "0",
        "blended_word": "dog dog",
        "mask": FULL_MASK_512,
        "src_file": "02.jpg",
    },
    "smoke_woman_001": {
        "image_path": "0_random_140/smoke_woman.jpg",
        "original_prompt": "woman",
        "editing_prompt": "Taylor Swift",
        "editing_instruction": "Change woman to Taylor Swift",
        "editing_type_id": "0",
        "blended_word": "woman woman",
        "mask": FULL_MASK_512,
        "src_file": "woman_face.jpg",
    },
}


def main() -> int:
    if not DEMO.is_dir():
        print(f"Thiếu {DEMO} — clone repo đề tài trước.", file=sys.stderr)
        return 1

    img_dir = SMOKE_DIR / "annotation_images" / "0_random_140"
    img_dir.mkdir(parents=True, exist_ok=True)

    mapping = {}
    for sid, spec in SAMPLES.items():
        src = DEMO / spec.pop("src_file")
        dst = img_dir / Path(spec["image_path"]).name
        if not src.is_file():
            print(f"Thiếu {src}", file=sys.stderr)
            return 1
        shutil.copy2(src, dst)
        mapping[sid] = spec

    mapping_path = SMOKE_DIR / "mapping_file.json"
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    print(f"[smoke] OK → {SMOKE_DIR}")
    print(f"  mapping: {mapping_path}")
    print(f"  samples: {len(mapping)}")
    print("Chạy thử:")
    print(
        "  python scripts/run_piebench_eval.py "
        f"--piebench-dir {SMOKE_DIR} --max-samples 2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
