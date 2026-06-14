#!/usr/bin/env python3
"""Tạo subset PIE-Bench từ HuggingFace UB-CVML-Group/PIE_Bench_pp (700 mẫu).

Dùng khi chưa có bản zip chính thức từ Google Form PnP Inversion.
Ảnh + prompt lấy từ parquet; format mapping tương thích run_piebench_eval.py.

Ví dụ:
  python scripts/create_piebench_subset.py --max-samples 20
  python scripts/run_piebench_eval.py --piebench-dir data/PIE-Bench-subset20 --max-samples 20 --no-resume
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "PIE-Bench-subset20"
HF_DATASET = "UB-CVML-Group/PIE_Bench_pp"

# Thư mục parquet → editing_type_id (theo PnP Inversion)
CATEGORY_DIRS = [
    ("0_random_140", "0"),
    ("1_change_object_80", "1"),
    ("2_add_object_80", "2"),
    ("3_delete_object_80", "3"),
    ("4_change_attribute_content_40", "4"),
    ("5_change_attribute_pose_40", "5"),
    ("6_change_attribute_color_40", "6"),
    ("7_change_attribute_material_40", "7"),
    ("8_change_background_80", "8"),
    ("9_change_style_80", "9"),
]


def _ensure_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        print("Cần pyarrow: python -m pip install pyarrow", file=sys.stderr)
        raise SystemExit(1) from e


def _download_hf(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if any(cache_dir.glob("*/V1-*.parquet")):
        return cache_dir
    print(f"[subset] Tải {HF_DATASET} → {cache_dir} ...")
    subprocess.run(
        ["hf", "download", HF_DATASET, "--repo-type", "dataset", "--local-dir", str(cache_dir)],
        check=True,
    )
    return cache_dir


def _parse_mask(mask_str: str) -> list[int]:
    parts = [int(x) for x in str(mask_str).split()]
    if len(parts) % 2:
        raise ValueError(f"mask RLE lẻ số phần tử: {mask_str!r}")
    return parts


def _parse_blended_word(blended_words: str) -> str:
    """Chuyển '["a,b", "c,d"]' → 'a b' (cặp đầu, dùng cho metadata)."""
    try:
        pairs = json.loads(blended_words)
    except json.JSONDecodeError:
        return blended_words.strip()
    if not pairs:
        return ""
    first = pairs[0]
    if "," in first:
        a, b = first.split(",", 1)
        return f"{a.strip()} {b.strip()}"
    return first.replace(",", " ").strip()


def _strip_brackets(prompt: str) -> str:
    return re.sub(r"\[([^\]]*)\]", r"\1", prompt)


def build_subset(
    *,
    out_dir: Path,
    max_samples: int,
    per_category: int,
    cache_dir: Path,
) -> Path:
    _ensure_pyarrow()
    import pandas as pd
    from PIL import Image

    hf_cache = _download_hf(cache_dir)
    out_dir = out_dir.resolve()
    img_root = out_dir / "annotation_images"
    if out_dir.is_dir():
        shutil.rmtree(out_dir)
    img_root.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, dict] = {}
    for folder, type_id in CATEGORY_DIRS:
        parquet_files = list((hf_cache / folder).glob("*.parquet"))
        if not parquet_files:
            print(f"[subset] skip {folder} — không có parquet")
            continue
        df = pd.read_parquet(parquet_files[0])
        take = min(per_category, len(df), max(0, max_samples - len(mapping)))
        for _, row in df.head(take).iterrows():
            if len(mapping) >= max_samples:
                break
            sample_id = str(row["id"])
            rel_path = f"{folder}/{sample_id}.jpg"
            dst = img_root / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)

            img_bytes = row["image"]["bytes"]
            Image.open(BytesIO(img_bytes)).convert("RGB").save(dst, quality=95)

            src_p = _strip_brackets(str(row["source_prompt"]))
            tgt_p = _strip_brackets(str(row["target_prompt"]))
            mapping[sample_id] = {
                "image_path": rel_path,
                "original_prompt": src_p,
                "editing_prompt": tgt_p,
                "editing_instruction": tgt_p,
                "editing_type_id": type_id,
                "blended_word": _parse_blended_word(str(row.get("blended_words", ""))),
                "mask": _parse_mask(str(row["mask"])),
            }
        if len(mapping) >= max_samples:
            break

    mapping_path = out_dir / "mapping_file.json"
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[subset] OK → {out_dir} ({len(mapping)} mẫu)")
    print(f"  mapping: {mapping_path}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Tạo subset PIE-Bench từ HuggingFace")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--per-category", type=int, default=2, help="Mẫu mỗi loại edit (mặc định 2×10=20)")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_pie_bench_pp")
    args = parser.parse_args()
    build_subset(
        out_dir=args.out_dir,
        max_samples=args.max_samples,
        per_category=args.per_category,
        cache_dir=args.cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
