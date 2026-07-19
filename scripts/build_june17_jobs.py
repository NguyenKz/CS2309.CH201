#!/usr/bin/env python3
"""Sinh jobs_june17.json = 200 ảnh × 3 template (khớp quality_speed_bench_2026-06-17).

Cần đã có data/PIE-Bench-auto200/mapping_file.json (chạy freeze_piebench_auto200.py trước).

  python scripts/build_june17_jobs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "PIE-Bench-auto200"

# Từ experimental_data/quality_speed_bench_2026-06-17/run_meta.json
JUNE17_TEMPLATES = [
    "{edit}",
    "{src} at night, dark lighting",
    "{src} in winter, covered in snow",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Mặc định: <dataset-dir>/jobs_june17.json",
    )
    args = parser.parse_args()
    ds = args.dataset_dir
    mapping_path = ds / "mapping_file.json"
    if not mapping_path.is_file():
        raise SystemExit(
            f"Thiếu {mapping_path}. Chạy: python scripts/freeze_piebench_auto200.py"
        )
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    jobs = []
    for _sid, meta in mapping.items():
        rel = meta["image_path"]
        src = meta["original_prompt"]
        edit = meta.get("editing_prompt") or meta.get("editing_instruction") or src
        img = ds / "annotation_images" / rel
        for tpl in JUNE17_TEMPLATES:
            edit_p = tpl.format(src=src, edit=edit)
            jobs.append(
                {
                    "image": str(img.relative_to(ROOT))
                    if img.is_relative_to(ROOT)
                    else str(img),
                    "image_rel": f"annotation_images/{rel}",
                    "src_prompt": src,
                    "edit_prompt": edit_p,
                    "sample_id": _sid,
                }
            )
    out = args.out or (ds / "jobs_june17.json")
    payload = {
        "source_run": "quality_speed_bench_2026-06-17",
        "templates": JUNE17_TEMPLATES,
        "n_images": len(mapping),
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(jobs)} jobs = {len(mapping)} × {len(JUNE17_TEMPLATES)})")
    # Bản nhẹ commit được (không kèm path tuyệt đối / chỉ image_rel)
    light = {
        "source_run": payload["source_run"],
        "templates": JUNE17_TEMPLATES,
        "n_images": len(mapping),
        "n_jobs": len(jobs),
        "dataset_dir": "data/PIE-Bench-auto200",
        "jobs": [
            {
                "image_rel": j["image_rel"],
                "src_prompt": j["src_prompt"],
                "edit_prompt": j["edit_prompt"],
                "sample_id": j["sample_id"],
            }
            for j in jobs
        ],
    }
    light_path = ROOT / "data" / "jobs_june17.json"
    light_path.parent.mkdir(parents=True, exist_ok=True)
    light_path.write_text(json.dumps(light, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Also wrote commit-friendly {light_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
