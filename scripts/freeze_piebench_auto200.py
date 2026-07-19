#!/usr/bin/env python3
"""Tạo lại PIE-Bench-auto200 giống bench 2026-06-17 (deterministic, không random).

Gọi create_piebench_subset.build_subset với max_samples=200, per_category=200
(cùng logic notebook quality_speed: đổ đầy theo thứ tự category).

  python scripts/freeze_piebench_auto200.py
  python scripts/build_june17_jobs.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "PIE-Bench-auto200"
CACHE = ROOT / "data" / ".hf_pie_bench_pp"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    from create_piebench_subset import CATEGORY_DIRS, build_subset

    build_subset(
        out_dir=args.out_dir,
        max_samples=args.max_samples,
        per_category=args.max_samples,
        cache_dir=args.cache_dir,
    )
    mapping = json.loads((args.out_dir / "mapping_file.json").read_text(encoding="utf-8"))
    ids = sorted(mapping.keys())
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]
    meta = {
        "name": "PIE-Bench-auto200",
        "n_samples": len(ids),
        "hf_dataset": "UB-CVML-Group/PIE_Bench_pp",
        "max_samples": args.max_samples,
        "per_category": args.max_samples,
        "category_order": [c[0] for c in CATEGORY_DIRS],
        "sample_id_sha256_16": digest,
        "sample_ids": ids,
        "matches_quality_speed_bench": "2026-06-17 run_meta dataset=PIE-Bench-auto200",
    }
    (args.out_dir / "MANIFEST.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"MANIFEST → {args.out_dir / 'MANIFEST.json'} (n={len(ids)}, hash={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
