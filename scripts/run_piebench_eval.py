#!/usr/bin/env python3
"""Chạy SwiftEdit trên subset PIE-Bench và ghi metrics.csv."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.utils import save_image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from piebench_metrics import PieBenchMetrics
from piebench_utils import (
    load_mapping,
    mask_decode,
    resolve_piebench_dir,
    select_samples,
    strip_prompt_brackets,
)

METRIC_FIELDS = [
    "timestamp",
    "sample_id",
    "editing_type_id",
    "image_path",
    "src_p",
    "edit_p",
    "device",
    "runtime_s",
    "psnr_unedit",
    "mse_unedit",
    "clip_whole",
    "clip_edited",
    "output_path",
]


def run_piebench_eval(
    *,
    project_root: Path,
    inverse_model,
    aux_model,
    ip_sb_model,
    edit_image_fn,
    device: str,
    piebench_dir: Path | None = None,
    output_dir: Path | None = None,
    max_samples: int = 50,
    edit_categories: list[str] | None = None,
    sample_ids: list[str] | None = None,
    resume: bool = True,
    compute_metrics: bool = True,
    scale_ta: float = 1.0,
    scale_edit: float = 0.2,
    scale_non_edit: float = 1.0,
) -> Path:
    piebench_root = resolve_piebench_dir(project_root, piebench_dir)
    out_root = output_dir or (project_root / "results" / "piebench")
    edited_dir = out_root / "edited_images"
    edited_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "metrics.csv"
    progress_path = out_root / "progress.json"

    mapping = load_mapping(piebench_root)
    samples = select_samples(
        mapping,
        edit_categories=edit_categories,
        max_samples=max_samples,
        sample_ids=sample_ids,
    )
    image_root = piebench_root / "annotation_images"

    metrics_calc = PieBenchMetrics(device) if compute_metrics else None
    done_ids: set[str] = set()
    if resume and progress_path.is_file():
        done_ids = set(json.loads(progress_path.read_text(encoding="utf-8")).get("done", []))

    rows: list[dict] = []
    for sample_id, item in samples:
        rel_path = item["image_path"]
        src_path = image_root / rel_path
        if not src_path.is_file():
            print(f"skip {sample_id} — thiếu {src_path}")
            continue

        out_rel = Path(rel_path).with_suffix(".png")
        out_path = edited_dir / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        src_p = strip_prompt_brackets(item["original_prompt"])
        edit_p = strip_prompt_brackets(item["editing_prompt"])
        edit_mask = mask_decode(item["mask"])
        mask3 = edit_mask[:, :, np.newaxis].repeat(3, axis=2)

        src_image = Image.open(src_path).convert("RGB").resize((512, 512))

        if resume and sample_id in done_ids and out_path.is_file():
            tgt_image = Image.open(out_path).convert("RGB").resize((512, 512))
            runtime_s = None
            print(f"resume {sample_id} — dùng {out_path}")
        else:
            t0 = time.time()
            result = edit_image_fn(
                str(src_path),
                src_p,
                edit_p,
                inverse_model,
                aux_model,
                ip_sb_model,
                scale_ta=scale_ta,
                scale_edit=scale_edit,
                scale_non_edit=scale_non_edit,
            )
            runtime_s = round(time.time() - t0, 3)
            save_image(result, out_path)
            tgt_image = Image.open(out_path).convert("RGB").resize((512, 512))
            done_ids.add(sample_id)
            progress_path.write_text(
                json.dumps({"done": sorted(done_ids)}, indent=2),
                encoding="utf-8",
            )
            print(f"{sample_id}: {runtime_s}s → {out_path}")

        metric_vals = {
            "psnr_unedit": "",
            "mse_unedit": "",
            "clip_whole": "",
            "clip_edited": "",
        }
        if metrics_calc is not None:
            scores = metrics_calc.evaluate_edit(
                src_image, tgt_image, mask3, src_p, edit_p
            )
            metric_vals = {k: round(v, 4) if v == v else "" for k, v in scores.items()}

        rows.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sample_id": sample_id,
                "editing_type_id": item.get("editing_type_id", ""),
                "image_path": rel_path,
                "src_p": src_p,
                "edit_p": edit_p,
                "device": device,
                "runtime_s": runtime_s if runtime_s is not None else "",
                **metric_vals,
                "output_path": str(out_path),
            }
        )

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)

    print(f"PIE-Bench: {len(rows)} mẫu → {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="SwiftEdit eval trên PIE-Bench subset")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--piebench-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-metrics", action="store_true")
    args = parser.parse_args()

    root = args.project_root or Path.cwd()
    if root.name == "notebooks":
        root = root.parent

    if not args.no_metrics:
        try:
            import torchmetrics  # noqa: F401
        except ImportError:
            print(
                "Thiếu torchmetrics. Cài: pip install -r scripts/phase3_requirements.txt",
                file=sys.stderr,
            )
            raise SystemExit(1) from None

    # Kiểm tra data TRƯỚC khi load model (~30s+ trên Mac)
    try:
        piebench_root = resolve_piebench_dir(root, args.piebench_dir)
        print(f"PIE-Bench: {piebench_root}")
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        print(
            "\nTạo bộ test mini (2 ảnh demo, không cần form):",
            file=sys.stderr,
        )
        print("  python scripts/create_piebench_smoke.py", file=sys.stderr)
        print(
            "  python scripts/run_piebench_eval.py "
            "--piebench-dir data/PIE-Bench-smoke --max-samples 2",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    sys.path.insert(0, str(root / "SwiftEdit"))
    from infer import edit_image, get_device
    from models import AuxiliaryModel, IPSBV2Model, InverseModel

    device = get_device()
    weights = root / "SwiftEdit" / "swiftedit_weights"
    inverse_model = InverseModel(str(weights / "inverse_ckpt-120k"), device=device)
    aux_model = AuxiliaryModel(device=device)
    ip_sb_model = IPSBV2Model(
        str(weights / "sbv2_0.5"),
        str(weights / "ip_adapter_ckpt-90k/ip_adapter.bin"),
        aux_model,
        device=device,
        with_ip_mask_controller=True,
    )

    run_piebench_eval(
        project_root=root,
        inverse_model=inverse_model,
        aux_model=aux_model,
        ip_sb_model=ip_sb_model,
        edit_image_fn=edit_image,
        device=str(device),
        piebench_dir=args.piebench_dir,
        max_samples=args.max_samples,
        edit_categories=args.categories,
        resume=not args.no_resume,
        compute_metrics=not args.no_metrics,
    )


if __name__ == "__main__":
    main()
