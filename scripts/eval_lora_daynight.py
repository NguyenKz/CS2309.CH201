#!/usr/bin/env python3
"""
Đánh giá hold-out day↔night: baseline SwiftEdit vs +LoRA.

Độ đo chỏi nhau:
  - CLIP text-image với prompt night/day (semantics)
  - CLIP image-image source↔edit (giữ content; không được sụp)
  - (Tuỳ chọn) DINO nếu torch.hub tải được

Ví dụ (cần weights SwiftEdit + meta.jsonl + ảnh holdout):
  python scripts/eval_lora_daynight.py \\
    --data-root data/daynight_lora --split holdout \\
    --lora-path results/lora_daynight \\
    --output-dir results/daynight_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def parse_args():
    p = argparse.ArgumentParser(description="Eval LoRA day↔night vs baseline")
    p.add_argument("--data-root", type=Path, default=ROOT / "data" / "daynight_lora")
    p.add_argument("--split", default="holdout")
    p.add_argument("--lora-path", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "daynight_eval")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--max-samples", type=int, default=20)
    p.add_argument(
        "--skip-infer",
        action="store_true",
        help="Chỉ tính metric trên ảnh đã có trong output-dir",
    )
    return p.parse_args()


def load_meta(data_root: Path, split: str):
    meta_path = data_root / "meta.jsonl"
    if not meta_path.exists():
        meta_path = data_root / "meta.example.jsonl"
    rows = []
    with meta_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split", "train") == split:
                rows.append(row)
    return rows


@torch.no_grad()
def clip_metrics(device: torch.device):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()

    def score_text(image: Image.Image, text: str) -> float:
        inputs = proc(text=[text], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        return float(out.logits_per_image[0, 0].cpu())

    def score_image(a: Image.Image, b: Image.Image) -> float:
        inputs = proc(images=[a, b], return_tensors="pt")
        pixel = inputs["pixel_values"].to(device)
        feats = model.get_image_features(pixel)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return float((feats[0] @ feats[1]).cpu())

    return score_text, score_image


def run_swiftedit_once(
    image_path: Path,
    src_p: str,
    edit_p: str,
    inverse_model,
    aux_model,
    ip_sb_model,
    out_path: Path,
):
    sys.path.insert(0, str(ROOT / "SwiftEdit"))
    from infer import edit_image
    from torchvision.utils import save_image

    res = edit_image(
        str(image_path),
        src_p or "",
        edit_p,
        inverse_model,
        aux_model,
        ip_sb_model,
        scale_edit=0.2,
        scale_non_edit=1.0,
        mask_threshold=0.5,
        user_mask=None,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(res, str(out_path))
    return Image.open(out_path).convert("RGB")


def main() -> int:
    args = parse_args()
    rows = load_meta(args.data_root, args.split)[: args.max_samples]
    if not rows:
        print(f"Không có mẫu split={args.split}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_text, score_image = clip_metrics(device)

    inverse_model = aux_model = ip_sb_model = None
    if not args.skip_infer:
        sys.path.insert(0, str(ROOT / "SwiftEdit"))
        from infer import get_device
        from models import AuxiliaryModel, InverseModel, IPSBV2Model

        from lora_utils import attach_lora_to_unet

        device_s = get_device()
        weights = ROOT / "SwiftEdit" / "swiftedit_weights"
        dtype = args.dtype
        channels_last = dtype == "fp16"
        inverse_model = InverseModel(
            str(weights / "inverse_ckpt-120k"),
            device=device_s,
            dtype=dtype,
            channels_last=channels_last,
        )
        aux_model = AuxiliaryModel(device=device_s, dtype=dtype)
        ip_sb_model = IPSBV2Model(
            str(weights / "sbv2_0.5"),
            str(weights / "ip_adapter_ckpt-90k/ip_adapter.bin"),
            aux_model,
            device=device_s,
            with_ip_mask_controller=True,
            dtype=dtype,
            channels_last=channels_last,
        )
        if args.lora_path and Path(args.lora_path).exists():
            print(f"[eval] gắn LoRA từ {args.lora_path}")
            ip_sb_model.unet = attach_lora_to_unet(ip_sb_model.unet, args.lora_path)

    csv_path = args.output_dir / "metrics.csv"
    fieldnames = [
        "id",
        "direction",
        "clip_target",
        "clip_image_sim",
        "has_lora",
        "output_path",
    ]
    rows_out = []

    for row in rows:
        for direction, src_key, edit_key, src_img_key, tgt_label in [
            (
                "day_to_night",
                "src_day",
                "edit_night",
                "day",
                "a photo taken at night",
            ),
            (
                "night_to_day",
                "src_night",
                "edit_day",
                "night",
                "a photo taken in daylight",
            ),
        ]:
            src_path = args.data_root / row[src_img_key]
            if not src_path.exists():
                print(f"[skip] thiếu {src_path}")
                continue
            tag = "lora" if args.lora_path else "baseline"
            out_path = args.output_dir / tag / f"{row['id']}_{direction}.png"
            src_img = Image.open(src_path).convert("RGB")
            if args.skip_infer:
                if not out_path.exists():
                    print(f"[skip] thiếu output {out_path}")
                    continue
                edited = Image.open(out_path).convert("RGB")
            else:
                edited = run_swiftedit_once(
                    src_path,
                    row.get(src_key, ""),
                    row.get(edit_key, ""),
                    inverse_model,
                    aux_model,
                    ip_sb_model,
                    out_path,
                )
            clip_t = score_text(edited, tgt_label)
            clip_i = score_image(src_img, edited)
            rec = {
                "id": row["id"],
                "direction": direction,
                "clip_target": round(clip_t, 4),
                "clip_image_sim": round(clip_i, 4),
                "has_lora": bool(args.lora_path),
                "output_path": str(out_path),
            }
            rows_out.append(rec)
            print(rec)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    if rows_out:
        mean_t = sum(r["clip_target"] for r in rows_out) / len(rows_out)
        mean_i = sum(r["clip_image_sim"] for r in rows_out) / len(rows_out)
        summary = {
            "n": len(rows_out),
            "clip_target_mean": mean_t,
            "clip_image_sim_mean": mean_i,
            "has_lora": bool(args.lora_path),
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print("[summary]", summary)
    print(f"[done] {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
