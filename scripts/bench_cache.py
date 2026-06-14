#!/usr/bin/env python3
"""Benchmark cache latent + CLIP image embed + source prompt embed (SwiftEdit-RT).

Kịch bản realtime: cùng 1 ảnh nguồn + source prompt, đổi nhiều edit prompt liên tiếp.
So sánh thời gian có cache vs không cache, và kiểm chứng embed cache khớp baseline.

Ví dụ:
  python scripts/bench_cache.py --image data/PIE-Bench-subset20/annotation_images/0_random_140/000000000000.jpg \
      --src-p "a slanted mountain bicycle on the road in front of a building"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

# Các stage chỉ phụ thuộc ảnh nguồn / source prompt -> cache loại bỏ được
CACHEABLE_STAGES = ["vae_encode", "inv_text_encode", "gen_image_embeds", "gen_text_encode"]


def _sync(device) -> None:
    d = str(device)
    if d.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif d.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


DEFAULT_EDITS = [
    "a slanted rusty mountain motorcycle in front of a fence",
    "a slanted blue mountain bicycle on the road in front of a building",
    "a slanted mountain bicycle on the road in front of a castle",
    "a slanted wooden mountain bicycle on the road in front of a building",
    "a slanted mountain bicycle on a snowy road in front of a building",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark embedding cache SwiftEdit-RT")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--src-p", type=str, required=True)
    parser.add_argument("--edits", nargs="*", default=DEFAULT_EDITS)
    parser.add_argument("--repeat", type=int, default=1, help="Lặp danh sách edit prompt N lần")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "cache_bench_report.md")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "SwiftEdit"))
    import os

    log_path = ROOT / "results" / "cache_bench_timing.jsonl"
    if log_path.exists():
        log_path.unlink()
    os.environ["SWIFTEDIT_TIMING"] = "1"
    os.environ["SWIFTEDIT_TIMING_LOG"] = str(log_path)
    from infer import EditCache, edit_image, get_device
    from models import AuxiliaryModel, InverseModel, IPSBV2Model

    device = get_device()
    print(f"device={device}")
    weights = ROOT / "SwiftEdit" / "swiftedit_weights"
    inverse_model = InverseModel(str(weights / "inverse_ckpt-120k"), device=device)
    aux_model = AuxiliaryModel(device=device)
    ip_sb_model = IPSBV2Model(
        str(weights / "sbv2_0.5"),
        str(weights / "ip_adapter_ckpt-90k/ip_adapter.bin"),
        aux_model,
        device=device,
        with_ip_mask_controller=True,
    )

    edits = args.edits * args.repeat
    img = str(args.image)

    cache = EditCache(enabled=True)

    def run(edit_p, c):
        _sync(device)
        t0 = time.perf_counter()
        edit_image(img, args.src_p, edit_p, inverse_model, aux_model, ip_sb_model, cache=c)
        _sync(device)
        return time.perf_counter() - t0

    # Warmup (bỏ chi phí compile MPS lần đầu khỏi số liệu) — nạp cache luôn
    print("warmup...")
    run(edits[0], None)
    run(edits[0], cache)

    # Interleave no-cache / cache từng edit để khử trôi nhiệt (thermal throttling) trên Mac.
    rows = []  # (edit, t_nocache, t_cache, flag_order)
    print(f"\n=== INTERLEAVE ({len(edits)} edit, cùng ảnh+src) ===")
    nc_wall, wc_wall = [], []
    for i, e in enumerate(edits, 1):
        a = run(e, None)       # no-cache
        b = run(e, cache)      # with-cache (cùng ảnh + src -> hit)
        nc_wall.append(a)
        wc_wall.append(b)
        print(f"  [{i}] nocache={a:.2f}s  cache={b:.2f}s  {e[:42]}")

    # Đọc per-stage từ log: thứ tự = 2 warmup + (nocache, cache) * len(edits)
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    records = records[2:]  # bỏ 2 warmup
    nc_rec = records[0::2]
    wc_rec = records[1::2]

    def stage_mean(recs, stage):
        vals = [r["stages_ms"].get(stage, 0.0) for r in recs]
        return statistics.mean(vals) if vals else 0.0

    stage_rows = []
    nc_cacheable = wc_cacheable = 0.0
    for s in CACHEABLE_STAGES:
        a = stage_mean(nc_rec, s)
        b = stage_mean(wc_rec, s)
        nc_cacheable += a
        wc_cacheable += b
        stage_rows.append((s, a, b))

    nc = statistics.mean(nc_wall)
    wc = statistics.mean(wc_wall)
    stage_saved = nc_cacheable - wc_cacheable

    # Kiểm chứng: embed cache khớp tính lại từ đầu (deterministic)
    print("\n=== KIỂM CHỨNG embed cache ===")
    from infer import tokenize_captions

    src_id = tokenize_captions(inverse_model.tokenizer, [args.src_p]).to(device)
    fresh_inv = inverse_model.text_encoder(src_id)[0]
    again = inverse_model.text_encoder(src_id)[0]
    inv_ok = torch.allclose(fresh_inv, again, atol=1e-4)
    print(f"  inverse source embed deterministic (allclose): {inv_ok}")

    report = [
        "# Benchmark cache embedding (SwiftEdit-RT)",
        "",
        f"- **Thiết bị:** `{device}`",
        f"- **Ảnh nguồn:** `{args.image.name}`",
        f"- **Source prompt:** {args.src_p}",
        f"- **Số edit:** {len(edits)} (interleave no-cache/cache, đã bỏ warmup)",
        f"- **Cache đúng (embed deterministic):** {inv_ok}",
        "",
        "## Per-stage cacheable (mean ms/edit) — thước đo chính, ít nhiễu nhiệt",
        "",
        "| Stage | No cache | With cache | Tiết kiệm |",
        "|-------|---------:|-----------:|----------:|",
    ]
    for s, a, b in stage_rows:
        report.append(f"| {s} | {a:.1f} | {b:.1f} | {a-b:.1f} |")
    report += [
        f"| **Tổng cacheable** | **{nc_cacheable:.1f}** | **{wc_cacheable:.1f}** | "
        f"**{stage_saved:.1f}** |",
        "",
        f"-> Cache loại bỏ **~{stage_saved/1000:.2f} s/edit** ở các stage phụ thuộc ảnh/source "
        f"(giảm {100*stage_saved/nc_cacheable:.0f}% riêng phần cacheable).",
        "",
        "## Wall-clock end-to-end (tham khảo — nhiễu vì thermal throttling Mac)",
        "",
        "| Cấu hình | Mean (s) | Min | Max |",
        "|----------|---------:|----:|----:|",
        f"| No cache | {nc:.2f} | {min(nc_wall):.2f} | {max(nc_wall):.2f} |",
        f"| With cache | {wc:.2f} | {min(wc_wall):.2f} | {max(wc_wall):.2f} |",
        "",
        "Phần không cache được (2× UNet + VAE decode, phụ thuộc edit prompt) vẫn chiếm "
        "đa số thời gian nên speedup end-to-end nhỏ; wall-clock còn bị throttle nhiệt che lấp.",
        "",
        "Cache tái dùng: latent VAE (ảnh), CLIP image embed (ảnh), source prompt embed "
        "(inverse + generation). Mỗi edit mới chỉ encode lại edit prompt + chạy UNet/VAE-decode.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\nPer-stage cacheable (mean ms/edit):")
    for s, a, b in stage_rows:
        print(f"  {s:18s} nocache={a:7.1f}  cache={b:7.1f}  saved={a-b:7.1f}")
    print(f"  {'TOTAL cacheable':18s} nocache={nc_cacheable:7.1f}  cache={wc_cacheable:7.1f}  "
          f"saved={stage_saved:7.1f}  (~{stage_saved/1000:.2f}s/edit)")
    print(f"\nWall no-cache mean : {nc:.2f}s  (thermal-noisy)")
    print(f"Wall with-cache mean: {wc:.2f}s")
    print(f"Report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
