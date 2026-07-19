#!/usr/bin/env python3
"""Eval precision theo configs user chọn (không bắt buộc chạy hết list).

Hệ quy chiếu jobs: data/jobs_june17.json (200×3).
Xuất bundle đủ để so sánh local sau: disk / memory / quality / inputs / outputs / meta.

Ví dụ:
  python scripts/run_precision_disk_vram_eval.py --configs fp32
  python scripts/run_precision_disk_vram_eval.py --configs fp16,fp8 --max-jobs 6
  python scripts/run_precision_disk_vram_eval.py --configs 16_weight,4_weight --max-jobs 600
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# T4 16GB: hạn chế thread CPU — tránh spike RAM/VRAM khi eval tuần tự.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from precision_catalog import (  # noqa: E402
    CONFIGS,
    help_table,
    needs_fp16_disk,
    resolve_config_names,
)

PREV_BENCH_HINT = {
    "baseline_fp32": "quality_speed_bench baseline_fp32",
    "improved_fp16_cache": "quality_speed_bench improved_fp16_cache",
    "improved_fp8_cache": "quality_speed_bench improved_fp8_cache",
    "improved_fp4_cache": "quality_speed_bench improved_fp4_cache",
    "fp16_disk": "disk fp16 + EditCache (mới vs June17)",
    "fp4_from_fp16": "disk fp16 → quant fp4 + EditCache (mới vs June17)",
}


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _fmt_mb(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.0f}"


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _free(device: str) -> None:
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    elif device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _reset_peak(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _peak_alloc_mb(device: str) -> float | None:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    if device.startswith("mps") and torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / (1024**2)
        except Exception:
            return None
    return None


def _driver_used_mb(device: str) -> float | None:
    if device.startswith("cuda") and torch.cuda.is_available():
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip().splitlines()[0])
    if device.startswith("mps") and torch.backends.mps.is_available():
        try:
            return torch.mps.driver_allocated_memory() / (1024**2)
        except Exception:
            return None
    return None


def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _jobs_hash(jobs: list[dict]) -> str:
    blob = json.dumps(
        [(j["job_id"], j["image_rel"], j["src_prompt"], j["edit_prompt"]) for j in jobs],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _jobs_from_june17_manifest(
    manifest: Path, *, max_jobs: int | None
) -> list[dict]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    dataset_dir = ROOT / data.get("dataset_dir", "data/PIE-Bench-auto200")
    jobs: list[dict] = []
    for i, j in enumerate(data["jobs"]):
        rel = j.get("image_rel") or ""
        img = dataset_dir / rel
        if not img.is_file():
            alt = Path(j.get("image", ""))
            if alt.is_file():
                img = alt
        if not img.is_file():
            continue
        sample_id = j.get("sample_id") or img.stem
        prompt_idx = sum(1 for x in jobs if x["sample_id"] == sample_id)
        job_id = f"{sample_id}_{prompt_idx}"
        jobs.append(
            {
                "job_id": job_id,
                "job_index": i,
                "sample_id": sample_id,
                "prompt_idx": prompt_idx,
                "image": img,
                "image_rel": rel or str(img),
                "src_prompt": j["src_prompt"],
                "edit_prompt": j["edit_prompt"],
            }
        )
        if max_jobs is not None and len(jobs) >= max_jobs:
            break
    return jobs


def _jobs_from_mapping(bench_root: Path, n_images: int) -> list[dict]:
    mapping_path = bench_root / "mapping_file.json"
    if not mapping_path.is_file():
        return []
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    jobs: list[dict] = []
    for _key, meta in data.items():
        rel = meta.get("image_path") or ""
        src_p = (meta.get("original_prompt") or "").strip()
        edit_p = (meta.get("editing_prompt") or "").strip()
        if not rel or not src_p or not edit_p:
            continue
        img = bench_root / "annotation_images" / rel
        if not img.is_file():
            img = bench_root / rel
        if not img.is_file():
            continue
        sample_id = Path(rel).stem
        job_id = f"{sample_id}_0"
        jobs.append(
            {
                "job_id": job_id,
                "job_index": len(jobs),
                "sample_id": sample_id,
                "prompt_idx": 0,
                "image": img,
                "image_rel": f"annotation_images/{rel}" if not rel.startswith("annotation") else rel,
                "src_prompt": src_p,
                "edit_prompt": edit_p,
            }
        )
        if len(jobs) >= n_images:
            break
    return jobs


def _pick_jobs(
    n_images: int,
    edits_per_image: int,
    *,
    jobs_manifest: Path | None,
    max_jobs: int | None,
) -> tuple[list[dict], str]:
    candidates: list[Path] = []
    if jobs_manifest is not None:
        candidates.append(jobs_manifest)
    candidates.extend(
        [
            ROOT / "data" / "jobs_june17.json",
            ROOT / "data" / "PIE-Bench-auto200" / "jobs_june17.json",
        ]
    )
    for mf in candidates:
        if not mf.is_file():
            continue
        limit = max_jobs
        if limit is None and n_images:
            limit = n_images * max(1, edits_per_image)
        jobs = _jobs_from_june17_manifest(mf, max_jobs=limit)
        if not jobs:
            continue
        n_missing = sum(1 for j in jobs if not j["image"].is_file())
        if n_missing:
            print(
                f"Manifest {mf}: thiếu ảnh. Chạy freeze_piebench_auto200.py",
                file=sys.stderr,
            )
        print(f"Jobs từ {mf} ({len(jobs)} jobs)")
        return jobs, str(mf)

    for base in (
        ROOT / "data" / "PIE-Bench-auto200",
        ROOT / "data" / "PIE-Bench-subset20",
        ROOT / "data" / "PIE-Bench-smoke",
    ):
        jobs = _jobs_from_mapping(base, n_images)
        if jobs:
            print(f"Jobs fallback mapping {base}")
            return jobs, f"mapping:{base}"
    raise FileNotFoundError(
        "Không có jobs. Chạy freeze_piebench_auto200.py + build_june17_jobs.py"
    )


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.dim() == 4:
        t = t[-1]
    arr = t.clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def _psnr_mse(a: Image.Image, b: Image.Image) -> tuple[float, float]:
    xa = np.asarray(a.convert("RGB").resize((512, 512)), dtype=np.float64) / 255.0
    xb = np.asarray(b.convert("RGB").resize((512, 512)), dtype=np.float64) / 255.0
    mse = float(np.mean((xa - xb) ** 2))
    if mse <= 1e-12:
        return 99.0, mse
    return 10.0 * np.log10(1.0 / mse), mse


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _zip_bundle(out_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            if p.is_file() and p.name != zip_path.name:
                zf.write(p, arcname=str(p.relative_to(out_dir)))


def _copy_inputs(jobs: list[dict], out_dir: Path) -> None:
    inp = out_dir / "inputs"
    img_dir = inp / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    jobs_export = []
    for j in jobs:
        sid = j["sample_id"]
        if sid not in seen and j["image"].is_file():
            dst = img_dir / f"{sid}{j['image'].suffix.lower() or '.jpg'}"
            shutil.copy2(j["image"], dst)
            seen.add(sid)
        jobs_export.append(
            {
                "job_id": j["job_id"],
                "job_index": j["job_index"],
                "sample_id": j["sample_id"],
                "prompt_idx": j["prompt_idx"],
                "image_rel": j["image_rel"],
                "input_copy": f"inputs/images/{sid}{j['image'].suffix.lower() or '.jpg'}",
                "src_prompt": j["src_prompt"],
                "edit_prompt": j["edit_prompt"],
            }
        )
    (inp / "jobs.json").write_text(
        json.dumps({"n_jobs": len(jobs_export), "jobs": jobs_export}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precision eval — chỉ chạy configs được chọn",
        epilog="Catalog:\n" + help_table(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="CSV alias/key — chỉ chạy những cái này. VD: fp32 | fp16,fp8 | 16_weight,4_weight",
    )
    parser.add_argument("--list-configs", action="store_true", help="In catalog rồi thoát")
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--edits-per-image", type=int, default=3)
    parser.add_argument(
        "--jobs-manifest",
        type=Path,
        default=ROOT / "data" / "jobs_june17.json",
    )
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument(
        "--weights-fp32",
        type=Path,
        default=ROOT / "SwiftEdit" / "swiftedit_weights",
    )
    parser.add_argument(
        "--weights-fp16",
        type=Path,
        default=ROOT / "SwiftEdit" / "swiftedit_weights_fp16",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--src-p", type=str, default=None)
    parser.add_argument("--edit-p", type=str, default=None)
    args = parser.parse_args()

    if args.list_configs:
        print(help_table())
        return 0
    if not args.configs:
        print("Cần --configs (hoặc --list-configs).", file=sys.stderr)
        return 2

    try:
        config_names = resolve_config_names(args.configs)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = "+".join(config_names)[:80]
    out_dir = args.out or (
        ROOT / "experimental_data" / f"precision_run_{stamp}_{slug}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "edited_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT / "SwiftEdit"))
    from infer import EditCache, edit_image, get_device
    from models import AuxiliaryModel, InverseModel, IPSBV2Model

    device = get_device()
    # Eval tuần tự 1 config → xong mới load config khác (không song song — OOM T4 16GB).
    print(
        f"device={device} configs={config_names} "
        f"(sequential only, OMP/MKL threads=1, VRAM~16GB safe)",
        flush=True,
    )
    print(f"out={out_dir}")

    # disk inventory (luôn đo cả 2 cây nếu tồn tại)
    disk_rows = []
    for label, path in [
        ("swiftedit_weights_fp32", args.weights_fp32),
        ("swiftedit_weights_fp16", args.weights_fp16),
    ]:
        b = _dir_bytes(path)
        disk_rows.append(
            {
                "label": label,
                "path": str(path),
                "bytes": b,
                "gib": round(b / (1024**3), 4),
                "exists": path.exists(),
            }
        )
    _write_csv(out_dir / "disk.csv", disk_rows)
    fp32_b = disk_rows[0]["bytes"]
    fp16_b = disk_rows[1]["bytes"]
    disk_save_pct = (1 - fp16_b / fp32_b) * 100 if fp32_b and fp16_b else None

    if needs_fp16_disk(config_names):
        from ensure_precision_weights import ensure_fp16_weights

        rc = ensure_fp16_weights(
            configs=config_names,
            weights_fp32=args.weights_fp32,
            weights_fp16=args.weights_fp16,
        )
        if rc != 0:
            return rc

    if any(CONFIGS[c]["weights"] == "fp32" for c in config_names):
        if not (args.weights_fp32 / "sbv2_0.5").is_dir():
            print(f"Thiếu weights fp32 tại {args.weights_fp32}", file=sys.stderr)
            return 1

    jobs, jobs_src = _pick_jobs(
        args.n_images,
        args.edits_per_image,
        jobs_manifest=args.jobs_manifest,
        max_jobs=args.max_jobs,
    )
    if args.src_p or args.edit_p:
        for j in jobs:
            if args.src_p:
                j["src_prompt"] = args.src_p
            if args.edit_p:
                j["edit_prompt"] = args.edit_p

    jhash = _jobs_hash(jobs)
    _copy_inputs(jobs, out_dir)

    memory_rows: list[dict] = []
    quality_rows: list[dict] = []
    ref_images: dict[str, Image.Image] = {}
    ref_config_name = (
        "baseline_fp32" if "baseline_fp32" in config_names else None
    )
    print(f"PSNR reference in-run: {ref_config_name or '(none — so fp32 lúc compare local)'}")
    print(f"jobs_hash={jhash}")

    for cname in config_names:
        meta = CONFIGS[cname]
        quant = meta["quant"]
        if meta["needs_cuda_quant"] and not device.startswith("cuda"):
            print(f"SKIP {cname} — cần CUDA", file=sys.stderr)
            memory_rows.append(
                {
                    "config": cname,
                    "phase": "skipped",
                    "device": device,
                    "peak_alloc_mb": None,
                    "driver_used_mb": None,
                    "load_seconds": None,
                    "note": "requires CUDA",
                }
            )
            continue

        wroot = (
            args.weights_fp16 if meta["weights"] == "fp16" else args.weights_fp32
        )
        # channels_last chỉ CUDA có ích; MPS giữ False nếu catalog True nhưng non-cuda
        ch_last = bool(meta["channels_last"] and device.startswith("cuda"))
        print(
            f"\n=== {cname} weights={wroot.name} dtype={meta['dtype']} "
            f"quant={quant} cache={meta['use_cache']} ==="
        )
        _free(device)
        _reset_peak(device)
        t_load0 = time.perf_counter()
        try:
            inv = InverseModel(
                str(wroot / "inverse_ckpt-120k"),
                device=device,
                dtype=meta["dtype"],
                channels_last=ch_last,
                quant=quant,
            )
            aux = AuxiliaryModel(device=device, dtype=meta["dtype"])
            ip = IPSBV2Model(
                str(wroot / "sbv2_0.5"),
                str(wroot / "ip_adapter_ckpt-90k" / "ip_adapter.bin"),
                aux,
                device=device,
                with_ip_mask_controller=True,
                dtype=meta["dtype"],
                channels_last=ch_last,
                quant=quant,
            )
        except Exception as e:
            print(f"SKIP {cname} load fail: {e}", file=sys.stderr)
            traceback.print_exc()
            memory_rows.append(
                {
                    "config": cname,
                    "phase": "load_failed",
                    "device": device,
                    "peak_alloc_mb": None,
                    "driver_used_mb": None,
                    "load_seconds": None,
                    "note": str(e)[:200],
                }
            )
            continue

        _sync(device)
        load_s = time.perf_counter() - t_load0
        peak_load = _peak_alloc_mb(device)
        driver_load = _driver_used_mb(device)
        memory_rows.append(
            {
                "config": cname,
                "phase": "after_load",
                "device": device,
                "load_seconds": round(load_s, 2),
                "peak_alloc_mb": round(peak_load, 2) if peak_load is not None else None,
                "driver_used_mb": round(driver_load, 2) if driver_load is not None else None,
                "note": PREV_BENCH_HINT.get(cname, ""),
            }
        )
        print(f"  load {load_s:.1f}s peak={peak_load} driver={driver_load}")

        cache = EditCache(enabled=True) if meta["use_cache"] else None

        # warmup
        wj = jobs[0]
        _ = edit_image(
            str(wj["image"]),
            wj["src_prompt"],
            wj["edit_prompt"],
            inv,
            aux,
            ip,
            scale_edit=0.2,
            scale_non_edit=1.0,
            mask_threshold=0.5,
            cache=cache,
        )
        _sync(device)
        if cache is not None:
            cache = EditCache(enabled=True)  # reset sau warmup

        peak_warm = _peak_alloc_mb(device)
        memory_rows.append(
            {
                "config": cname,
                "phase": "after_warmup_edit",
                "device": device,
                "load_seconds": None,
                "peak_alloc_mb": round(peak_warm, 2) if peak_warm is not None else None,
                "driver_used_mb": _driver_used_mb(device),
                "note": "warmup done",
            }
        )

        last_sample: str | None = None
        n_jobs = len(jobs)
        t_jobs0 = time.perf_counter()
        print(f"  >>> eval {cname}: {n_jobs} jobs (progress mỗi job)", flush=True)
        for ji, j in enumerate(jobs, start=1):
            # cache miss khi đổi ảnh (hoặc tắt cache)
            if cache is not None and j["sample_id"] != last_sample:
                cache_state = "miss"
            elif cache is not None:
                cache_state = "hit"
            else:
                cache_state = "off"
            last_sample = j["sample_id"]

            _reset_peak(device)
            t0 = time.perf_counter()
            res = edit_image(
                str(j["image"]),
                j["src_prompt"],
                j["edit_prompt"],
                inv,
                aux,
                ip,
                scale_edit=0.2,
                scale_non_edit=1.0,
                mask_threshold=0.5,
                cache=cache,
            )
            _sync(device)
            dt = time.perf_counter() - t0
            peak_ed = _peak_alloc_mb(device)
            pil = _tensor_to_pil(res)
            out_png = img_dir / cname / f"{j['job_id']}.png"
            out_png.parent.mkdir(parents=True, exist_ok=True)
            pil.save(out_png)

            psnr_ref = mse_ref = None
            psnr_fp32 = None
            if ref_config_name == cname:
                ref_images[j["job_id"]] = pil.copy()
                psnr_ref, mse_ref = 99.0, 0.0
                psnr_fp32 = 99.0
            elif ref_config_name and j["job_id"] in ref_images:
                psnr_ref, mse_ref = _psnr_mse(pil, ref_images[j["job_id"]])
                if ref_config_name == "baseline_fp32":
                    psnr_fp32 = psnr_ref

            quality_rows.append(
                {
                    "config": cname,
                    "job_id": j["job_id"],
                    "job_index": j["job_index"],
                    "sample_id": j["sample_id"],
                    "prompt_idx": j["prompt_idx"],
                    "image_rel": j["image_rel"],
                    "src_prompt": j["src_prompt"],
                    "edit_prompt": j["edit_prompt"],
                    "seconds": round(dt, 4),
                    "cache": cache_state,
                    "psnr_vs_ref": round(psnr_ref, 4) if psnr_ref is not None else None,
                    "psnr_vs_fp32": round(psnr_fp32, 4) if psnr_fp32 is not None else None,
                    "ref_config": ref_config_name,
                    "mse_vs_ref": mse_ref,
                    "peak_alloc_mb": round(peak_ed, 2) if peak_ed is not None else None,
                    "driver_used_mb": _driver_used_mb(device),
                    "out_png": str(out_png.relative_to(out_dir)),
                }
            )
            elapsed = time.perf_counter() - t_jobs0
            avg = elapsed / ji
            remain = avg * (n_jobs - ji)
            pct = 100.0 * ji / n_jobs
            eta_m, eta_s = divmod(int(remain), 60)
            psnr_s = f"{psnr_ref:.2f}" if psnr_ref is not None else "n/a"
            print(
                f"  [{cname}] {ji}/{n_jobs} ({pct:5.1f}%)  "
                f"{j['job_id']}  {dt:.2f}s  cache={cache_state}  PSNR={psnr_s}  "
                f"ETA {eta_m}m{eta_s:02d}s",
                flush=True,
            )

        del inv, aux, ip, cache
        inv = aux = ip = cache = None
        _free(device)
        print(f"=== done {cname} — freed VRAM before next config ===", flush=True)

    if not quality_rows:
        print("Không có quality rows.", file=sys.stderr)
        return 1

    _write_csv(out_dir / "memory.csv", memory_rows)
    _write_csv(out_dir / "quality.csv", quality_rows)

    summary_rows = []
    for cname in config_names:
        qs = [r for r in quality_rows if r["config"] == cname]
        ms = [r for r in memory_rows if r["config"] == cname and r["phase"] == "after_load"]
        if not qs and not ms:
            continue
        psnrs = [r["psnr_vs_fp32"] for r in qs if r.get("psnr_vs_fp32") is not None]
        psnrs_ref = [r["psnr_vs_ref"] for r in qs if r.get("psnr_vs_ref") is not None]
        secs = [r["seconds"] for r in qs]
        hits = [r["seconds"] for r in qs if r.get("cache") == "hit"]
        misses = [r["seconds"] for r in qs if r.get("cache") == "miss"]
        summary_rows.append(
            {
                "config": cname,
                "n_edits": len(qs),
                "seconds_mean": round(statistics.mean(secs), 4) if secs else None,
                "seconds_cache_hit_mean": round(statistics.mean(hits), 4) if hits else None,
                "seconds_cache_miss_mean": round(statistics.mean(misses), 4) if misses else None,
                "psnr_vs_fp32_mean": round(statistics.mean(psnrs), 4) if psnrs else None,
                "psnr_vs_ref_mean": round(statistics.mean(psnrs_ref), 4) if psnrs_ref else None,
                "ref_config": ref_config_name,
                "peak_alloc_mb_after_load": ms[0]["peak_alloc_mb"] if ms else None,
                "driver_used_mb_after_load": ms[0]["driver_used_mb"] if ms else None,
                "load_seconds": ms[0]["load_seconds"] if ms else None,
                "map_to_prev_bench": PREV_BENCH_HINT.get(cname, ""),
            }
        )
    _write_csv(out_dir / "quality_summary.csv", summary_rows)

    inventory = {
        "schema": "precision_run_v1",
        "jobs_hash": jhash,
        "jobs_source": jobs_src,
        "n_jobs": len(jobs),
        "configs_requested": config_names,
        "configs_with_quality_rows": sorted({r["config"] for r in quality_rows}),
        "has_baseline_fp32_images": (out_dir / "edited_images" / "baseline_fp32").is_dir(),
        "files": [
            "disk.csv",
            "memory.csv",
            "quality.csv",
            "quality_summary.csv",
            "run_meta.json",
            "inventory.json",
            "inputs/jobs.json",
            "inputs/images/",
            "edited_images/",
            "report.md",
            "bundle.zip",
        ],
    }
    (out_dir / "inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )

    meta = {
        "schema": "precision_run_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git_commit": _git_commit(),
        "configs": config_names,
        "config_defs": {c: CONFIGS[c] for c in config_names},
        "n_jobs": len(jobs),
        "jobs_hash": jhash,
        "jobs_source": jobs_src,
        "weights_fp32": str(args.weights_fp32),
        "weights_fp16": str(args.weights_fp16),
        "disk_fp32_bytes": fp32_b,
        "disk_fp16_bytes": fp16_b,
        "disk_save_pct": disk_save_pct,
        "ref_config": ref_config_name,
        "out_dir": str(out_dir),
        "note": "So sánh đa config local bằng scripts/compare_precision_runs.py --runs ...",
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Precision run",
        "",
        f"- UTC: `{meta['timestamp_utc']}`",
        f"- Device: `{device}` | torch `{meta['torch']}` | git `{meta['git_commit']}`",
        f"- Configs: {', '.join(f'`{c}`' for c in config_names)}",
        f"- Jobs: **{len(jobs)}** | `jobs_hash={jhash}`",
        f"- PSNR in-run vs: `{ref_config_name or 'n/a — so fp32 lúc merge local'}`",
        "",
        "## Disk",
        "",
        "| Label | GiB | Exists |",
        "|---|---:|:---:|",
    ]
    for r in disk_rows:
        lines.append(f"| {r['label']} | {r['gib']} | {r['exists']} |")
    lines += [
        "",
        "## Summary",
        "",
        "| Config | n | s/edit | hit | miss | PSNR↔fp32 | peak MB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['config']} | {r['n_edits']} | {r['seconds_mean']} | "
            f"{r['seconds_cache_hit_mean']} | {r['seconds_cache_miss_mean']} | "
            f"{r['psnr_vs_fp32_mean']} | {r['peak_alloc_mb_after_load']} |"
        )
    lines += [
        "",
        "## Export checklist",
        "",
        "- `inputs/jobs.json` + `inputs/images/` — input đầy đủ",
        "- `edited_images/<config>/*.png` — output",
        "- `quality.csv` — duration, cache, PSNR, VRAM/job",
        "- `memory.csv` — load / warmup peak",
        "- `disk.csv` — dung lượng weights",
        "- `inventory.json` + `run_meta.json` — join key = `jobs_hash`",
        "",
        "Local merge: `python scripts/compare_precision_runs.py`",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    zip_path = out_dir / "bundle.zip"
    _zip_bundle(out_dir, zip_path)
    print(f"Done. Bundle: {zip_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
