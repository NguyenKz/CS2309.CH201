#!/usr/bin/env python3
"""So sánh đầy đủ test_data: tốc độ (miss/hit) + PSNR vs ảnh baseline_fp32.

1. Giải nén zip → extract-dir/<config>/ (skip nếu đã đủ PNG).
2. Tốc độ A (prompt0 fair) + B (EditCache).
3. PSNR vs FP32 gốc: prompt0 (n=200) + full 600 (+ miss/hit nếu có).

Ví dụ:
  python scripts/compare_test_data_full.py --zips-dir test_data --extract-dir test_data/extracted
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

# zip name substring → config dir name
ZIP_TO_CONFIG = [
    ("baseline_fp32", "baseline_fp32"),
    ("fp16_disk_xformers", "fp16_disk_xformers"),
    ("fp16_disk", "fp16_disk"),
    ("improved_fp16_cache", "improved_fp16_cache"),
    ("improved_fp4_cache", "improved_fp4_cache"),
]

CONFIG_ORDER = [
    "baseline_fp32",
    "improved_fp16_cache",
    "fp16_disk",
    "fp16_disk_xformers",
    "improved_fp4_cache",
]

EXTRACT_MEMBERS_SUFFIXES = (
    "quality.csv",
    "quality_summary.csv",
    "run_meta.json",
    "memory.csv",
    "inventory.json",
)


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _fmt(x: float | None, nd: int = 4) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def _pct(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return 100.0 * num / den


def _psnr(a: Path, b: Path) -> float | None:
    if not a.is_file() or not b.is_file():
        return None
    xa = np.asarray(Image.open(a).convert("RGB").resize((512, 512)), dtype=np.float64) / 255.0
    xb = np.asarray(Image.open(b).convert("RGB").resize((512, 512)), dtype=np.float64) / 255.0
    mse = float(np.mean((xa - xb) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _config_from_zip_name(name: str) -> str | None:
    # longer / more specific keys first (xformers before fp16_disk)
    for key, cfg in ZIP_TO_CONFIG:
        if key in name:
            return cfg
    return None


def _count_pngs(cfg_dir: Path, config: str) -> int:
    d = cfg_dir / "edited_images" / config
    if not d.is_dir():
        return 0
    return sum(1 for _ in d.glob("*.png"))


def extract_zip(zip_path: Path, out_dir: Path, config: str, force: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_png = _count_pngs(out_dir, config)
    has_q = (out_dir / "quality.csv").is_file()
    if not force and n_png >= 600 and has_q:
        print(f"  skip extract {config}: already {n_png} png + quality.csv")
        return

    print(f"  extracting {zip_path.name} → {out_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            # strip leading folder if any
            base = name.split("/")[-1]
            keep = False
            if "/edited_images/" in name or name.startswith("edited_images/"):
                keep = True
            elif any(name.endswith(s) or base == s for s in EXTRACT_MEMBERS_SUFFIXES):
                keep = True
            if not keep:
                continue
            # normalize destination path (drop zip root prefix if present)
            parts = Path(name).parts
            if parts and parts[0].startswith("precision_run"):
                rel = Path(*parts[1:]) if len(parts) > 1 else Path(base)
            else:
                rel = Path(name)
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, dest.open("wb") as dst:
                dst.write(src.read())
    print(f"  done {config}: png={_count_pngs(out_dir, config)}")


def load_quality(cfg_dir: Path) -> list[dict]:
    p = cfg_dir / "quality.csv"
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_meta(cfg_dir: Path) -> dict:
    p = cfg_dir / "run_meta.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def analyze_speed(config: str, rows: list[dict]) -> dict:
    a_rows = [
        r
        for r in rows
        if str(r.get("prompt_idx", "")) == "0" and r.get("cache") in ("off", "miss")
    ]
    a_secs = [float(r["seconds"]) for r in a_rows]
    miss_secs = [float(r["seconds"]) for r in rows if r.get("cache") == "miss"]
    hit_secs = [float(r["seconds"]) for r in rows if r.get("cache") == "hit"]
    all_secs = [float(r["seconds"]) for r in rows]
    has_cache = bool(miss_secs) or bool(hit_secs)
    t_miss = _mean(miss_secs) if has_cache else _mean(a_secs)
    t_hit = _mean(hit_secs) if hit_secs else None
    t_all = _mean(all_secs)
    delta = (t_miss - t_hit) if (t_miss is not None and t_hit is not None) else None
    return {
        "config": config,
        "a_n": len(a_secs),
        "a_mean_s": _mean(a_secs),
        "a_median_s": _median(a_secs),
        "has_edit_cache": has_cache,
        "n_miss": len(miss_secs),
        "n_hit": len(hit_secs),
        "t_miss_s": t_miss,
        "t_hit_s": t_hit,
        "t_all_s": t_all,
        "delta_miss_minus_hit_s": delta,
        "pct_save_on_hit": _pct(delta, t_miss),
        "pct_save_overall_vs_miss": _pct(
            (t_miss - t_all) if (has_cache and t_miss and t_all) else None, t_miss
        ),
    }


def _psnr_stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None}
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
    }


def analyze_psnr(
    config: str,
    cfg_dir: Path,
    rows: list[dict],
    fp32_dir: Path,
) -> dict:
    img_dir = cfg_dir / "edited_images" / config
    by_job = {r["job_id"]: r for r in rows}

    def collect(pred) -> list[float]:
        out = []
        for r in rows:
            if not pred(r):
                continue
            jid = r["job_id"]
            a = img_dir / f"{jid}.png"
            b = fp32_dir / f"{jid}.png"
            v = _psnr(a, b)
            if v is not None:
                out.append(v)
        return out

    prompt0 = collect(lambda r: str(r.get("prompt_idx", "")) == "0")
    full = collect(lambda _: True)
    miss = collect(lambda r: r.get("cache") == "miss")
    hit = collect(lambda r: r.get("cache") == "hit")
    # fp32 vs self
    if config == "baseline_fp32":
        prompt0 = collect(lambda r: str(r.get("prompt_idx", "")) == "0")
        full = collect(lambda _: True)

    return {
        "config": config,
        "psnr_prompt0": _psnr_stats(prompt0),
        "psnr_full": _psnr_stats(full),
        "psnr_miss": _psnr_stats(miss),
        "psnr_hit": _psnr_stats(hit),
        "n_fp32_refs": sum(1 for _ in fp32_dir.glob("*.png")) if fp32_dir.is_dir() else 0,
        "n_jobs_quality": len(by_job),
    }


def build_report(
    speed: list[dict],
    psnr: list[dict],
    metas: dict[str, dict],
    out_md: Path,
    out_csv: Path,
) -> None:
    hashes = {metas[s["config"]].get("jobs_hash") for s in speed if s["config"] in metas}
    seeds = {metas[s["config"]].get("eval_seed_base") for s in speed if s["config"] in metas}
    fp32_s = next((s for s in speed if s["config"] == "baseline_fp32"), None)
    fp32_a = fp32_s["a_mean_s"] if fp32_s else None
    fp32_all = fp32_s["t_all_s"] if fp32_s else None

    # CSV merge
    psnr_by = {p["config"]: p for p in psnr}
    fieldnames = [
        "config",
        "a_n",
        "a_mean_s",
        "speedup_vs_fp32_a",
        "t_miss_s",
        "t_hit_s",
        "t_all_s",
        "pct_save_on_hit",
        "pct_save_overall_vs_miss",
        "speedup_vs_fp32_all",
        "psnr_prompt0_n",
        "psnr_prompt0_mean",
        "psnr_prompt0_median",
        "psnr_prompt0_min",
        "psnr_full_n",
        "psnr_full_mean",
        "psnr_full_median",
        "psnr_full_min",
        "psnr_miss_mean",
        "psnr_hit_mean",
        "jobs_hash",
        "eval_seed_base",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in speed:
            p = psnr_by.get(s["config"], {})
            p0 = p.get("psnr_prompt0") or {}
            pf = p.get("psnr_full") or {}
            pm = p.get("psnr_miss") or {}
            ph = p.get("psnr_hit") or {}
            meta = metas.get(s["config"], {})
            sp_a = (fp32_a / s["a_mean_s"]) if (fp32_a and s["a_mean_s"]) else None
            sp_o = (fp32_all / s["t_all_s"]) if (fp32_all and s["t_all_s"]) else None
            row = {
                "config": s["config"],
                "a_n": s["a_n"],
                "a_mean_s": round(s["a_mean_s"], 6) if s["a_mean_s"] else None,
                "speedup_vs_fp32_a": round(sp_a, 6) if sp_a else None,
                "t_miss_s": round(s["t_miss_s"], 6) if s["t_miss_s"] else None,
                "t_hit_s": round(s["t_hit_s"], 6) if s["t_hit_s"] else None,
                "t_all_s": round(s["t_all_s"], 6) if s["t_all_s"] else None,
                "pct_save_on_hit": round(s["pct_save_on_hit"], 4) if s["pct_save_on_hit"] else None,
                "pct_save_overall_vs_miss": round(s["pct_save_overall_vs_miss"], 4)
                if s["pct_save_overall_vs_miss"]
                else None,
                "speedup_vs_fp32_all": round(sp_o, 6) if sp_o else None,
                "psnr_prompt0_n": p0.get("n"),
                "psnr_prompt0_mean": round(p0["mean"], 4) if p0.get("mean") is not None else None,
                "psnr_prompt0_median": round(p0["median"], 4)
                if p0.get("median") is not None
                else None,
                "psnr_prompt0_min": round(p0["min"], 4) if p0.get("min") is not None else None,
                "psnr_full_n": pf.get("n"),
                "psnr_full_mean": round(pf["mean"], 4) if pf.get("mean") is not None else None,
                "psnr_full_median": round(pf["median"], 4) if pf.get("median") is not None else None,
                "psnr_full_min": round(pf["min"], 4) if pf.get("min") is not None else None,
                "psnr_miss_mean": round(pm["mean"], 4) if pm.get("mean") is not None else None,
                "psnr_hit_mean": round(ph["mean"], 4) if ph.get("mean") is not None else None,
                "jobs_hash": meta.get("jobs_hash"),
                "eval_seed_base": meta.get("eval_seed_base"),
            }
            w.writerow(row)

    lines: list[str] = [
        "# Báo cáo so sánh đầy đủ test_data — tốc độ + PSNR vs FP32",
        "",
        f"- Bundles: **{len(speed)}**",
        f"- `jobs_hash`: `{', '.join(sorted(str(h) for h in hashes if h))}`",
        f"- `eval_seed_base`: `{', '.join(sorted(str(s) for s in seeds if s is not None))}`",
        "- Baseline chất lượng: ảnh `edited_images/baseline_fp32/{job_id}.png`",
        "- Báo cáo cache-only (cũ): [`CACHE_CONTRIBUTION_REPORT.md`](./CACHE_CONTRIBUTION_REPORT.md)",
        "",
        "---",
        "",
        "## A — Tốc độ không cache (fair, prompt_idx=0, n≈200)",
        "",
        "| Config | n | mean s/edit | median | vs fp32 |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in speed:
        sp = (fp32_a / s["a_mean_s"]) if (fp32_a and s["a_mean_s"]) else None
        lines.append(
            f"| `{s['config']}` | {s['a_n']} | {_fmt(s['a_mean_s'])} | {_fmt(s['a_median_s'])} | "
            f"{_fmt(sp, 3)}× |"
        )

    lines += [
        "",
        "## B — Tốc độ có EditCache",
        "",
        "| Config | n_miss | n_hit | t_miss | t_hit | t_all | % save hit | % save overall vs miss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in speed:
        if not s["has_edit_cache"]:
            lines.append(
                f"| `{s['config']}` | — | — | {_fmt(s['t_miss_s'])}* | — | {_fmt(s['t_all_s'])} | — | — |"
            )
            continue
        lines.append(
            f"| `{s['config']}` | {s['n_miss']} | {s['n_hit']} | {_fmt(s['t_miss_s'])} | "
            f"{_fmt(s['t_hit_s'])} | {_fmt(s['t_all_s'])} | {_fmt(s['pct_save_on_hit'], 2)}% | "
            f"{_fmt(s['pct_save_overall_vs_miss'], 2)}% |"
        )
    lines += [
        "",
        "> `*` fp32: proxy = mean prompt0 (cache off).",
        "",
        "### Speedup vs fp32 (overall 600)",
        "",
    ]
    for s in speed:
        sp = (fp32_all / s["t_all_s"]) if (fp32_all and s["t_all_s"]) else None
        lines.append(f"- `{s['config']}`: **{_fmt(sp, 3)}×**")

    lines += [
        "",
        "---",
        "",
        "## C — Độ sai khác vs FP32 gốc (PSNR ↑ tốt hơn)",
        "",
        "### C1 — Fair: chỉ prompt_idx=0 (n≈200)",
        "",
        "| Config | n | PSNR mean | median | min |",
        "|---|---:|---:|---:|---:|",
    ]
    for p in psnr:
        p0 = p["psnr_prompt0"]
        lines.append(
            f"| `{p['config']}` | {p0['n']} | {_fmt(p0['mean'], 2)} | {_fmt(p0['median'], 2)} | "
            f"{_fmt(p0['min'], 2)} |"
        )

    lines += [
        "",
        "### C2 — Full 600 job",
        "",
        "| Config | n | PSNR mean | median | min |",
        "|---|---:|---:|---:|---:|",
    ]
    for p in psnr:
        pf = p["psnr_full"]
        lines.append(
            f"| `{p['config']}` | {pf['n']} | {_fmt(pf['mean'], 2)} | {_fmt(pf['median'], 2)} | "
            f"{_fmt(pf['min'], 2)} |"
        )

    lines += [
        "",
        "### C3 — PSNR miss vs hit (config có EditCache)",
        "",
        "| Config | PSNR miss | PSNR hit |",
        "|---|---:|---:|",
    ]
    for p in psnr:
        if p["config"] == "baseline_fp32":
            continue
        pm, ph = p["psnr_miss"], p["psnr_hit"]
        if not pm["n"] and not ph["n"]:
            continue
        lines.append(
            f"| `{p['config']}` | {_fmt(pm['mean'], 2)} (n={pm['n']}) | {_fmt(ph['mean'], 2)} (n={ph['n']}) |"
        )

    # conclusions
    primary = next((s for s in speed if s["config"] == "fp16_disk"), None)
    p_fp16 = next((p for p in psnr if p["config"] == "fp16_disk"), None)
    p_fp4 = next((p for p in psnr if p["config"] == "improved_fp4_cache"), None)

    lines += ["", "## Kết luận ngắn", ""]
    if primary and fp32_s and p_fp16:
        lines += [
            f"- **Precision (cold / prompt0):** `{primary['config']}` "
            f"{_fmt(primary['a_mean_s'])} s vs fp32 {_fmt(fp32_a)} s → "
            f"**{_fmt((fp32_a / primary['a_mean_s']) if primary['a_mean_s'] else None, 3)}×**; "
            f"PSNR vs FP32 mean **{_fmt(p_fp16['psnr_prompt0']['mean'], 2)} dB**.",
            f"- **EditCache:** hit tiết kiệm **{_fmt(primary['pct_save_on_hit'], 1)}%** so miss; "
            f"overall 600 tiết kiệm **{_fmt(primary['pct_save_overall_vs_miss'], 1)}%** so thuần miss.",
        ]
    if p_fp4:
        lines.append(
            f"- **fp4:** PSNR vs FP32 (prompt0) mean **{_fmt(p_fp4['psnr_prompt0']['mean'], 2)} dB** "
            f"(so với fp16 nếu có ở trên)."
        )
    lines += [
        "",
        "## File phụ",
        "",
        f"- `{out_csv.name}`",
        "",
        "Sinh bởi: `python scripts/compare_test_data_full.py`",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="So sánh đầy đủ test_data: tốc độ + PSNR vs FP32")
    ap.add_argument("--zips-dir", type=Path, default=ROOT / "test_data")
    ap.add_argument("--extract-dir", type=Path, default=ROOT / "test_data" / "extracted")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--skip-psnr", action="store_true", help="Chỉ tốc độ (debug)")
    args = ap.parse_args()

    zdir = args.zips_dir.resolve()
    edir = args.extract_dir.resolve()
    edir.mkdir(parents=True, exist_ok=True)

    zips = sorted(zdir.glob("*.zip"))
    if not zips:
        raise SystemExit(f"Không có *.zip trong {zdir}")

    extracted: dict[str, Path] = {}
    print("=== Extract ===")
    for zp in zips:
        cfg = _config_from_zip_name(zp.name)
        if not cfg:
            print(f"  skip unknown zip: {zp.name}")
            continue
        out = edir / cfg
        extract_zip(zp, out, cfg, force=args.force_extract)
        extracted[cfg] = out

    if "baseline_fp32" not in extracted:
        raise SystemExit("Thiếu baseline_fp32 — không tính được PSNR vs FP32")

    fp32_img = extracted["baseline_fp32"] / "edited_images" / "baseline_fp32"
    if not fp32_img.is_dir() or _count_pngs(extracted["baseline_fp32"], "baseline_fp32") < 1:
        raise SystemExit(f"Thiếu ảnh FP32 tại {fp32_img}")

    # analyze
    speed_list: list[dict] = []
    psnr_list: list[dict] = []
    metas: dict[str, dict] = {}

    configs = [c for c in CONFIG_ORDER if c in extracted] + [
        c for c in extracted if c not in CONFIG_ORDER
    ]

    print("=== Speed ===")
    for cfg in configs:
        rows = load_quality(extracted[cfg])
        meta = load_meta(extracted[cfg])
        metas[cfg] = meta
        sp = analyze_speed(cfg, rows)
        speed_list.append(sp)
        print(f"  {cfg}: a_mean={_fmt(sp['a_mean_s'])} all={_fmt(sp['t_all_s'])}")

    if not args.skip_psnr:
        print("=== PSNR vs FP32 (có thể mất vài phút) ===")
        for cfg in configs:
            rows = load_quality(extracted[cfg])
            print(f"  computing {cfg} ...", flush=True)
            p = analyze_psnr(cfg, extracted[cfg], rows, fp32_img)
            psnr_list.append(p)
            print(
                f"    prompt0 mean={_fmt(p['psnr_prompt0']['mean'], 2)} "
                f"full mean={_fmt(p['psnr_full']['mean'], 2)} n={p['psnr_full']['n']}"
            )
    else:
        psnr_list = [{"config": c, "psnr_prompt0": {"n": 0}, "psnr_full": {"n": 0},
                      "psnr_miss": {"n": 0}, "psnr_hit": {"n": 0}} for c in configs]

    out_md = args.out_md or (zdir / "FULL_COMPARE_REPORT.md")
    out_csv = args.out_csv or (zdir / "full_compare_summary.csv")
    build_report(speed_list, psnr_list, metas, out_md, out_csv)
    print(f"Wrote {out_md}")
    print(f"Wrote {out_csv}")

    # Link from CACHE report
    cache_md = zdir / "CACHE_CONTRIBUTION_REPORT.md"
    if cache_md.is_file():
        text = cache_md.read_text(encoding="utf-8")
        link_line = (
            "\n> **Báo cáo đầy đủ (tốc độ + PSNR vs FP32):** "
            "[`FULL_COMPARE_REPORT.md`](./FULL_COMPARE_REPORT.md)\n"
        )
        if "FULL_COMPARE_REPORT.md" not in text:
            # insert after title
            lines = text.splitlines()
            if lines and lines[0].startswith("#"):
                lines.insert(1, link_line.rstrip("\n"))
                cache_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
            else:
                cache_md.write_text(link_line + text, encoding="utf-8")
            print(f"Linked from {cache_md.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
