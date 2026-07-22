#!/usr/bin/env python3
"""So sánh đóng góp EditCache từ bundle precision trong test_data/.

Hai kịch bản:
  A — Không cache (fair): chỉ prompt_idx=0 (~200 job).
      fp32: cache=off; config có EditCache: cache=miss.
  B — Có cache: t_miss / t_hit / t_all + % tiết kiệm.

Chỉ đọc quality.csv + run_meta.json từ zip (không giải nén PNG).

Ví dụ:
  python scripts/compare_cache_contribution.py --zips-dir test_data
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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


def _read_zip_member(zf: zipfile.ZipFile, name: str) -> bytes | None:
    # member có thể ở root hoặc trong subdir
    candidates = [n for n in zf.namelist() if n.endswith(name) and not n.endswith("/")]
    if not candidates:
        return None
    # ưu tiên path ngắn nhất (thường là root)
    candidates.sort(key=len)
    return zf.read(candidates[0])


def load_bundle(path: Path) -> dict:
    """Load quality rows + meta từ .zip hoặc thư mục đã giải nén."""
    if path.is_file() and path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            meta_b = _read_zip_member(zf, "run_meta.json")
            qual_b = _read_zip_member(zf, "quality.csv")
            if not meta_b or not qual_b:
                raise FileNotFoundError(f"Thiếu run_meta.json/quality.csv trong {path.name}")
            meta = json.loads(meta_b.decode("utf-8"))
            rows = list(csv.DictReader(io.StringIO(qual_b.decode("utf-8"))))
    elif path.is_dir():
        meta_p = next(path.rglob("run_meta.json"), None)
        qual_p = next(path.rglob("quality.csv"), None)
        if not meta_p or not qual_p:
            raise FileNotFoundError(f"Thiếu run_meta.json/quality.csv trong {path}")
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        with qual_p.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        raise FileNotFoundError(path)

    configs = meta.get("configs") or []
    config = configs[0] if configs else (rows[0]["config"] if rows else "unknown")
    return {
        "path": path,
        "meta": meta,
        "rows": rows,
        "config": config,
        "jobs_hash": meta.get("jobs_hash"),
        "eval_seed_base": meta.get("eval_seed_base"),
        "n_jobs": meta.get("n_jobs") or len(rows),
    }


def _seconds(row: dict) -> float:
    return float(row["seconds"])


def analyze_bundle(bundle: dict) -> dict:
    rows = bundle["rows"]
    config = bundle["config"]

    # A: fair no-cache — prompt_idx == 0
    # fp32: off; others with cache: miss (also require prompt_idx 0)
    a_rows = []
    for r in rows:
        if str(r.get("prompt_idx", "")) != "0":
            continue
        c = r.get("cache", "")
        if c == "off" or c == "miss":
            a_rows.append(r)

    a_secs = [_seconds(r) for r in a_rows]

    miss_rows = [r for r in rows if r.get("cache") == "miss"]
    hit_rows = [r for r in rows if r.get("cache") == "hit"]
    off_rows = [r for r in rows if r.get("cache") == "off"]
    all_secs = [_seconds(r) for r in rows]
    miss_secs = [_seconds(r) for r in miss_rows]
    hit_secs = [_seconds(r) for r in hit_rows]
    off_secs = [_seconds(r) for r in off_rows]

    # Với fp32 (cache=off): t_miss proxy = prompt0 / all off
    has_edit_cache = bool(miss_secs) or bool(hit_secs)
    t_miss = _mean(miss_secs) if has_edit_cache else _mean(a_secs)
    t_hit = _mean(hit_secs) if hit_secs else None
    t_all = _mean(all_secs)
    t_a = _mean(a_secs)
    t_a_med = _median(a_secs)

    delta_hit = (t_miss - t_hit) if (t_miss is not None and t_hit is not None) else None
    pct_hit = _pct(delta_hit, t_miss) if delta_hit is not None else None
    delta_all = (t_miss - t_all) if (t_miss is not None and t_all is not None and has_edit_cache) else None
    pct_all = _pct(delta_all, t_miss) if delta_all is not None else None

    return {
        "config": config,
        "source": bundle["path"].name,
        "jobs_hash": bundle["jobs_hash"],
        "eval_seed_base": bundle["eval_seed_base"],
        "n_jobs": len(rows),
        "has_edit_cache": has_edit_cache,
        # A
        "a_n": len(a_secs),
        "a_mean_s": t_a,
        "a_median_s": t_a_med,
        # B
        "n_miss": len(miss_secs),
        "n_hit": len(hit_secs),
        "n_off": len(off_secs),
        "t_miss_s": t_miss,
        "t_hit_s": t_hit,
        "t_all_s": t_all,
        "t_off_s": _mean(off_secs) if off_secs else None,
        "delta_miss_minus_hit_s": delta_hit,
        "pct_save_on_hit": pct_hit,
        "delta_miss_minus_all_s": delta_all,
        "pct_save_overall_vs_miss": pct_all,
    }


def build_report(stats: list[dict], out_md: Path, out_csv: Path) -> None:
    hashes = {s["jobs_hash"] for s in stats}
    seeds = {s["eval_seed_base"] for s in stats}
    warn_hash = len(hashes) > 1
    warn_seed = len(seeds) > 1

    fp32 = next((s for s in stats if s["config"] == "baseline_fp32"), None)
    fp32_a = fp32["a_mean_s"] if fp32 else None
    fp32_all = fp32["t_all_s"] if fp32 else None

    # CSV
    fieldnames = [
        "config",
        "source",
        "jobs_hash",
        "eval_seed_base",
        "has_edit_cache",
        "a_n",
        "a_mean_s",
        "a_median_s",
        "speedup_vs_fp32_a",
        "n_miss",
        "n_hit",
        "n_off",
        "t_miss_s",
        "t_hit_s",
        "t_all_s",
        "delta_miss_minus_hit_s",
        "pct_save_on_hit",
        "delta_miss_minus_all_s",
        "pct_save_overall_vs_miss",
        "speedup_vs_fp32_all",
        "speedup_vs_fp32_miss",
    ]
    csv_rows = []
    for s in stats:
        sp_a = (fp32_a / s["a_mean_s"]) if (fp32_a and s["a_mean_s"]) else None
        sp_all = (fp32_all / s["t_all_s"]) if (fp32_all and s["t_all_s"]) else None
        sp_miss = (fp32_a / s["t_miss_s"]) if (fp32_a and s["t_miss_s"]) else None
        csv_rows.append(
            {
                **{k: s.get(k) for k in fieldnames if k in s},
                "speedup_vs_fp32_a": sp_a,
                "speedup_vs_fp32_all": sp_all,
                "speedup_vs_fp32_miss": sp_miss,
            }
        )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in csv_rows:
            # round floats for readability
            out = {}
            for k, v in r.items():
                if isinstance(v, float):
                    out[k] = round(v, 6)
                else:
                    out[k] = v
            w.writerow(out)

    lines: list[str] = []
    lines += [
        "# Báo cáo đóng góp EditCache — test_data",
        "",
        "> Chỉ so **tốc độ** từ `quality.csv`. PSNR cross-run chưa tính (cần giải nén `edited_images`).",
        "",
        "## Metadata",
        "",
        f"- Số bundle: **{len(stats)}**",
        f"- `jobs_hash`: `{', '.join(sorted(str(h) for h in hashes))}`"
        + (" — **CẢNH BÁO: khác nhau giữa bundle**" if warn_hash else " (đồng nhất)"),
        f"- `eval_seed_base`: `{', '.join(sorted(str(s) for s in seeds))}`"
        + (" — **CẢNH BÁO: khác seed**" if warn_seed else " (đồng nhất)"),
        "",
        "## Quy ước",
        "",
        "- Mỗi ảnh 3 prompt (`prompt_idx` 0/1/2).",
        "- Config có EditCache: `_0` = miss, `_1`/`_2` = hit.",
        "- `baseline_fp32`: cache luôn `off` (không EditCache).",
        "",
        "---",
        "",
        "## A — Không cache (fair, ~200 case)",
        "",
        "Chỉ `prompt_idx=0`: fp32 = `cache=off`; config khác = `cache=miss`.",
        "So precision **không lẫn** lợi ích cache.",
        "",
        "| Config | n | mean s/edit | median | vs fp32 (prompt0) |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in stats:
        sp = (fp32_a / s["a_mean_s"]) if (fp32_a and s["a_mean_s"]) else None
        lines.append(
            f"| `{s['config']}` | {s['a_n']} | {_fmt(s['a_mean_s'])} | {_fmt(s['a_median_s'])} | "
            f"{_fmt(sp, 3)}× |"
        )

    lines += [
        "",
        "---",
        "",
        "## B — Có cache (đóng góp EditCache)",
        "",
        "| Config | n_miss | n_hit | t_miss | t_hit | t_all | Δ miss−hit | % save hit | % save overall vs miss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in stats:
        if not s["has_edit_cache"]:
            lines.append(
                f"| `{s['config']}` | — | — | {_fmt(s['t_miss_s'])}* | — | {_fmt(s['t_all_s'])} | — | — | — |"
            )
            continue
        lines.append(
            f"| `{s['config']}` | {s['n_miss']} | {s['n_hit']} | {_fmt(s['t_miss_s'])} | "
            f"{_fmt(s['t_hit_s'])} | {_fmt(s['t_all_s'])} | {_fmt(s['delta_miss_minus_hit_s'])} | "
            f"{_fmt(s['pct_save_on_hit'], 2)}% | {_fmt(s['pct_save_overall_vs_miss'], 2)}% |"
        )
    lines += [
        "",
        "> `*` Với `baseline_fp32`: t_miss proxy = mean prompt0 (cache off).",
        "",
        "### Speedup vs fp32",
        "",
        "| Config | miss vs fp32 prompt0 | overall 600 vs fp32 600 |",
        "|---|---:|---:|",
    ]
    for s in stats:
        sp_m = (fp32_a / s["t_miss_s"]) if (fp32_a and s["t_miss_s"]) else None
        sp_o = (fp32_all / s["t_all_s"]) if (fp32_all and s["t_all_s"]) else None
        lines.append(f"| `{s['config']}` | {_fmt(sp_m, 3)}× | {_fmt(sp_o, 3)}× |")

    # Narrative cache contribution — pick primary fp16_disk if present
    primary = next((s for s in stats if s["config"] == "fp16_disk"), None)
    if primary is None:
        primary = next((s for s in stats if s["has_edit_cache"]), None)

    lines += ["", "## Kết luận ngắn", ""]
    if primary and primary["has_edit_cache"] and fp32:
        lines += [
            f"- Config tham chiếu cache: `{primary['config']}`.",
            f"- Cold (miss / prompt0): **{_fmt(primary['t_miss_s'])} s** vs fp32 prompt0 **{_fmt(fp32_a)} s** "
            f"→ **{_fmt((fp32_a / primary['t_miss_s']) if primary['t_miss_s'] else None, 3)}×** "
            f"(lợi từ precision, chưa tính cache).",
            f"- Hit: **{_fmt(primary['t_hit_s'])} s** — tiết kiệm **{_fmt(primary['delta_miss_minus_hit_s'])} s/edit** "
            f"(**{_fmt(primary['pct_save_on_hit'], 1)}%** so miss).",
            f"- Overall 600 (1 miss + 2 hit/ảnh): **{_fmt(primary['t_all_s'])} s** — "
            f"tiết kiệm **{_fmt(primary['pct_save_overall_vs_miss'], 1)}%** so với chạy thuần miss; "
            f"vs fp32 overall **{_fmt((fp32_all / primary['t_all_s']) if (fp32_all and primary['t_all_s']) else None, 3)}×**.",
            "",
            "→ **EditCache** đóng góp thêm trên hit (cùng ảnh, đổi prompt); "
            "so precision fair phải dùng kịch bản A (prompt0 / miss only).",
        ]
    else:
        lines.append("- Không đủ config có EditCache hoặc thiếu baseline_fp32.")

    lines += [
        "",
        "## Nguồn bundle",
        "",
        "| Config | File |",
        "|---|---|",
    ]
    for s in stats:
        lines.append(f"| `{s['config']}` | `{s['source']}` |")

    lines += [
        "",
        "## File phụ",
        "",
        f"- CSV: `{out_csv.name}`",
        "",
        "Sinh bởi: `python scripts/compare_cache_contribution.py`",
        "",
    ]

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="So sánh đóng góp EditCache từ test_data bundles")
    ap.add_argument(
        "--zips-dir",
        type=Path,
        default=ROOT / "test_data",
        help="Thư mục chứa *.zip (hoặc thư mục đã extract)",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Mặc định: <zips-dir>/CACHE_CONTRIBUTION_REPORT.md",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Mặc định: <zips-dir>/cache_contribution_summary.csv",
    )
    args = ap.parse_args()
    zdir = args.zips_dir.resolve()
    if not zdir.is_dir():
        raise SystemExit(f"Không thấy thư mục: {zdir}")

    sources: list[Path] = sorted(zdir.glob("*.zip"))
    # cũng nhận thư mục con có run_meta (trừ _peek nếu muốn — vẫn OK)
    for d in sorted(zdir.iterdir()):
        if d.is_dir() and d.name.startswith("_"):
            continue
        if d.is_dir() and (d / "run_meta.json").is_file():
            sources.append(d)
        elif d.is_dir() and any(d.rglob("run_meta.json")):
            # skip nested peek duplicates if zip already covers
            pass

    if not sources:
        raise SystemExit(f"Không có *.zip trong {zdir}")

    bundles = []
    for p in sources:
        try:
            bundles.append(load_bundle(p))
            print(f"OK {p.name} → {bundles[-1]['config']} n={len(bundles[-1]['rows'])}")
        except Exception as e:
            print(f"SKIP {p.name}: {e}")

    if not bundles:
        raise SystemExit("Không load được bundle nào")

    # dedupe by config — giữ zip mới hơn (tên có timestamp) nếu trùng
    by_cfg: dict[str, dict] = {}
    for b in bundles:
        cfg = b["config"]
        if cfg not in by_cfg or b["path"].name > by_cfg[cfg]["path"].name:
            by_cfg[cfg] = b
    bundles = list(by_cfg.values())
    # order: fp32 first, then others
    order = [
        "baseline_fp32",
        "improved_fp16_cache",
        "fp16_disk",
        "fp16_disk_xformers",
        "improved_fp4_cache",
    ]
    bundles.sort(
        key=lambda b: (order.index(b["config"]) if b["config"] in order else 99, b["config"])
    )

    stats = [analyze_bundle(b) for b in bundles]
    out_md = args.out_md or (zdir / "CACHE_CONTRIBUTION_REPORT.md")
    out_csv = args.out_csv or (zdir / "cache_contribution_summary.csv")
    build_report(stats, out_md, out_csv)
    print(f"Wrote {out_md}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
