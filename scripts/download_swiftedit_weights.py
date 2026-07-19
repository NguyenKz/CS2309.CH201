#!/usr/bin/env python3
"""Tải swiftedit_weights từ GitHub Releases — tối ưu dung lượng tạm + log tiến độ.

Mặc định --stream: từng part curl/HTTP → pipe tar (không giữ .part/.tar.gz).
In tiến độ: [2/5] part-ab  45%  900/2000 MB  12.3 MB/s  ETA 89s
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0"
PARTS = ("aa", "ab", "ac", "ad", "ae")
# Kích thước thực tế release v1.0 (MB) — dùng làm mẫu số % khi server không gửi Content-Length
PART_SIZE_MB = {
    "aa": 2000,
    "ab": 2000,
    "ac": 2000,
    "ad": 2000,
    "ae": 1297,
}
TOTAL_MB = sum(PART_SIZE_MB.values())


def _tree_ok(weights: Path) -> bool:
    return (weights / "inverse_ckpt-120k").is_dir() and (weights / "sbv2_0.5").is_dir()


def _clean_archives(work: Path) -> None:
    for p in list(work.glob("swiftedit_weights.tar.gz*")):
        mb = p.stat().st_size // (1024**2)
        print(f"clean: rm {p.name} ({mb} MB)", flush=True)
        p.unlink(missing_ok=True)


def _free_gb(path: Path) -> float | None:
    try:
        u = shutil.disk_usage(path)
        return u.free / (1024**3)
    except Exception:
        return None


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "?s"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m{s % 60:02d}s"


class _Progress:
    def __init__(
        self,
        *,
        part_idx: int,
        n_parts: int,
        part: str,
        expected_mb: float,
        overall_done_mb: float,
    ):
        self.part_idx = part_idx
        self.n_parts = n_parts
        self.part = part
        self.expected = max(expected_mb * 1024 * 1024, 1.0)
        self.overall_done_mb = overall_done_mb
        self.bytes = 0
        self.t0 = time.perf_counter()
        self.t_last = self.t0
        self.b_last = 0

    def update(self, n: int, content_length: int | None = None) -> None:
        self.bytes += n
        now = time.perf_counter()
        # cập nhật mỗi ~0.4s hoặc khi xong
        if now - self.t_last < 0.4 and (
            content_length is None or self.bytes < content_length
        ):
            return
        total = float(content_length) if content_length else self.expected
        pct = min(100.0, 100.0 * self.bytes / total)
        dt = max(now - self.t_last, 1e-6)
        speed = (self.bytes - self.b_last) / dt / (1024 * 1024)
        avg = self.bytes / max(now - self.t0, 1e-6) / (1024 * 1024)
        remain = (total - self.bytes) / max(avg * 1024 * 1024, 1.0)
        done_mb = self.bytes / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        overall = self.overall_done_mb + done_mb
        overall_pct = min(100.0, 100.0 * overall / TOTAL_MB)
        print(
            f"\r[{self.part_idx}/{self.n_parts}] part-{self.part}  "
            f"{pct:5.1f}%  {done_mb:7.1f}/{total_mb:.0f} MB  "
            f"{speed:5.1f} MB/s  ETA {_fmt_eta(remain)}  | "
            f"tổng ~{overall_pct:4.1f}% ({overall:.0f}/{TOTAL_MB} MB)",
            end="",
            flush=True,
            file=sys.stderr,
        )
        self.t_last = now
        self.b_last = self.bytes

    def done(self) -> float:
        now = time.perf_counter()
        mb = self.bytes / (1024 * 1024)
        avg = mb / max(now - self.t0, 1e-6)
        print(
            f"\r[{self.part_idx}/{self.n_parts}] part-{self.part}  "
            f"100.0%  {mb:7.1f}/{mb:.0f} MB  "
            f"avg {avg:5.1f} MB/s  OK"
            + " " * 40,
            flush=True,
            file=sys.stderr,
        )
        return mb


def _http_stream(url: str, sink, progress: _Progress) -> None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CS2309-SwiftEdit-weights/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        cl = resp.headers.get("Content-Length")
        content_length = int(cl) if cl and cl.isdigit() else None
        while True:
            chunk = resp.read(1024 * 1024)  # 1 MiB
            if not chunk:
                break
            sink.write(chunk)
            progress.update(len(chunk), content_length)


def download_stream(work: Path) -> None:
    """HTTP từng part → stdin của tar (không ghi archive)."""
    free = _free_gb(work)
    print(
        f"stream download → tar extract (5 part, ~{TOTAL_MB} MB, không lưu .part/.tar.gz)",
        flush=True,
    )
    if free is not None:
        print(f"disk trống: ~{free:.1f} GB (cần ≥12 GB)", flush=True)
        if free < 12:
            print("Cảnh báo: dung lượng thấp.", file=sys.stderr, flush=True)

    tar = subprocess.Popen(
        ["tar", "zxf", "-"],
        stdin=subprocess.PIPE,
        cwd=work,
    )
    assert tar.stdin is not None
    overall = 0.0
    try:
        for i, part in enumerate(PARTS, start=1):
            url = f"{BASE}/swiftedit_weights.tar.gz.part-{part}"
            print(f"\n→ Bắt đầu part-{part} ({i}/{len(PARTS)})", flush=True)
            prog = _Progress(
                part_idx=i,
                n_parts=len(PARTS),
                part=part,
                expected_mb=PART_SIZE_MB[part],
                overall_done_mb=overall,
            )
            try:
                _http_stream(url, tar.stdin, prog)
            except urllib.error.URLError as e:
                tar.kill()
                raise RuntimeError(f"Lỗi tải part-{part}: {e}") from e
            overall += prog.done()
        tar.stdin.close()
        rc = tar.wait()
        if rc != 0:
            raise RuntimeError(f"tar extract failed (exit {rc})")
    except Exception:
        try:
            tar.kill()
        except Exception:
            pass
        raise
    print(f"\nExtract xong. Đã stream ~{overall:.0f} MB.", flush=True)


def download_parts_then_pipe(work: Path) -> None:
    """Tải từng .part (có resume) rồi cat|tar; log tiến độ từng part."""
    part_files: list[Path] = []
    overall = 0.0
    for i, part in enumerate(PARTS, start=1):
        fname = f"swiftedit_weights.tar.gz.part-{part}"
        dest = work / fname
        expected = PART_SIZE_MB[part] * 1024 * 1024
        if dest.exists() and dest.stat().st_size > expected * 0.98:
            mb = dest.stat().st_size / (1024 * 1024)
            print(
                f"[{i}/{len(PARTS)}] part-{part}  skip (đã có {mb:.0f} MB)",
                flush=True,
            )
            overall += mb
            part_files.append(dest)
            continue

        # resume: mở append nếu đã có một phần
        mode = "ab" if dest.exists() and dest.stat().st_size > 0 else "wb"
        already = dest.stat().st_size if mode == "ab" else 0
        url = f"{BASE}/{fname}"
        print(
            f"\n→ Bắt đầu part-{part} ({i}/{len(PARTS)})"
            + (f" resume từ {already // (1024*1024)} MB" if already else ""),
            flush=True,
        )
        prog = _Progress(
            part_idx=i,
            n_parts=len(PARTS),
            part=part,
            expected_mb=PART_SIZE_MB[part],
            overall_done_mb=overall,
        )
        prog.bytes = already
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "CS2309-SwiftEdit-weights/1.0"},
        )
        if already:
            req.add_header("Range", f"bytes={already}-")
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, mode) as out:
            cl = resp.headers.get("Content-Length")
            # với Range, Content-Length là phần còn lại
            content_length = (
                already + int(cl) if cl and cl.isdigit() else int(expected)
            )
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                prog.update(len(chunk), content_length)
        prog.done()
        overall = sum(
            (work / f"swiftedit_weights.tar.gz.part-{p}").stat().st_size
            for p in PARTS[:i]
            if (work / f"swiftedit_weights.tar.gz.part-{p}").exists()
        ) / (1024 * 1024)
        part_files.append(dest)

    print("\ncat parts | tar extract...", flush=True)
    cat = subprocess.Popen(
        ["cat", *[str(p) for p in part_files]],
        stdout=subprocess.PIPE,
        cwd=work,
    )
    tar = subprocess.run(["tar", "zxf", "-"], stdin=cat.stdout, cwd=work)
    if cat.stdout:
        cat.stdout.close()
    cat.wait()
    if tar.returncode != 0:
        raise RuntimeError("tar extract failed")
    for p in part_files:
        print(f"clean part {p.name}", flush=True)
        p.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="HTTP…|tar (mặc định) — tiết kiệm disk nhất",
    )
    parser.add_argument(
        "--parts",
        action="store_true",
        help="Tải .part rồi cat|tar (resume được nếu mạng đứt)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="Xóa leftover .part/.tar.gz trước và sau",
    )
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    work = root / "SwiftEdit"
    work.mkdir(parents=True, exist_ok=True)
    weights = work / "swiftedit_weights"

    if _tree_ok(weights):
        print("swiftedit_weights: đã có, bỏ qua tải.", flush=True)
        if args.clean and not args.no_clean:
            _clean_archives(work)
        return 0

    if args.clean and not args.no_clean:
        _clean_archives(work)

    t0 = time.perf_counter()
    try:
        if args.parts:
            download_parts_then_pipe(work)
        else:
            download_stream(work)
    except Exception as e:
        print(f"\nLỗi tải weights: {e}", file=sys.stderr, flush=True)
        print(
            "Thử: python scripts/download_swiftedit_weights.py --parts --clean",
            file=sys.stderr,
        )
        return 1

    if args.clean and not args.no_clean:
        _clean_archives(work)

    if not _tree_ok(weights):
        print("FAIL: sau extract vẫn thiếu layout.", file=sys.stderr)
        return 1

    du = subprocess.run(["du", "-sh", str(weights)], capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    print(f"done: {du.stdout.strip() or weights}  ({elapsed / 60:.1f} phút)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
