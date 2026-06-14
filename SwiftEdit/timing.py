"""Đo timing từng công đoạn của SwiftEdit và ghi ra log file (JSONL).

Bật/tắt và đổi đường dẫn qua biến môi trường:
    SWIFTEDIT_TIMING=0           -> tắt hoàn toàn (no-op)
    SWIFTEDIT_TIMING_LOG=<path>  -> đổi file log (mặc định: results/timing.log)

Mỗi lần edit ghi 1 dòng JSON gồm thời gian (ms) của từng stage + total.
Đọc nhanh không cần xem cả terminal:
    tail -n 5 results/timing.log
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch


def _sync(device) -> None:
    """Đồng bộ GPU để đo wall-time chính xác (CUDA/MPS chạy bất đồng bộ)."""
    d = str(device)
    if d.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif d.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _enabled() -> bool:
    return os.environ.get("SWIFTEDIT_TIMING", "1") not in ("0", "false", "False")


def default_log_path() -> Path:
    env = os.environ.get("SWIFTEDIT_TIMING_LOG")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "results" / "timing.log"


class StageTimer:
    """Thu thập thời gian từng stage; dump 1 dòng JSONL khi xong."""

    def __init__(self, device, label: str = "", log_path: Path | None = None):
        self.device = device
        self.label = label
        self.enabled = _enabled()
        self.log_path = Path(log_path) if log_path else default_log_path()
        self.stages: dict[str, float] = {}
        self._t_start = time.perf_counter() if self.enabled else 0.0

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        _sync(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.device)
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - t0)

    def merge(self, other: "StageTimer | None", prefix: str = "") -> None:
        """Gộp các stage từ timer con (vd: gen_img) vào timer cha."""
        if not (self.enabled and other and other.stages):
            return
        for k, v in other.stages.items():
            self.stages[f"{prefix}{k}"] = v

    def dump(self, extra: dict | None = None) -> dict | None:
        if not self.enabled:
            return None
        _sync(self.device)
        total_ms = round(1000 * (time.perf_counter() - self._t_start), 2)
        stages_ms = {k: round(1000 * v, 2) for k, v in self.stages.items()}
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "label": self.label,
            "device": str(self.device),
            "total_ms": total_ms,
            "stages_ms": stages_ms,
        }
        if extra:
            record.update(extra)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
