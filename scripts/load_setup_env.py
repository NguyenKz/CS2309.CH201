#!/usr/bin/env python3
"""Đọc .setup_env (do setup_macos.sh / setup_colab.sh tạo) và trả về paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    start = start or Path.cwd()
    if start.name == "notebooks":
        return start.parent
    for p in [start, *start.parents]:
        if (p / "SwiftEdit" / "infer.py").is_file():
            return p
    raise FileNotFoundError(
        f"Không tìm thấy SwiftEdit/infer.py từ {start}. Chạy setup script trước."
    )


def load_setup_env(project_root: Path | None = None) -> dict[str, str]:
    root = project_root or find_project_root()
    env_file = root / ".setup_env"
    if not env_file.is_file():
        raise FileNotFoundError(
            f"Thiếu {env_file}. Chạy: bash scripts/setup_macos.sh hoặc setup_colab.sh"
        )

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        key, _, val = line[7:].partition("=")
        val = val.strip().strip('"').strip("'")
        os.environ[key] = val

    paths = {
        "PROJECT_ROOT": os.environ["PROJECT_ROOT"],
        "SWIFTEDIT_DIR": os.environ["SWIFTEDIT_DIR"],
        "WEIGHTS_DIR": os.environ["WEIGHTS_DIR"],
        "RESULTS_DIR": os.environ["RESULTS_DIR"],
        "PHASE3_DIR": os.environ["PHASE3_DIR"],
        "ABLATION_DIR": os.environ["ABLATION_DIR"],
        "METRICS_DIR": os.environ["METRICS_DIR"],
        "PIEBENCH_DIR": os.environ.get("PIEBENCH_DIR", ""),
        "PIEBENCH_RESULTS_DIR": os.environ.get("PIEBENCH_RESULTS_DIR", ""),
        "IN_COLAB": os.environ.get("IN_COLAB", "0"),
        "HF_HOME": os.environ.get("HF_HOME", ""),
    }

    swiftedit = Path(paths["SWIFTEDIT_DIR"])
    if str(swiftedit) not in sys.path:
        sys.path.insert(0, str(swiftedit))
    os.chdir(swiftedit)

    return paths


if __name__ == "__main__":
    p = load_setup_env()
    for k, v in p.items():
        print(f"{k}={v}")
