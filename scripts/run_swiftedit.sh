#!/usr/bin/env bash
# Chạy demo SwiftEdit (Mac MPS / CUDA / CPU)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Chưa có .venv — chạy: pyenv local 3.12.10 && python -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements-mac.txt"
  exit 1
fi

if [[ ! -d SwiftEdit/swiftedit_weights/inverse_ckpt-120k ]]; then
  echo "Chưa có weights — chạy: bash scripts/download_swiftedit_weights.sh"
  exit 1
fi

source .venv/bin/activate
cd SwiftEdit
python infer.py
