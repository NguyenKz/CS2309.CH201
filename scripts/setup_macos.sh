#!/usr/bin/env bash
# Setup SwiftEdit trên macOS (MPS) cho Giai đoạn 3.
# Chạy từ repo: bash scripts/setup_macos.sh
# Khuyến nghị: source .venv/bin/activate trước (hoặc script tự activate nếu có .venv)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIB="$ROOT/scripts/lib/setup_common.sh"
# shellcheck source=/dev/null
source "$LIB"

setup_log "macOS setup — $ROOT"

if [[ -d "$ROOT/.venv/bin" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
  setup_log "venv: $ROOT/.venv"
fi

export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-900}"

REQ_MAC="$ROOT/requirements-mac.txt"
REQ_P3="$ROOT/scripts/phase3_requirements.txt"
[[ -f "$REQ_MAC" ]] || {
  echo "Không thấy $REQ_MAC" >&2
  exit 1
}

setup_log "pip install..."
setup_pip_stack "$REQ_MAC" "$REQ_P3"

WEIGHTS="$ROOT/SwiftEdit/swiftedit_weights"
if ! setup_verify_weights "$WEIGHTS"; then
  setup_log "Tải swiftedit_weights (~9.6 GB)..."
  setup_download_weights "$ROOT"
  setup_verify_weights "$WEIGHTS"
fi

setup_log "Tải HF models (nếu thiếu cache)..."
setup_download_hf "$ROOT"

setup_write_env_file "$ROOT" 0 "${HF_HOME:-}"
setup_torch_report

setup_log "done — source $ROOT/.setup_env hoặc chạy notebook phase3 cell Setup"
