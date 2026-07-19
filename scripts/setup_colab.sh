#!/usr/bin/env bash
# Bước 2 trên Colab: pip + weights + HF (chạy SAU clone_colab_repo.sh).
#   cd /content/CS2309.CH201 && bash scripts/setup_colab.sh
set -euo pipefail

LIB_DIR="$(cd "$(dirname "$0")/lib" && pwd)"
# shellcheck source=/dev/null
source "$LIB_DIR/setup_common.sh"

COLAB_REPO_DIR="${COLAB_REPO_DIR:-/content/CS2309.CH201}"

setup_log "Colab setup (bước 2 — cần clone trước)"
setup_check_colab_gpu

if [[ ! -f "$COLAB_REPO_DIR/SwiftEdit/infer.py" ]]; then
  echo "Chưa có code tại $COLAB_REPO_DIR." >&2
  echo "Chạy trước: bash scripts/clone_colab_repo.sh" >&2
  echo "Hoặc cell「Colab — Clone repo」trong notebook phase3." >&2
  exit 1
fi

ROOT="$COLAB_REPO_DIR"
cd "$ROOT"

export HF_HOME="${HF_HOME:-/content/huggingface}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-900}"
mkdir -p "$HF_HOME"

REQ="$ROOT/SwiftEdit/requirements.txt"
REQ_P3="$ROOT/scripts/phase3_requirements.txt"
[[ -f "$REQ" ]] || {
  echo "Không thấy $REQ — push đủ SwiftEdit/ lên GitHub." >&2
  exit 1
}

setup_log "pip install..."
setup_pip_stack "$REQ" "$REQ_P3"

WEIGHTS="$ROOT/SwiftEdit/swiftedit_weights"
# Precision notebook: SWIFTEDIT_SKIP_WEIGHTS_DOWNLOAD=1 → Drive/prepare gắn weights,
# tránh Qualcomm ~10GB+ khi chỉ chạy *_weight.
if [[ "${SWIFTEDIT_SKIP_WEIGHTS_DOWNLOAD:-0}" == "1" ]]; then
  setup_log "SKIP tải Qualcomm (SWIFTEDIT_SKIP_WEIGHTS_DOWNLOAD=1) — dùng prepare_colab_weights / Drive"
  # Dọn leftover archive nếu còn từ lần cũ
  rm -f "$ROOT/SwiftEdit/swiftedit_weights.tar.gz" \
        "$ROOT/SwiftEdit/swiftedit_weights.tar.gz.part-"* 2>/dev/null || true
elif ! setup_verify_weights "$WEIGHTS"; then
  setup_log "Tải swiftedit_weights (stream, ~10GB extract — không giữ .part/.tar)..."
  setup_download_weights "$ROOT"
  setup_verify_weights "$WEIGHTS" || true
fi

setup_log "Tải HF models..."
setup_download_hf "$ROOT"

setup_write_env_file "$ROOT" 1 "$HF_HOME"
setup_torch_report

setup_log "done — PROJECT_ROOT=$ROOT"
