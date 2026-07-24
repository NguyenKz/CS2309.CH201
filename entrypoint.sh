#!/usr/bin/env bash
# Entrypoint — chạy SwiftEdit Gradio WebUI trên MacBook (MPS).
#
#   ./entrypoint.sh
#   DTYPE=fp32 ./entrypoint.sh
#   PORT=7861 SHARE=1 ./entrypoint.sh
#   ./entrypoint.sh --selftest assets/imgs_demo/dog.jpg   # (nếu có ảnh demo)
#
# Không dùng xFormers (CUDA). Mac = fp16 + channels_last + EditCache.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

DTYPE="${DTYPE:-fp16}"
PORT="${PORT:-7860}"
SHARE="${SHARE:-0}"
SETUP_IF_MISSING="${SETUP_IF_MISSING:-0}"

APP="$ROOT/scripts/app_gradio.py"
WEIGHTS="$ROOT/SwiftEdit/swiftedit_weights"
VENV="$ROOT/.venv"

usage() {
  cat <<'EOF'
Usage: ./entrypoint.sh [args...]

Env:
  DTYPE=fp16|fp32     (mặc định fp16)
  PORT=7860
  SHARE=0|1           tạo link Gradio public
  SETUP_IF_MISSING=1  nếu thiếu weights/.venv → chạy setup_macos.sh

Args còn lại chuyển thẳng sang app_gradio.py (vd: --selftest path.png).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$VENV/bin" ]]; then
  if [[ "$SETUP_IF_MISSING" == "1" ]]; then
    echo "[entrypoint] Chưa có .venv → bash scripts/setup_macos.sh"
    bash "$ROOT/scripts/setup_macos.sh"
  else
    echo "Chưa có .venv tại $VENV" >&2
    echo "  python -m venv .venv && source .venv/bin/activate" >&2
    echo "  pip install -r requirements-mac.txt" >&2
    echo "  # hoặc: SETUP_IF_MISSING=1 ./entrypoint.sh" >&2
    exit 1
  fi
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

if [[ ! -d "$WEIGHTS/inverse_ckpt-120k" ]]; then
  if [[ "$SETUP_IF_MISSING" == "1" ]]; then
    echo "[entrypoint] Chưa có weights → bash scripts/setup_macos.sh"
    bash "$ROOT/scripts/setup_macos.sh"
  else
    echo "Chưa có weights: $WEIGHTS/inverse_ckpt-120k" >&2
    echo "  bash scripts/download_swiftedit_weights.sh" >&2
    echo "  # hoặc: SETUP_IF_MISSING=1 ./entrypoint.sh" >&2
    exit 1
  fi
fi

if [[ ! -f "$APP" ]]; then
  echo "Thiếu $APP" >&2
  exit 1
fi

CMD=(python -u "$APP" --dtype "$DTYPE" --port "$PORT")
if [[ "$SHARE" == "1" || "$SHARE" == "true" || "$SHARE" == "yes" ]]; then
  CMD+=(--share)
fi
CMD+=("$@")

echo "[entrypoint] Mac WebUI — dtype=$DTYPE port=$PORT share=$SHARE"
echo "[entrypoint] Mở: http://127.0.0.1:${PORT}"
echo "[entrypoint] Lệnh: ${CMD[*]}"
echo "[entrypoint] Dừng: Ctrl+C"
echo

exec "${CMD[@]}"
