#!/usr/bin/env bash
# Bước 1 trên Colab: clone repo đề tài về /content (chạy TRƯỚC setup_colab.sh).
#   bash scripts/clone_colab_repo.sh
# Biến môi trường: REPO_SLUG, COLAB_REPO_DIR, REPO_URL (URL đầy đủ, kể cả token)
set -euo pipefail

REPO_SLUG="${REPO_SLUG:-NguyenKz/CS2309.CH201}"
COLAB_REPO_DIR="${COLAB_REPO_DIR:-/content/CS2309.CH201}"
REPO_URL="${REPO_URL:-https://github.com/${REPO_SLUG}.git}"

echo "[clone] Colab repo → $COLAB_REPO_DIR"

if [[ -f "$COLAB_REPO_DIR/SwiftEdit/infer.py" ]]; then
  echo "[clone] Đã có code — skip (giống test notebook; xóa $COLAB_REPO_DIR thủ công nếu muốn clone lại)"
  exit 0
fi

command -v git >/dev/null 2>&1 || {
  echo "Thiếu git" >&2
  exit 1
}

echo "[clone] Cloning → $COLAB_REPO_DIR"
git clone --depth 1 "$REPO_URL" "$COLAB_REPO_DIR"

if [[ ! -f "$COLAB_REPO_DIR/SwiftEdit/infer.py" ]]; then
  echo "Clone xong nhưng thiếu SwiftEdit/infer.py — push đủ repo lên GitHub." >&2
  exit 1
fi

echo "[clone] OK — $(ls -la "$COLAB_REPO_DIR" | head -5)"
echo "[clone] Tiếp theo: bash $COLAB_REPO_DIR/scripts/setup_colab.sh"
