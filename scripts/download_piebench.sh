#!/usr/bin/env bash
# Chuẩn bị PIE-Bench (Prompt-driven Image Editing Benchmark).
#
# Dataset KHÔNG có trên git — tải qua Google Form (PnP Inversion / ICLR 2024):
#   https://forms.gle/hVMkTABb4uvZVjme9
#
# Cách dùng:
#   bash scripts/download_piebench.sh                    # kiểm tra thư mục mặc định
#   bash scripts/download_piebench.sh /path/to/PIE.zip   # giải nén zip đã tải
#
# Biến môi trường: PROJECT_ROOT, PIEBENCH_DIR, PIEBENCH_ZIP
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PIEBENCH_DIR="${PIEBENCH_DIR:-$ROOT/data/PIE-Bench}"
ZIP_PATH="${1:-${PIEBENCH_ZIP:-}}"

echo "[piebench] PROJECT_ROOT=$ROOT"
echo "[piebench] PIEBENCH_DIR=$PIEBENCH_DIR"

_has_data() {
  [[ -f "$1/mapping_file.json" && -d "$1/annotation_images" ]]
}

if _has_data "$PIEBENCH_DIR"; then
  echo "[piebench] OK — đã có data tại $PIEBENCH_DIR"
  echo "  mapping: $(wc -l < "$PIEBENCH_DIR/mapping_file.json" 2>/dev/null || echo '?') lines"
  echo "  images:  $(find "$PIEBENCH_DIR/annotation_images" -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ') jpg"
  exit 0
fi

# Thử tìm trong data/ sau khi giải nén lỏng
for sub in "$ROOT/data" "$ROOT/data/PIE-Bench_v1" "$PIEBENCH_DIR"; do
  if _has_data "$sub"; then
    echo "[piebench] Tìm thấy tại $sub"
    exit 0
  fi
  for inner in "$sub"/*; do
    if [[ -d "$inner" ]] && _has_data "$inner"; then
      echo "[piebench] Tìm thấy tại $inner"
      echo "[piebench] Gợi ý: export PIEBENCH_DIR=$inner"
      exit 0
    fi
  done
done

if [[ -z "$ZIP_PATH" || ! -f "$ZIP_PATH" ]]; then
  echo "[piebench] Chưa có PIE-Bench đầy đủ (700 mẫu)." >&2
  echo "" >&2
  echo "Cách 1 — Dataset chính thức:" >&2
  echo "  1. Form: https://forms.gle/hVMkTABb4uvZVjme9" >&2
  echo "  2. Giải nén vào: $PIEBENCH_DIR" >&2
  echo "  3. Hoặc: bash scripts/download_piebench.sh /path/to/downloaded.zip" >&2
  echo "" >&2
  echo "Cách 2 — Test pipeline ngay (2 ảnh demo):" >&2
  echo "  python scripts/create_piebench_smoke.py" >&2
  echo "  python scripts/run_piebench_eval.py --piebench-dir data/PIE-Bench-smoke --max-samples 2" >&2
  echo "" >&2
  echo "Repo: https://github.com/cure-lab/PnPInversion" >&2
  exit 1
fi

command -v unzip >/dev/null 2>&1 || {
  echo "Thiếu unzip" >&2
  exit 1
}

mkdir -p "$(dirname "$PIEBENCH_DIR")"
TMP="${PIEBENCH_DIR}.extract.$$"
rm -rf "$TMP"
mkdir -p "$TMP"
echo "[piebench] Giải nén $ZIP_PATH → $TMP ..."
unzip -q "$ZIP_PATH" -d "$TMP"

FOUND=""
for candidate in "$TMP" "$TMP/data" "$TMP/PIE-Bench" "$TMP/PIE-Bench_v1"; do
  if _has_data "$candidate"; then
    FOUND="$candidate"
    break
  fi
done

if [[ -z "$FOUND" ]]; then
  for inner in "$TMP"/* "$TMP"/*/*; do
    if [[ -d "$inner" ]] && _has_data "$inner"; then
      FOUND="$inner"
      break
    fi
  done
fi

if [[ -z "$FOUND" ]]; then
  echo "Giải nén xong nhưng không thấy mapping_file.json + annotation_images/" >&2
  exit 1
fi

rm -rf "$PIEBENCH_DIR"
mv "$FOUND" "$PIEBENCH_DIR"
rm -rf "$TMP"
echo "[piebench] OK → $PIEBENCH_DIR"
