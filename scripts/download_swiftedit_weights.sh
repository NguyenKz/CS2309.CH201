#!/usr/bin/env bash
# Tải checkpoint SwiftEdit v1.0 — stream vào tar, không giữ .part + .tar.gz.
# Fallback resume: SWIFTEDIT_WEIGHTS_PARTS=1 bash scripts/download_swiftedit_weights.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/SwiftEdit"

clean_archives() {
  rm -f swiftedit_weights.tar.gz
  rm -f swiftedit_weights.tar.gz.part-*
}

if [[ -d swiftedit_weights/inverse_ckpt-120k && -d swiftedit_weights/sbv2_0.5 ]]; then
  echo "swiftedit_weights: đã có, bỏ qua."
  clean_archives
  exit 0
fi

# Xóa leftover cũ (nguyên nhân “kéo 50GB rồi xóa 40GB”)
clean_archives

BASE="https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0"
CURL=(curl -fL --retry 10 --retry-delay 5 --connect-timeout 60)

# Ủy thác cho Python (log tiến độ % / MB/s / ETA rõ hơn bash)
if [[ "${SWIFTEDIT_WEIGHTS_NATIVE_BASH:-0}" != "1" ]]; then
  exec python3 "$ROOT/scripts/download_swiftedit_weights.py" \
    $([ "${SWIFTEDIT_WEIGHTS_PARTS:-0}" = "1" ] && echo --parts) --clean
fi

if [[ "${SWIFTEDIT_WEIGHTS_PARTS:-0}" == "1" ]]; then
  echo "mode=parts (resume)..."
  i=0
  for part in aa ab ac ad ae; do
    i=$((i + 1))
    f="swiftedit_weights.tar.gz.part-${part}"
    sz=0
    if [[ -f "$f" ]]; then
      sz=$(wc -c <"$f" | tr -d ' ')
    fi
    if [[ "$sz" -gt 1000000 ]]; then
      echo "[$i/5] part-$part skip ($(echo "$sz" | awk '{printf "%.0f", $1/1024/1024}') MB)"
    else
      echo "[$i/5] download $f ..."
      "${CURL[@]}" -C - --progress-bar -o "$f" "${BASE}/${f}"
      echo
    fi
  done
  echo "cat | tar (không tạo .tar.gz)..."
  cat swiftedit_weights.tar.gz.part-* | tar zxf -
  clean_archives
else
  echo "mode=stream (curl…|tar) — không lưu archive..."
  i=0
  (
    for part in aa ab ac ad ae; do
      i=$((i + 1))
      f="swiftedit_weights.tar.gz.part-${part}"
      echo "[$i/5] stream $f ..." >&2
      "${CURL[@]}" --progress-bar "${BASE}/${f}"
      echo >&2
    done
  ) | tar zxf -
  clean_archives
fi

echo "done: $(du -sh swiftedit_weights | cut -f1)"

