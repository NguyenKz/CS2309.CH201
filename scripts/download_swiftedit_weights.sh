#!/usr/bin/env bash
# Tải checkpoint SwiftEdit — curl nhanh (như ban đầu), không giữ .tar.gz trung gian.
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

clean_archives

# Python wrapper = cùng curl + log [i/5]
exec python3 "$ROOT/scripts/download_swiftedit_weights.py" --clean
