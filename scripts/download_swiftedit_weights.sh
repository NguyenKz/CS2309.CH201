#!/usr/bin/env bash
# Tải checkpoint SwiftEdit v1.0 (5 phần) vào SwiftEdit/swiftedit_weights/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/SwiftEdit"
BASE="https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0"
for part in aa ab ac ad ae; do
  f="swiftedit_weights.tar.gz.part-${part}"
  if [[ -f "$f" ]]; then
    echo "skip $f (exists)"
  else
    echo "download $f ..."
    curl -fL --retry 10 --retry-delay 5 --connect-timeout 60 -C - -o "$f" "${BASE}/${f}"
  fi
done
cat swiftedit_weights.tar.gz.part-* > swiftedit_weights.tar.gz
tar zxf swiftedit_weights.tar.gz
echo "done: $(du -sh swiftedit_weights | cut -f1)"
