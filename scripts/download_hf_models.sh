#!/usr/bin/env bash
# Tải model Hugging Face tối thiểu cho SwiftEdit (không tải full repo ~30GB).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-900}"

python <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download

SD21 = "Manojb/stable-diffusion-2-1-base"
SD_TURBO = "stabilityai/sd-turbo"
AUX_PATTERNS = ["scheduler/*", "vae/*", "tokenizer/*", "text_encoder/*"]

print("1/3 SD 2.1 aux (scheduler, vae, tokenizer, text_encoder)...")
snapshot_download(SD21, allow_patterns=AUX_PATTERNS)

print("2/3 sd-turbo inversion (scheduler, vae, tokenizer, text_encoder)...")
snapshot_download(SD_TURBO, allow_patterns=AUX_PATTERNS)

print("3/3 IP-Adapter image_encoder...")
for f in (
    "models/image_encoder/config.json",
    "models/image_encoder/model.safetensors",
    "models/image_encoder/pytorch_model.bin",
):
    hf_hub_download("h94/IP-Adapter", f)

print("done.")
PY
