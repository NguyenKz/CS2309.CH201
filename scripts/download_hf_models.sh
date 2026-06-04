#!/usr/bin/env bash
# Tải model Hugging Face cần cho SwiftEdit (lần đầu chạy infer).
# Repo gốc stabilityai/stable-diffusion-2-1-base có thể 401 → dùng mirror trong models.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -d .venv ]]; then
  source .venv/bin/activate
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  :
else
  echo "Dùng python hệ thống (Colab / chưa có .venv)."
fi
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-900}"

python <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download

print("1/3 Manojb/stable-diffusion-2-1-base (mirror SD 2.1)...")
snapshot_download("Manojb/stable-diffusion-2-1-base")

print("2/3 stabilityai/sd-turbo (inversion)...")
snapshot_download("stabilityai/sd-turbo")

print("3/3 h94/IP-Adapter — image_encoder...")
for f in (
    "models/image_encoder/config.json",
    "models/image_encoder/model.safetensors",
    "models/image_encoder/pytorch_model.bin",
):
    hf_hub_download("h94/IP-Adapter", f)

print("done.")
PY
