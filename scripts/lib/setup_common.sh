# shellcheck shell=bash
# Hàm dùng chung cho setup_macos.sh và setup_colab.sh

setup_log() { echo "[setup] $*"; }

setup_require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "Thiếu lệnh: $cmd" >&2
    exit 1
  }
}

setup_check_colab_gpu() {
  setup_require_cmd nvidia-smi
  if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q .; then
    echo "Colab chưa có GPU. Extension: New Colab Server → GPU → T4." >&2
    exit 1
  fi
  setup_log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
}

setup_pip_stack() {
  local req_primary="$1"
  local req_phase3="$2"
  setup_require_cmd python3
  python3 -m pip install -q -U pip
  python3 -m pip install -q -U -r "$req_primary"
  if [[ -f "$req_phase3" ]]; then
    python3 -m pip install -q -U -r "$req_phase3"
  fi
  python3 <<'PY'
import importlib.metadata as im
from packaging.version import Version

for pkg, min_ver in (("transformers", "4.46.0"), ("diffusers", "0.32.0")):
    got = im.version(pkg)
    print(f"  {pkg}: {got} (>= {min_ver})")
    if Version(got) < Version(min_ver):
        raise SystemExit(f"Sai {pkg}=={got}, can >= {min_ver}")

import diffusers  # noqa: F401
from transformers import EncoderDecoderCache  # noqa: F401

import numpy as np
import torchvision

print(f"  numpy: {np.__version__}")
print(f"  torchvision: {torchvision.__version__}")
print("  diffusers + transformers OK")
PY
}

setup_verify_weights() {
  local weights_dir="$1"
  local missing=0
  for name in inverse_ckpt-120k sbv2_0.5 ip_adapter_ckpt-90k/ip_adapter.bin; do
    if [[ ! -e "$weights_dir/$name" ]]; then
      echo "  thiếu: $weights_dir/$name" >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    return 1
  fi
  setup_log "swiftedit_weights OK"
}

setup_download_weights() {
  local root="$1"
  bash "$root/scripts/download_swiftedit_weights.sh"
}

setup_download_hf() {
  local root="$1"
  bash "$root/scripts/download_hf_models.sh"
}

setup_write_env_file() {
  local root="$1"
  local in_colab="${2:-0}"
  local hf_home="${3:-}"
  mkdir -p "$root/results/phase3/ablation" "$root/results/phase3/metrics"
  mkdir -p "$root/results/piebench/edited_images" "$root/data/PIE-Bench"
  cat >"$root/.setup_env" <<EOF
export PROJECT_ROOT="$root"
export SWIFTEDIT_DIR="$root/SwiftEdit"
export WEIGHTS_DIR="$root/SwiftEdit/swiftedit_weights"
export RESULTS_DIR="$root/results"
export PHASE3_DIR="$root/results/phase3"
export ABLATION_DIR="$root/results/phase3/ablation"
export METRICS_DIR="$root/results/phase3/metrics"
export PIEBENCH_DIR="$root/data/PIE-Bench"
export PIEBENCH_RESULTS_DIR="$root/results/piebench"
export IN_COLAB="$in_colab"
export HF_HOME="$hf_home"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-900}"
EOF
  setup_log "Wrote $root/.setup_env"
}

setup_torch_report() {
  python3 <<'PY'
import torch

print(f"  torch: {torch.__version__}")
print(f"  cuda: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
mps = getattr(torch.backends, "mps", None)
print(f"  mps: {bool(mps and torch.backends.mps.is_available())}")
PY
}
