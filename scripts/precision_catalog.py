"""Catalog cấu hình precision dùng chung (eval / ensure-weights / compare).

Mỗi lần Colab chỉ chạy những key user chọn — không bắt buộc chạy tuần tự cả list.
So sánh cuối (local) gộp nhiều bundle cùng jobs_hash, baseline = baseline_fp32.
"""

from __future__ import annotations

# Key chuẩn (xuất CSV / report). Alias ngắn map về key này.
CONFIGS: dict[str, dict] = {
    "baseline_fp32": dict(
        weights="fp32",
        dtype="fp32",
        channels_last=False,
        use_cache=False,
        quant=None,
        use_xformers=False,
        needs_fp16_disk=False,
        needs_cuda_quant=False,
        needs_cuda_xformers=False,
        label="FP32 full (reference)",
    ),
    "improved_fp16_cache": dict(
        weights="fp32",
        dtype="fp16",
        channels_last=True,
        use_cache=True,
        quant=None,
        use_xformers=False,
        needs_fp16_disk=False,
        needs_cuda_quant=False,
        needs_cuda_xformers=False,
        label="FP16 compute + EditCache (disk vẫn fp32)",
    ),
    "improved_fp8_cache": dict(
        weights="fp32",
        dtype="fp16",
        channels_last=True,
        use_cache=True,
        quant="fp8",
        use_xformers=False,
        needs_fp16_disk=False,
        needs_cuda_quant=True,
        needs_cuda_xformers=False,
        label="FP8 weight-only + EditCache (disk fp32)",
    ),
    "improved_fp4_cache": dict(
        weights="fp32",
        dtype="fp16",
        channels_last=True,
        use_cache=True,
        quant="fp4",
        use_xformers=False,
        needs_fp16_disk=False,
        needs_cuda_quant=True,
        needs_cuda_xformers=False,
        label="FP4 weight-only + EditCache (disk fp32)",
    ),
    "fp16_disk": dict(
        weights="fp16",
        dtype="fp16",
        channels_last=True,
        use_cache=True,
        quant=None,
        use_xformers=False,
        needs_fp16_disk=True,
        needs_cuda_quant=False,
        needs_cuda_xformers=False,
        label="FP16 weights trên disk + EditCache",
    ),
    "fp16_disk_xformers": dict(
        weights="fp16",
        dtype="fp16",
        channels_last=True,
        use_cache=True,
        quant=None,
        use_xformers=True,
        needs_fp16_disk=True,
        needs_cuda_quant=False,
        needs_cuda_xformers=True,
        label="FP16 disk + EditCache + xFormers MEA (mới vs fp16_disk)",
    ),
    "fp4_from_fp16": dict(
        weights="fp16",
        dtype="fp16",
        channels_last=True,
        use_cache=True,
        quant="fp4",
        use_xformers=False,
        needs_fp16_disk=True,
        needs_cuda_quant=True,
        needs_cuda_xformers=False,
        label="Load fp16 disk rồi quant FP4 + EditCache",
    ),
}

# Alias thân thiện trong notebook / CLI
ALIASES: dict[str, str] = {
    "fp32": "baseline_fp32",
    "32": "baseline_fp32",
    "baseline": "baseline_fp32",
    "fp16": "improved_fp16_cache",
    "16": "improved_fp16_cache",
    "fp16_cache": "improved_fp16_cache",
    "fp8": "improved_fp8_cache",
    "8": "improved_fp8_cache",
    "fp8_cache": "improved_fp8_cache",
    "fp4": "improved_fp4_cache",
    "4": "improved_fp4_cache",
    "fp4_cache": "improved_fp4_cache",
    "fp16_weight": "fp16_disk",
    "16_weight": "fp16_disk",
    "fp16_disk_cache": "fp16_disk",
    "fp16_weight_xformers": "fp16_disk_xformers",
    "16_weight_xformers": "fp16_disk_xformers",
    "fp16_xformers": "fp16_disk_xformers",
    "fp4_weight": "fp4_from_fp16",
    "4_weight": "fp4_from_fp16",
    "fp4_from_fp16_cache": "fp4_from_fp16",
}

ALL_CANONICAL = list(CONFIGS.keys())


def resolve_config_names(raw: list[str] | str) -> list[str]:
    """Nhận CSV hoặc list → list key chuẩn, giữ thứ tự, bỏ trùng.

    Nếu có baseline_fp32 thì đưa lên đầu (để PSNR nội bộ run).
    """
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in raw if str(p).strip()]

    out: list[str] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for p in parts:
        key = ALIASES.get(p, ALIASES.get(p.lower(), p))
        if key not in CONFIGS:
            unknown.append(p)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    if unknown:
        raise ValueError(
            "Config không hợp lệ: "
            + ", ".join(unknown)
            + "\nCanonical: "
            + ", ".join(ALL_CANONICAL)
            + "\nAlias: "
            + ", ".join(sorted(ALIASES))
        )
    if "baseline_fp32" in out:
        out = ["baseline_fp32"] + [c for c in out if c != "baseline_fp32"]
    return out


def needs_fp16_disk(names: list[str]) -> bool:
    return any(CONFIGS[n]["needs_fp16_disk"] for n in names)


def help_table() -> str:
    lines = ["| Key | Alias gợi ý | Mô tả |", "|---|---|---|"]
    hint = {
        "baseline_fp32": "fp32, 32",
        "improved_fp16_cache": "fp16, 16",
        "improved_fp8_cache": "fp8, 8",
        "improved_fp4_cache": "fp4, 4",
        "fp16_disk": "fp16_weight, 16_weight",
        "fp16_disk_xformers": "fp16_weight_xformers",
        "fp4_from_fp16": "fp4_weight, 4_weight",
    }
    for k, meta in CONFIGS.items():
        lines.append(f"| `{k}` | {hint.get(k, '')} | {meta['label']} |")
    return "\n".join(lines)
