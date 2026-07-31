"""Helpers gắn / gỡ LoRA lên UNet (PEFT) cho SwiftEdit gen path."""

from __future__ import annotations

from pathlib import Path


def attach_lora_to_unet(unet, lora_path: str | Path, *, adapter_name: str = "daynight"):
    """Nạp LoRA vào UNet. Yêu cầu package `peft` + `safetensors`/`diffusers`."""
    lora_path = Path(lora_path)
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "Cần cài peft: pip install peft. "
            "Hoặc merge LoRA vào weight rồi load checkpoint gen như bình thường."
        ) from exc

    if (lora_path / "adapter_config.json").exists():
        return PeftModel.from_pretrained(unet, str(lora_path), adapter_name=adapter_name)

    # Diffusers-style export
    try:
        unet.load_attn_procs(str(lora_path))
        return unet
    except Exception as exc:
        raise RuntimeError(
            f"Không nạp được LoRA từ {lora_path}. "
            "Kỳ vọng thư mục PEFT (adapter_config.json) hoặc diffusers attn procs."
        ) from exc


def detach_lora(unet):
    """Trả về base model nếu đang bọc PeftModel; ngược lại giữ nguyên."""
    try:
        from peft import PeftModel
    except ImportError:
        return unet
    if isinstance(unet, PeftModel):
        return unet.get_base_model()
    return unet
