# Shared attention op: PyTorch SDP (mặc định) hoặc xFormers Memory-Efficient Attention.
#
# Không gọi unet.enable_xformers_memory_efficient_attention() trên gen UNet vì
# SwiftEdit đã gắn IPAttnProcessor tùy chỉnh — API Diffusers sẽ ghi đè processor.
# Inverse UNet (không IP) vẫn dùng Diffusers enable_* an toàn.

from __future__ import annotations

import torch
import torch.nn.functional as F

_USE_XFORMERS = False


def set_use_xformers(enabled: bool) -> None:
    """Bật/tắt MEA cho các processor custom (AttnProcessor2_0 / IP*)."""
    global _USE_XFORMERS
    if enabled:
        import xformers  # noqa: F401
        import xformers.ops  # noqa: F401

    _USE_XFORMERS = bool(enabled)


def use_xformers() -> bool:
    return _USE_XFORMERS


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """query/key/value: (batch, heads, seq, head_dim) — cùng layout SDP.

    Trả về cùng layout (batch, heads, seq, head_dim).
    """
    if not _USE_XFORMERS:
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

    import xformers.ops as xops

    # xFormers: (batch, seq, heads, head_dim)
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()

    bias = None
    if attn_mask is not None:
        # SDP mask: (B, H, Q, K) bool hoặc float. xFormers muốn attn_bias float.
        if attn_mask.dtype == torch.bool:
            bias = torch.zeros(
                attn_mask.shape, dtype=q.dtype, device=q.device
            ).masked_fill(~attn_mask, torch.finfo(q.dtype).min)
        else:
            bias = attn_mask.to(dtype=q.dtype)

    out = xops.memory_efficient_attention(q, k, v, attn_bias=bias, p=0.0)
    return out.transpose(1, 2)
