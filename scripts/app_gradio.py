#!/usr/bin/env python3
"""Demo UI Gradio cho SwiftEdit (SwiftEdit-RT).

Tích hợp các tối ưu đã làm:
  - fp16 + channels_last (VAE giữ fp32) -> nhanh ~3.3-7× so fp32, chất lượng ~không đổi.
  - EditCache -> cùng ảnh + source prompt, đổi nhiều edit prompt thì các lần sau nhanh hơn.

Chạy:
  python scripts/app_gradio.py                 # mặc định fp16 trên MPS/CUDA
  python scripts/app_gradio.py --dtype fp32    # so sánh baseline
  python scripts/app_gradio.py --share         # tạo link public tạm thời
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from hybrid_editing import (
    Candidate,
    EditSession,
    SquareROI,
    commit_candidate,
    crop_mask,
    crop_square,
    ensure_session,
    hybrid_composite,
    paste_square,
    square_roi_from_mask,
    undo_session,
)

ROOT = Path(__file__).resolve().parent.parent

EDIT_SIZE = 512

EXAMPLE_PROMPTS = [
    ["a slanted mountain bicycle on the road in front of a building",
     "a slanted rusty mountain motorcycle in front of a fence"],
]

REMOVAL_EXAMPLE_PROMPTS = [
    ["a cat wearing headphones on a gray background",
     "a cat on a plain gray background"],
]

_VAGUE_EDIT_PROMPTS = {"empty background", "background", "empty", ""}

BRUSH_REFRESH_JS = r"""
() => {
  const findButton = (root, name) => Array.from(root.querySelectorAll("button")).find(
    (button) => (button.getAttribute("aria-label") || button.textContent || "")
      .trim().toLowerCase() === name
  );

  const refreshBrush = () => {
    const root = document.querySelector("#mask-editor");
    const image = root?.querySelector("img");
    if (!root || !image || !image.currentSrc) return;
    const textureKey = [
      image.currentSrc,
      image.naturalWidth,
      image.naturalHeight,
      root.clientWidth,
      root.clientHeight,
    ].join(":");
    if (image.dataset.brushTextureKey === textureKey) return;
    const erase = findButton(root, "erase");
    const brush = findButton(root, "brush");
    if (!erase || !brush) return;
    image.dataset.brushTextureKey = textureKey;
    // Gradio 5 chỉ recreate brush textures khi draw/erase mode thay đổi.
    erase.click();
    requestAnimationFrame(() => brush.click());
  };

  const scheduleRefresh = () => {
    window.setTimeout(refreshBrush, 80);
    window.setTimeout(refreshBrush, 250);
  };

  const syncBrushSize = () => {
    const root = document.querySelector("#mask-editor");
    if (!root) return;
    const slider = Array.from(root.querySelectorAll('input[type="range"]')).find(
      (input) => input.min === "1" && input.max === "100"
    );
    if (!slider) return;
    root.dataset.brushSize = slider.value;
    if (slider.dataset.liveMaskBound === "1") return;
    slider.dataset.liveMaskBound = "1";
    const update = () => {
      root.dataset.brushSize = slider.value;
    };
    slider.addEventListener("input", update);
    slider.addEventListener("change", update);
  };

  const setupLivePreview = () => {
    const root = document.querySelector("#mask-editor");
    if (!root || root.dataset.liveMaskBound === "1") return;
    const baseCanvas = () => Array.from(root.querySelectorAll("canvas"))
      .find((canvas) => !canvas.classList.contains("live-mask-preview"));
    if (!baseCanvas()) return;
    root.dataset.liveMaskBound = "1";
    root.style.position = "relative";

    const overlay = document.createElement("canvas");
    overlay.className = "live-mask-preview";
    overlay.style.position = "absolute";
    overlay.style.pointerEvents = "none";
    overlay.style.zIndex = "50";
    root.appendChild(overlay);
    let drawing = false;
    let lastPoint = null;

    const syncOverlay = () => {
      const base = baseCanvas();
      if (!base) return null;
      const baseRect = base.getBoundingClientRect();
      const rootRect = root.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      overlay.style.left = `${baseRect.left - rootRect.left}px`;
      overlay.style.top = `${baseRect.top - rootRect.top}px`;
      overlay.style.width = `${baseRect.width}px`;
      overlay.style.height = `${baseRect.height}px`;
      const width = Math.max(1, Math.round(baseRect.width * ratio));
      const height = Math.max(1, Math.round(baseRect.height * ratio));
      if (overlay.width !== width || overlay.height !== height) {
        overlay.width = width;
        overlay.height = height;
      }
      const context = overlay.getContext("2d");
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { baseRect, context };
    };

    const brushIsActive = () => {
      const brush = findButton(root, "brush");
      return !!brush && brush.classList.contains("highlight");
    };
    const pointFor = (event, rect) => ({
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    });
    const brushWidth = () => {
      const zoomText = Array.from(root.querySelectorAll("span"))
        .map((span) => span.textContent.trim())
        .find((text) => /^\d+%$/.test(text));
      const zoom = zoomText ? Number.parseFloat(zoomText) / 100 : 1;
      const size = Number.parseFloat(root.dataset.brushSize || "28");
      return Math.max(2, size * zoom);
    };
    const inside = (event, rect) => (
      event.clientX >= rect.left && event.clientX <= rect.right
      && event.clientY >= rect.top && event.clientY <= rect.bottom
    );
    const drawSegment = (from, to, context) => {
      context.strokeStyle = "rgba(255, 0, 0, 0.5)";
      context.fillStyle = "rgba(255, 0, 0, 0.5)";
      context.lineWidth = brushWidth();
      context.lineCap = "round";
      context.lineJoin = "round";
      context.beginPath();
      context.moveTo(from.x, from.y);
      context.lineTo(to.x, to.y);
      context.stroke();
    };

    root.addEventListener("pointerdown", (event) => {
      const synced = syncOverlay();
      if (!synced || !brushIsActive() || !inside(event, synced.baseRect)) return;
      drawing = true;
      lastPoint = pointFor(event, synced.baseRect);
      drawSegment(lastPoint, lastPoint, synced.context);
    }, true);
    root.addEventListener("pointermove", (event) => {
      if (!drawing || !lastPoint) return;
      const synced = syncOverlay();
      if (!synced) return;
      const nextPoint = pointFor(event, synced.baseRect);
      drawSegment(lastPoint, nextPoint, synced.context);
      lastPoint = nextPoint;
    }, true);
    const finish = () => {
      if (!drawing) return;
      drawing = false;
      lastPoint = null;
      window.setTimeout(() => {
        const context = overlay.getContext("2d");
        context.clearRect(0, 0, overlay.width, overlay.height);
      }, 120);
    };
    root.addEventListener("pointerup", finish, true);
    root.addEventListener("pointercancel", finish, true);
    new ResizeObserver(syncOverlay).observe(root);
    syncOverlay();
  };

  new MutationObserver(() => {
    scheduleRefresh();
    syncBrushSize();
    setupLivePreview();
  }).observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["src"],
  });
  window.addEventListener("resize", () => {
    scheduleRefresh();
    syncBrushSize();
    setupLivePreview();
  });
  scheduleRefresh();
  syncBrushSize();
  setupLivePreview();
}
"""


def _sync(device) -> None:
    d = str(device)
    if d.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif d.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    # edit_image trả batch [2,3,H,W] = [source_recon, edited]; lấy ảnh edit (cuối).
    if t.dim() == 4:
        t = t[-1]
    arr = t.clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def image_editor_value(image: Image.Image) -> dict:
    image = image.convert("RGB")
    return {"background": image, "layers": [], "composite": image}


def _letterbox_meta(img: Image.Image, size: int = EDIT_SIZE) -> dict:
    """Metadata pad/scale để đưa ảnh gốc vào canvas vuông `size` và khôi phục sau."""
    w, h = img.size
    if w == size and h == size:
        return {"pad": (0, 0), "content_size": (size, size), "orig_size": (w, h)}
    scale = size / max(w, h)
    cw, ch = int(round(w * scale)), int(round(h * scale))
    left = (size - cw) // 2
    top = (size - ch) // 2
    return {"pad": (left, top), "content_size": (cw, ch), "orig_size": (w, h)}


def _letterbox_image(img: Image.Image, size: int = EDIT_SIZE) -> tuple[Image.Image, dict]:
    img = img.convert("RGB")
    meta = _letterbox_meta(img, size)
    w, h = meta["orig_size"]
    if w == size and h == size:
        return img.copy(), meta
    cw, ch = meta["content_size"]
    left, top = meta["pad"]
    resized = img.resize((cw, ch), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    canvas.paste(resized, (left, top))
    return canvas, meta


def _letterbox_mask(mask: np.ndarray, meta: dict, size: int = EDIT_SIZE) -> np.ndarray:
    m = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    cw, ch = meta["content_size"]
    left, top = meta["pad"]
    if m.size == (cw, ch):
        resized = m
    else:
        resized = m.resize((cw, ch), Image.Resampling.NEAREST)
    canvas = np.zeros((size, size), np.float32)
    canvas[top : top + ch, left : left + cw] = (np.asarray(resized) > 127).astype(np.float32)
    return canvas


def _unletterbox(img: Image.Image, meta: dict, size: int = EDIT_SIZE) -> Image.Image:
    left, top = meta["pad"]
    cw, ch = meta["content_size"]
    orig_size = meta["orig_size"]
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    cropped = img.crop((left, top, left + cw, top + ch))
    if cropped.size == orig_size:
        return cropped
    return cropped.resize(orig_size, Image.Resampling.LANCZOS)


def _model_image_cache_path(image_path: str) -> Path:
    p = Path(image_path)
    st = p.stat()
    key = hashlib.sha256(f"{p.resolve()}:{st.st_mtime_ns}:{st.st_size}".encode()).hexdigest()[:24]
    return Path(tempfile.gettempdir()) / "swiftedit_demo" / f"{key}.png"


def _prepare_model_image(image_path: str) -> tuple[str, dict]:
    """Letterbox ảnh gốc → file 512×512 ổn định cho infer + EditCache."""
    orig = Image.open(image_path).convert("RGB")
    boxed, meta = _letterbox_image(orig, EDIT_SIZE)
    cache_path = _model_image_cache_path(image_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    boxed.save(cache_path)
    return str(cache_path), meta


def _prepare_roi_image(session: EditSession, roi: SquareROI) -> tuple[str, Image.Image]:
    """Cắt ROI vuông và lưu proxy 512×512 ổn định cho một candidate batch."""
    crop = crop_square(session.master, roi)
    model_image = crop.resize((EDIT_SIZE, EDIT_SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(crop)
    key = hashlib.sha256(arr.tobytes()).hexdigest()[:24]
    cache_path = Path(tempfile.gettempdir()) / "swiftedit_demo" / f"roi_{key}.png"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model_image.save(cache_path)
    return str(cache_path), crop


def extract_editor_mask(editor_value):
    """Lấy mask nhị phân (H,W) từ giá trị gr.ImageEditor và ảnh nền (PIL).

    Vùng người dùng tô (alpha > 0 ở các layer) = 1 = vật thể cần xóa.
    """
    if not editor_value:
        return None, None
    bg = editor_value.get("background")
    layers = editor_value.get("layers") or []
    if bg is None:
        return None, None
    bg = np.asarray(bg)
    h, w = bg.shape[:2]
    mask = np.zeros((h, w), np.float32)
    for layer in layers:
        la = np.asarray(layer)
        if la.ndim == 3 and la.shape[-1] == 4:
            mask = np.maximum(mask, (la[..., 3] > 0).astype(np.float32))
        elif la.ndim == 3:
            mask = np.maximum(mask, (la.sum(-1) > 0).astype(np.float32))
        elif la.ndim == 2:
            mask = np.maximum(mask, (la > 0).astype(np.float32))
    # Gradio đôi khi ghi nét vẽ vào composite thay vì layers (layers=False).
    if mask.sum() < 1:
        composite = editor_value.get("composite")
        if composite is not None:
            comp = np.asarray(composite)
            if comp.shape[:2] == (h, w):
                diff = np.abs(
                    comp[..., :3].astype(np.float32) - bg[..., :3].astype(np.float32)
                ).sum(-1)
                mask = (diff > 8).astype(np.float32)
    return Image.fromarray(bg[..., :3].astype(np.uint8)), mask


def _align_mask_to_image(mask: np.ndarray, meta: dict) -> np.ndarray:
    """Đưa mask người dùng vào cùng letterbox 512×512 với ảnh inference."""
    return _letterbox_mask(mask, meta, EDIT_SIZE)


def _removal_prompt_hints(src_p: str, edit_p: str) -> str:
    hints: list[str] = []
    src = (src_p or "").strip()
    edit = (edit_p or "").strip()
    if len(src.split()) < 4:
        hints.append(
            "source prompt quá ngắn — mô tả **toàn bộ ảnh** "
            "(người/vật + nền), không chỉ vùng cần xóa"
        )
    if edit.lower() in _VAGUE_EDIT_PROMPTS:
        hints.append(
            "edit prompt quá chung — mô tả **nền thay thế cụ thể** "
            "(vd: `plain white banner`, `blue sky and green trees`)"
        )
    if any(k in edit.lower() for k in ("text", "letter", "word", "chữ", "banner")) or \
       any(k in src.lower() for k in ("text", "letter", "word", "chữ", "banner")):
        hints.append(
            "xóa **chữ/typography** thường không hiệu quả — SwiftEdit không phải inpainting chuyên dụng; "
            "thử vật thể rời (tai nghe, chai, biển báo) hoặc LaMa sau này"
        )
    if not hints:
        return ""
    return "  \n⚠️ **Gợi ý:** " + " · ".join(hints)


def build_app(dtype: str):
    sys.path.insert(0, str(ROOT / "SwiftEdit"))
    import os

    os.environ.setdefault("SWIFTEDIT_TIMING", "0")  # tắt log file cho demo
    from infer import EditCache, edit_image, get_device
    from models import AuxiliaryModel, InverseModel, IPSBV2Model

    device = get_device()
    channels_last = dtype == "fp16"
    print(f"[demo] device={device} dtype={dtype} channels_last={channels_last} — đang nạp model...")

    weights = ROOT / "SwiftEdit" / "swiftedit_weights"
    inverse_model = InverseModel(
        str(weights / "inverse_ckpt-120k"), device=device,
        dtype=dtype, channels_last=channels_last,
    )
    aux_model = AuxiliaryModel(device=device, dtype=dtype)
    ip_sb_model = IPSBV2Model(
        str(weights / "sbv2_0.5"),
        str(weights / "ip_adapter_ckpt-90k/ip_adapter.bin"),
        aux_model, device=device, with_ip_mask_controller=True,
        dtype=dtype, channels_last=channels_last,
    )
    print("[demo] nạp model xong.")

    cache = EditCache()

    def run_edit(image_path, src_p, edit_p, scale_edit, scale_non_edit, mask_threshold, use_cache):
        if not image_path:
            raise gr.Error("Vui lòng tải lên ảnh nguồn.")
        if not edit_p or not edit_p.strip():
            raise gr.Error("Vui lòng nhập Edit prompt.")
        active_cache = cache if use_cache else None
        model_path, lb_meta = _prepare_model_image(image_path)
        hit = use_cache and active_cache is not None and active_cache._img_path == model_path \
            and active_cache._src_p == src_p

        _sync(device)
        t0 = time.perf_counter()
        res = edit_image(
            model_path, src_p or "", edit_p,
            inverse_model, aux_model, ip_sb_model,
            scale_edit=scale_edit, scale_non_edit=scale_non_edit,
            mask_threshold=mask_threshold, cache=active_cache,
        )
        _sync(device)
        dt = time.perf_counter() - t0

        note = "cache hit (cùng ảnh + source prompt)" if hit else (
            "cache nạp mới" if use_cache else "không dùng cache")
        orig_w, orig_h = lb_meta["orig_size"]
        info = (
            f"**Thời gian:** {dt:.2f}s  \n"
            f"**Thiết bị:** `{device}` | **dtype:** `{dtype}`"
            f"{' + channels_last' if channels_last else ''} (VAE fp32)  \n"
            f"**Kích thước gốc:** {orig_w}×{orig_h} (infer letterbox {EDIT_SIZE}×{EDIT_SIZE})  \n"
            f"**Cache:** {note}"
        )
        return _unletterbox(tensor_to_pil(res), lb_meta), info

    def generate_paper_candidate_batch(
        image_path,
        src_p,
        edit_p,
        scale_edit,
        scale_non_edit,
        mask_threshold,
        batch_counter,
    ):
        """Paper-style: chỉ ảnh + prompt → 1 kết quả full-frame (self-guided mask)."""
        if not image_path:
            raise gr.Error("Vui lòng tải lên ảnh nguồn.")
        if not edit_p or not edit_p.strip():
            raise gr.Error("Vui lòng nhập Edit prompt.")
        model_path, lb_meta = _prepare_model_image(image_path)
        n = int(batch_counter or 0) + 1
        seed = 250101049 + n
        yield (
            None,
            f"**Paper demo · lần {n}:** đang sinh (self-guided mask, không tô tay)…",
            n,
        )
        _sync(device)
        t0 = time.perf_counter()
        res = edit_image(
            model_path,
            src_p or "",
            edit_p.strip(),
            inverse_model,
            aux_model,
            ip_sb_model,
            scale_edit=scale_edit,
            scale_non_edit=scale_non_edit,
            mask_threshold=mask_threshold,
            cache=EditCache(),
            seed=seed,
            user_mask=None,
        )
        out = _unletterbox(tensor_to_pil(res), lb_meta)
        _sync(device)
        elapsed = time.perf_counter() - t0
        info = (
            f"**Paper demo** (khớp paper: chỉ prompt, self-guided mask) · "
            f"**Lần {n}** · xong trong {elapsed:.2f}s  \n"
            f"**Thiết bị:** `{device}` · **dtype:** `{dtype}` · "
            f"**Seed:** `{seed}`  \n"
            f"Source prompt có thể để trống (paper Fig. 8)."
        )
        yield out, info, n

    def reset_edit_session(image_path, src_p):
        if not image_path:
            return None, None, "Tải ảnh để bắt đầu phiên chỉnh sửa."
        session = ensure_session(None, image_path, src_p or "")
        w, h = session.master.size
        return (
            session,
            image_editor_value(session.master),
            f"**Ảnh hiện tại:** {w}×{h} · lượt 0",
        )

    def generate_candidate_batch(
        image_path,
        src_p,
        edit_p,
        scale_edit,
        scale_non_edit,
        mask_threshold,
        roi_padding_percent,
        mask_blur,
        latent_strategy,
        editor_value,
        session,
    ):
        if not image_path:
            raise gr.Error("Vui lòng tải lên ảnh nguồn.")
        if not edit_p or not edit_p.strip():
            raise gr.Error("Vui lòng nhập Edit prompt.")
        session = ensure_session(session, image_path, src_p or "")
        session.source_prompt = (src_p or session.source_prompt).strip()
        _, mask = extract_editor_mask(editor_value)
        if mask is None or mask.sum() < 1:
            raise gr.Error("Chưa tô mask. Hãy dùng cọ tô vùng cần sửa trên ảnh hiện tại.")
        if mask.shape[::-1] != session.master.size:
            mask = np.asarray(
                Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    session.master.size,
                    Image.Resampling.NEAREST,
                ),
                dtype=np.float32,
            ) / 255.0
        try:
            roi = square_roi_from_mask(
                mask,
                padding_ratio=float(roi_padding_percent) / 100.0,
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        model_path, source_crop = _prepare_roi_image(session, roi)
        source_mask = crop_mask(mask, roi)
        model_user_mask = np.asarray(
            source_mask.resize((EDIT_SIZE, EDIT_SIZE), Image.Resampling.NEAREST),
            dtype=np.float32,
        ) / 255.0
        batch_cache = EditCache()
        candidates: list[Candidate] = []
        candidate_images = [None, None, None]
        batch_seed = 250101049 + session.turn * 10000 + session.batch_index * 3
        session.batch_index += 1
        session.candidates = []
        use_latent_strategy = latent_strategy == "latent"
        # Mỗi lượt encode master hiện tại đúng một lần. Không nối trực tiếp clean_latent
        # đã chọn vì local composite làm latent đó không còn khớp pixel master ngoài mask.
        batch_source_latent = None

        yield (
            *candidate_images,
            gr.skip(),
            f"**Batch {session.batch_index}:** đang sinh candidate 1/3 theo thứ tự…",
            session,
        )
        _sync(device)
        t0 = time.perf_counter()
        for index in range(3):
            candidate_started = time.perf_counter()
            seed = batch_seed + index
            # Giữ cache embedding, nhưng mỗi candidate baseline phải VAE sample bằng seed riêng.
            if batch_source_latent is None:
                batch_cache.latents = None
            details = edit_image(
                model_path,
                session.source_prompt,
                edit_p.strip(),
                inverse_model,
                aux_model,
                ip_sb_model,
                scale_edit=scale_edit,
                scale_non_edit=scale_non_edit,
                mask_threshold=mask_threshold,
                cache=batch_cache,
                seed=seed,
                user_mask=model_user_mask,
                source_latent=batch_source_latent,
                latent_jitter_strength=(0.05 if batch_source_latent is not None else 0.0),
                return_details=True,
            )
            # Batch đầu chưa có latent đã commit: candidate 0 lấy latent source,
            # candidate 1–2 jitter quanh cùng latent đó để tạo khác biệt rõ hơn.
            if use_latent_strategy and batch_source_latent is None:
                batch_source_latent = details["source_latent"][-1:].float().cpu()
            model_output = tensor_to_pil(details["image"])
            edited_crop = model_output.resize(
                (roi.size, roi.size),
                Image.Resampling.LANCZOS,
            )
            composed_crop = hybrid_composite(
                source_crop,
                edited_crop,
                source_mask,
                mode="local",
                dilation=0,
                blur=float(mask_blur),
            )
            composed = paste_square(session.master, composed_crop, roi)
            candidates.append(
                Candidate(
                    image=composed,
                    model_image=model_output,
                    mask=source_mask,
                    clean_latent=details["clean_latent"][-1:].float().cpu(),
                    seed=seed,
                    mode="masked",
                    source_prompt=session.source_prompt,
                    edit_prompt=edit_p.strip(),
                )
            )
            session.candidates = candidates
            candidate_images[index] = candidates[index].image
            _sync(device)
            candidate_elapsed = time.perf_counter() - candidate_started
            total_elapsed = time.perf_counter() - t0
            strategy_note = (
                "encode master 1 lần + latent jitter 0.05"
                if use_latent_strategy
                else "VAE seed độc lập"
            )
            if index < 2:
                progress = (
                    f"Đã có candidate {index + 1}/3 ({candidate_elapsed:.2f}s); "
                    f"đang sinh candidate {index + 2}/3 trong cùng task…"
                )
            else:
                progress = (
                    f"Đã đủ 3 candidate trong {total_elapsed:.2f}s. "
                    "Bấm **Chọn** để commit hoặc **Regen** để tạo batch mới."
                )
            modes = ", ".join(candidate.mode for candidate in candidates)
            mask_coverage = 100 * float(
                np.asarray(source_mask, dtype=np.float32).mean() / 255.0
            )
            info = (
                f"**Lượt:** {session.turn} · **Batch:** {session.batch_index} · {progress}  \n"
                f"**ROI:** x={roi.x}, y={roi.y}, {roi.size}×{roi.size} px · "
                f"mask chiếm {mask_coverage:.1f}% ROI · pixel ngoài mask giữ nguyên  \n"
                f"**Seeds đã xong:** {batch_seed}–{seed} · **Mode:** {modes} · "
                f"**Chiến lược:** {strategy_note}"
            )
            yield (
                *candidate_images,
                gr.skip(),
                info,
                session,
            )

    def pick_candidate(index, session):
        if session is None or not session.candidates:
            raise gr.Error("Chưa có candidate để chọn.")
        selected_seed = session.candidates[int(index)].seed
        commit_candidate(session, int(index))
        info = (
            f"**Đã chọn ảnh {int(index) + 1}.** Ảnh này trở thành master lượt "
            f"{session.turn}; có thể chỉnh tiếp hoặc Undo. (seed {selected_seed})"
        )
        return (
            image_editor_value(session.master),
            None,
            None,
            None,
            info,
            session,
            session.source_prompt,
        )

    def undo_edit(session):
        if session is None:
            raise gr.Error("Chưa có phiên chỉnh sửa.")
        had_history = bool(session.history)
        undo_session(session)
        info = (
            f"**Đã quay lại lượt {session.turn}.**"
            if had_history
            else "**Không có lượt trước để Undo.**"
        )
        return (
            image_editor_value(session.master),
            None,
            None,
            None,
            info,
            session,
            session.source_prompt,
        )

    def run_removal(editor_value, src_p, edit_p, scale_edit, scale_non_edit, mask_threshold):
        bg, mask = extract_editor_mask(editor_value)
        if bg is None:
            raise gr.Error("Vui lòng tải ảnh và khoanh vùng vật thể cần xóa.")
        if mask is None or mask.sum() < 1:
            raise gr.Error("Chưa khoanh vùng nào. Hãy tô lên vật thể cần xóa.")
        edit_p = (edit_p or "").strip() or "empty background"
        bg = bg.convert("RGB")
        lb_meta = _letterbox_meta(bg, EDIT_SIZE)
        bg512, _ = _letterbox_image(bg, EDIT_SIZE)
        mask512 = _align_mask_to_image(mask, lb_meta)
        tmp = Path(tempfile.mkdtemp()) / "removal_src.png"
        bg512.save(tmp)

        _sync(device)
        t0 = time.perf_counter()
        res = edit_image(
            str(tmp), src_p or "", edit_p,
            inverse_model, aux_model, ip_sb_model,
            scale_edit=scale_edit, scale_non_edit=scale_non_edit,
            mask_threshold=mask_threshold, user_mask=mask512,
        )
        _sync(device)
        dt = time.perf_counter() - t0

        mask_prev = _unletterbox(Image.fromarray((mask512 * 255).astype(np.uint8)), lb_meta)
        hints = _removal_prompt_hints(src_p, edit_p)
        orig_w, orig_h = lb_meta["orig_size"]
        info = (
            f"**Thời gian:** {dt:.2f}s  \n"
            f"**Thiết bị:** `{device}` | **dtype:** `{dtype}`"
            f"{' + channels_last' if channels_last else ''} (VAE fp32)  \n"
            f"**Kích thước gốc:** {orig_w}×{orig_h} (infer letterbox {EDIT_SIZE}×{EDIT_SIZE})  \n"
            f"**Tỉ lệ vùng khoanh:** {100*mask512.mean():.1f}% ảnh  \n"
            f"{hints}\n"
            f"_Lưu ý: SwiftEdit xóa tốt vật **rời nhỏ/vừa** (tai nghe, lon, biển báo); "
            f"**chữ trên banner**, vật chiếm gần hết khung thường không sạch._"
        )
        return _unletterbox(tensor_to_pil(res), lb_meta), mask_prev, info

    with gr.Blocks(
        title="SwiftEdit-RT Demo",
        theme=gr.themes.Soft(),
        js=BRUSH_REFRESH_JS,
    ) as demo:
        gr.Markdown(
            "# SwiftEdit-RT — Demo one-step editing\n"
            f"**Paper demo** = chỉ upload + prompt (self-guided). "
            "**ROI / tô mask** = mở rộng đề tài. "
            f"Đang chạy `{dtype}` trên `{device}`."
        )
        with gr.Tabs():
            with gr.Tab("Paper demo (chỉ prompt)"):
                paper_batch = gr.State(value=0)
                gr.Markdown(
                    "Khớp paper SwiftEdit: **upload ảnh → source/edit prompt → 1 kết quả**. "
                    "Không cần tô mask (self-guided). Source prompt có thể để trống.\n\n"
                    "**Gợi ý demo:** `woman` → `Taylor Swift` (case README, đổi rõ). "
                    "Đổi giới tính toàn cục (`a man`) thường **under-edit** vì IP-Adapter giữ identity. "
                    "Nếu yếu: mở ARaM, hạ `scale_edit` (0–0.1) hoặc dùng tab ROI + tô mặt."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        paper_image = gr.Image(
                            label="Ảnh nguồn", type="filepath", height=320,
                        )
                        paper_src = gr.Textbox(
                            label="Source prompt (tuỳ chọn)",
                            placeholder="vd: a mountain bicycle on the road — có thể để trống",
                        )
                        paper_edit = gr.Textbox(
                            label="Edit prompt (bắt buộc)",
                            placeholder="vd: a rusty motorcycle on the road",
                        )
                        with gr.Accordion("Tùy chọn ARaM", open=False):
                            paper_sl_edit = gr.Slider(
                                0.0, 1.0, value=0.2, step=0.05, label="scale_edit",
                            )
                            paper_sl_non = gr.Slider(
                                0.0, 2.0, value=1.0, step=0.05, label="scale_non_edit",
                            )
                            paper_sl_mask = gr.Slider(
                                0.0, 1.0, value=0.5, step=0.05, label="mask_threshold",
                            )
                        paper_btn = gr.Button("Tạo kết quả", variant="primary")
                        paper_regen = gr.Button("Regen")
                    with gr.Column(scale=1):
                        paper_info = gr.Markdown()
                        paper_out = gr.Image(label="Kết quả", height=420)
                gr.Examples(
                    examples=[[p[0], p[1]] for p in EXAMPLE_PROMPTS],
                    inputs=[paper_src, paper_edit],
                    label="Ví dụ prompt",
                )
                paper_inputs = [
                    paper_image,
                    paper_src,
                    paper_edit,
                    paper_sl_edit,
                    paper_sl_non,
                    paper_sl_mask,
                    paper_batch,
                ]
                paper_outputs = [paper_out, paper_info, paper_batch]
                paper_btn.click(
                    generate_paper_candidate_batch,
                    inputs=paper_inputs,
                    outputs=paper_outputs,
                )
                paper_regen.click(
                    generate_paper_candidate_batch,
                    inputs=paper_inputs,
                    outputs=paper_outputs,
                )

            with gr.Tab("ROI / tô mask (mở rộng)"):
                edit_session = gr.State(value=None)
                with gr.Row():
                    with gr.Column(scale=1):
                        inp_image = gr.Image(label="Ảnh nguồn", type="filepath", height=320)
                        gr.Markdown(
                            "**Mở rộng đề tài (không bắt buộc theo paper):** tô mask trên "
                            "*Ảnh hiện tại* → crop ROI → Hybrid blend. "
                            "Dùng khi cần kiểm soát vùng sửa / multi-turn."
                        )
                        current_image = gr.ImageEditor(
                            label="Ảnh hiện tại — tô mask vùng cần sửa",
                            type="numpy",
                            height=420,
                            elem_id="mask-editor",
                            brush=gr.Brush(
                                colors=["rgba(255, 0, 0, 0.5)"],
                                default_size=28,
                            ),
                            layers=False,
                            transforms=[],
                        )
                        inp_src = gr.Textbox(
                            label="Source prompt (mô tả nội dung trong ROI)",
                            placeholder="vd: a mountain bicycle on the road",
                        )
                        inp_edit = gr.Textbox(
                            label="Edit prompt (nội dung ROI sau khi sửa)",
                            placeholder="vd: a rusty motorcycle on the road",
                        )
                        with gr.Accordion("Tùy chọn nâng cao", open=False):
                            sl_edit = gr.Slider(0.0, 1.0, value=0.2, step=0.05,
                                                label="scale_edit (vùng chỉnh sửa)")
                            sl_non = gr.Slider(0.0, 2.0, value=1.0, step=0.05,
                                               label="scale_non_edit (giữ nền)")
                            sl_mask = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                                label="mask_threshold")
                            roi_padding = gr.Slider(
                                0,
                                100,
                                value=25,
                                step=5,
                                label="Context padding quanh mask (%)",
                            )
                            mask_blur = gr.Slider(
                                0,
                                20,
                                value=4,
                                step=1,
                                label="Mask blur khi ghép (px)",
                            )
                            latent_strategy = gr.Radio(
                                choices=[
                                    ("Encode 1 lần + latent jitter (khuyến nghị)", "latent"),
                                    ("VAE seed độc lập (đối chứng)", "baseline"),
                                ],
                                value="latent",
                                label="Chiến lược tạo candidate",
                            )
                        with gr.Row():
                            btn = gr.Button("Tạo 3 kết quả", variant="primary")
                            regen_btn = gr.Button("Regen 3 kết quả")
                            undo_btn = gr.Button("Undo")
                    with gr.Column(scale=1):
                        out_info = gr.Markdown()
                        with gr.Row():
                            with gr.Column():
                                candidate_1 = gr.Image(label="Candidate 1", height=230)
                                pick_1 = gr.Button("Chọn ảnh 1")
                            with gr.Column():
                                candidate_2 = gr.Image(label="Candidate 2", height=230)
                                pick_2 = gr.Button("Chọn ảnh 2")
                            with gr.Column():
                                candidate_3 = gr.Image(label="Candidate 3", height=230)
                                pick_3 = gr.Button("Chọn ảnh 3")
                gr.Examples(
                    examples=[[p[0], p[1]] for p in EXAMPLE_PROMPTS],
                    inputs=[inp_src, inp_edit],
                    label="Ví dụ prompt",
                )
                inp_image.change(
                    reset_edit_session,
                    inputs=[inp_image, inp_src],
                    outputs=[edit_session, current_image, out_info],
                )
                candidate_inputs = [
                    inp_image,
                    inp_src,
                    inp_edit,
                    sl_edit,
                    sl_non,
                    sl_mask,
                    roi_padding,
                    mask_blur,
                    latent_strategy,
                    current_image,
                    edit_session,
                ]
                candidate_outputs = [
                    candidate_1,
                    candidate_2,
                    candidate_3,
                    current_image,
                    out_info,
                    edit_session,
                ]
                btn.click(
                    generate_candidate_batch,
                    inputs=candidate_inputs,
                    outputs=candidate_outputs,
                )
                regen_btn.click(
                    generate_candidate_batch,
                    inputs=candidate_inputs,
                    outputs=candidate_outputs,
                )
                pick_outputs = [
                    current_image,
                    candidate_1,
                    candidate_2,
                    candidate_3,
                    out_info,
                    edit_session,
                    inp_src,
                ]
                pick_1.click(
                    lambda state: pick_candidate(0, state),
                    inputs=[edit_session],
                    outputs=pick_outputs,
                )
                pick_2.click(
                    lambda state: pick_candidate(1, state),
                    inputs=[edit_session],
                    outputs=pick_outputs,
                )
                pick_3.click(
                    lambda state: pick_candidate(2, state),
                    inputs=[edit_session],
                    outputs=pick_outputs,
                )
                undo_btn.click(
                    undo_edit,
                    inputs=[edit_session],
                    outputs=pick_outputs,
                )

            with gr.Tab("Xóa vật thể (khoanh vùng)"):
                gr.Markdown(
                    "Tải ảnh, **dùng cọ tô lên vật thể cần xóa**, rồi mô tả ảnh gốc + nền "
                    "sau khi xóa.\n\n"
                    "**Phù hợp:** vật rời nhỏ/vừa (tai nghe, lon, biển báo).  \n"
                    "**Không phù hợp:** chữ/typography trên banner, vật chiếm gần hết khung."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        rm_editor = gr.ImageEditor(
                            label="Khoanh vùng vật thể cần xóa (dùng cọ)",
                            type="numpy", height=360,
                            brush=gr.Brush(
                                colors=["rgba(255, 0, 0, 0.5)"],
                                default_size=28,
                            ),
                            layers=False, transforms=[],
                        )
                        rm_src = gr.Textbox(
                            label="Mô tả ảnh gốc (source prompt)",
                            placeholder="vd: a woman in a blue dress in front of a white banner",
                        )
                        rm_edit = gr.Textbox(
                            label="Mô tả vùng sau khi xóa (nền thay thế)",
                            placeholder="vd: plain white banner without text",
                        )
                        with gr.Accordion("Tùy chọn nâng cao", open=False):
                            rm_sl_edit = gr.Slider(0.0, 1.0, value=0.15, step=0.05,
                                                   label="scale_edit (vùng xóa; thấp = ít giữ vật gốc)")
                            rm_sl_non = gr.Slider(0.0, 2.0, value=1.2, step=0.05,
                                                  label="scale_non_edit (giữ nền)")
                            rm_sl_mask = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                                   label="mask_threshold")
                        rm_btn = gr.Button("Xóa vật thể", variant="primary")
                    with gr.Column(scale=1):
                        rm_out = gr.Image(label="Ảnh sau khi xóa", height=320)
                        rm_mask = gr.Image(label="Mask vùng khoanh", height=200)
                        rm_info = gr.Markdown()
                gr.Examples(
                    examples=[[p[0], p[1]] for p in REMOVAL_EXAMPLE_PROMPTS],
                    inputs=[rm_src, rm_edit],
                    label="Ví dụ prompt xóa vật thể",
                )
                rm_btn.click(
                    run_removal,
                    inputs=[rm_editor, rm_src, rm_edit, rm_sl_edit, rm_sl_non, rm_sl_mask],
                    outputs=[rm_out, rm_mask, rm_info],
                )
    # Một model dùng chung: chỉ chạy một inference task tại một thời điểm.
    # Generator phía trên yield từng candidate để UI hiển thị ngay khi ảnh hoàn tất.
    demo.queue(max_size=8, default_concurrency_limit=1)
    return demo, run_edit, run_removal


def main() -> int:
    parser = argparse.ArgumentParser(description="Gradio demo SwiftEdit-RT")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--share", action="store_true", help="Tạo link public tạm thời")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--selftest", type=Path, default=None,
                        help="Chạy thử 1 edit trên ảnh này rồi thoát (không mở server)")
    parser.add_argument("--selftest-removal", type=Path, default=None,
                        help="Chạy thử xóa vật thể (mask chữ nhật giữa) rồi thoát")
    args = parser.parse_args()

    demo, run_edit, run_removal = build_app(args.dtype)

    if args.selftest is not None:
        img, info = run_edit(
            str(args.selftest),
            "a slanted mountain bicycle on the road in front of a building",
            "a slanted rusty mountain motorcycle in front of a fence",
            0.2, 1.0, 0.5, True,
        )
        out = ROOT / "results" / "app_selftest.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        print(f"[selftest] OK -> {out}\n{info}")
        return 0

    if args.selftest_removal is not None:
        bg = np.asarray(Image.open(args.selftest_removal).convert("RGB").resize((512, 512)))
        layer = np.zeros((512, 512, 4), np.uint8)
        layer[110:440, 120:430, 3] = 255  # tô vùng giữa
        editor_value = {"background": bg, "layers": [layer], "composite": bg}
        img, mask, info = run_removal(
            editor_value,
            "a slanted mountain bicycle on the road in front of a building",
            "empty asphalt road in front of a building",
            0.15, 1.2, 0.5,
        )
        out = ROOT / "results" / "app_removal_selftest.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        print(f"[selftest-removal] OK -> {out}\n{info}")
        return 0

    print(
        f"[demo] Launch Gradio port={args.port} share={args.share} ...",
        flush=True,
    )
    if args.share:
        print(
            "[demo] Đang tạo public URL (*.gradio.live) — chờ sau khi server sẵn sàng...",
            flush=True,
        )
    # server_name=0.0.0.0: Colab/tunnel ổn định hơn; flush log trước khi block.
    launch_kwargs = dict(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0",
        show_error=True,
    )
    try:
        out = demo.launch(**launch_kwargs)
    except TypeError:
        # Gradio cũ hơn có thể không nhận show_error
        out = demo.launch(share=args.share, server_port=args.port, server_name="0.0.0.0")

    # Gradio 4/5 đôi khi trả (app, local_url, share_url) — in rõ để Colab không “nuốt” log.
    share_url = None
    local_url = None
    if isinstance(out, (tuple, list)):
        if len(out) >= 2:
            local_url = out[1]
        if len(out) >= 3:
            share_url = out[2]
    if local_url:
        print(f"[demo] Local URL: {local_url}", flush=True)
    if share_url:
        print(f"[demo] Public URL: {share_url}", flush=True)
    elif args.share:
        print(
            "[demo] Chưa lấy được share_url từ return value — "
            "xem dòng 'Running on public URL' phía trên.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
