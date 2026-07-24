#!/usr/bin/env python3
"""Demo Gradio cho Google Colab T4 — config `fp16_disk_xformers` + EditCache.

Khác `app_gradio.py` (Mac MPS / fp16 compute trên weights FP32):
  - Bắt buộc CUDA (T4)
  - Weights **FP16 trên disk** (`swiftedit_weights_fp16`)
  - Bật **xFormers Memory-Efficient Attention**
  - Ưu tiên **Google Drive**; thiếu thì tải Qualcomm (+ convert fp16 nếu cần)
  - Mặc định **lưu lại Drive** sau tải/convert (`--save-to-drive`)

Pipeline: kiểm tra weights → thiếu thì tải (log) → load GPU → mới launch + share.

  python scripts/app_gradio_t4_xformers.py --share-mode gradio --port 7860
  # share-mode: gradio | ngrok | both | none  (ngrok cần NGROK_AUTHTOKEN)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

print("[t4] Khởi động script…", flush=True)

# Import nặng được gọi trong main() — để --help không cần gradio/torch.
gr = np = torch = Image = None  # type: ignore


def _import_runtime() -> None:
    global gr, np, torch, Image
    if torch is not None:
        return
    print(
        "[t4] Đang import torch/gradio/xformers (có thể 30–90s, VRAM chưa tăng)…",
        flush=True,
    )
    import gradio as _gr
    import numpy as _np
    import torch as _torch
    from PIL import Image as _Image

    gr, np, torch, Image = _gr, _np, _torch, _Image
    print(
        f"[t4] Import xong | torch={torch.__version__} cuda={torch.cuda.is_available()}",
        flush=True,
    )


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DEFAULT_DRIVE_FP16 = Path("/content/drive/MyDrive/CS2309/swiftedit_weights_fp16")
DEFAULT_DRIVE_FP32 = Path("/content/drive/MyDrive/CS2309/swiftedit_weights")
LOCAL_FP16 = ROOT / "SwiftEdit" / "swiftedit_weights_fp16"
LOCAL_FP32 = ROOT / "SwiftEdit" / "swiftedit_weights"
EDIT_SIZE = 512
CONFIG_NAME = "fp16_disk_xformers"


def _step(n: int, total: int, msg: str) -> None:
    print(f"\n{'=' * 60}\n[t4] BƯỚC {n}/{total}: {msg}\n{'=' * 60}", flush=True)


def _log_vram(prefix: str = "[t4]") -> None:
    if torch is None or not torch.cuda.is_available():
        return
    used = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"{prefix} VRAM alloc={used:.0f} MB | reserved={reserved:.0f} MB", flush=True)

EXAMPLE_PROMPTS = [
    [
        "a slanted mountain bicycle on the road in front of a building",
        "a slanted rusty mountain motorcycle in front of a fence",
    ],
]
REMOVAL_EXAMPLE_PROMPTS = [
    [
        "a cat wearing headphones on a gray background",
        "a cat on a plain gray background",
    ],
]
_VAGUE_EDIT_PROMPTS = {"empty background", "background", "empty", ""}

# Giống app_gradio.py (Mac) — refresh brush Gradio 5 + live preview mask.
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


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _tree_ok(path: Path) -> bool:
    return (
        (path / "sbv2_0.5").is_dir()
        and (path / "ip_adapter_ckpt-90k" / "ip_adapter.bin").is_file()
        and (path / "inverse_ckpt-120k").exists()
    )


def _sync(device) -> None:
    d = str(device)
    if d.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.dim() == 4:
        t = t[-1]
    arr = t.clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def _letterbox_meta(img: Image.Image, size: int = EDIT_SIZE) -> dict:
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
    if m.size != (cw, ch):
        m = m.resize((cw, ch), Image.Resampling.NEAREST)
    canvas = np.zeros((size, size), np.float32)
    canvas[top : top + ch, left : left + cw] = (np.asarray(m) > 127).astype(np.float32)
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


def _prepare_model_image(image_path: str) -> tuple[str, dict]:
    orig = Image.open(image_path).convert("RGB")
    boxed, meta = _letterbox_image(orig, EDIT_SIZE)
    p = Path(image_path)
    st = p.stat()
    key = hashlib.sha256(f"{p.resolve()}:{st.st_mtime_ns}:{st.st_size}".encode()).hexdigest()[:24]
    cache_path = Path(tempfile.gettempdir()) / "swiftedit_t4_demo" / f"{key}.png"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    boxed.save(cache_path)
    return str(cache_path), meta


def image_editor_value(image: Image.Image) -> dict:
    image = image.convert("RGB")
    return {"background": image, "layers": [], "composite": image}


def _prepare_roi_image(session, roi) -> tuple[str, Image.Image]:
    """Cắt ROI vuông → proxy 512×512 (giống Mac app_gradio)."""
    from hybrid_editing import crop_square

    crop = crop_square(session.master, roi)
    model_image = crop.resize((EDIT_SIZE, EDIT_SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(crop)
    key = hashlib.sha256(arr.tobytes()).hexdigest()[:24]
    cache_path = Path(tempfile.gettempdir()) / "swiftedit_t4_demo" / f"roi_{key}.png"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model_image.save(cache_path)
    return str(cache_path), crop


def extract_editor_mask(editor_value):
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
    if any(k in edit.lower() for k in ("text", "letter", "word", "chữ", "banner")) or any(
        k in src.lower() for k in ("text", "letter", "word", "chữ", "banner")
    ):
        hints.append(
            "xóa **chữ/typography** thường không hiệu quả — SwiftEdit không phải inpainting chuyên dụng"
        )
    if not hints:
        return ""
    return "  \n⚠️ **Gợi ý:** " + " · ".join(hints)


def ensure_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Cần CUDA (Google Colab T4).\n"
            "Runtime → Change runtime type → T4 GPU, rồi chạy lại."
        )
    name = torch.cuda.get_device_name(0)
    print(f"[t4] CUDA OK: {name}", flush=True)
    return "cuda"


def mount_drive_if_needed(drive_fp16: Path, drive_fp32: Path) -> None:
    if not _in_colab():
        print("[t4] Không phải Colab — bỏ qua mount Drive.", flush=True)
        return
    need_mount = any(
        str(p).startswith("/content/drive") for p in (drive_fp16, drive_fp32)
    )
    if not need_mount:
        return
    if Path("/content/drive/MyDrive").is_dir():
        print("[t4] Drive đã mount.", flush=True)
        return
    # drive.mount() trong subprocess thường KHÔNG hiện UI auth Colab.
    # Notebook phải mount trước (cell ③). Vẫn thử; fail → hướng dẫn rõ.
    print(
        "[t4] Drive chưa mount trong kernel — thử mount (nên mount sẵn trong notebook)...",
        flush=True,
    )
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except Exception as e:
        raise RuntimeError(
            "Google Drive chưa mount.\n"
            "Trong notebook CS2309_SwiftEdit_webui_t4_xformers: chạy cell ③ "
            "(có `_mount_drive()`) trước — mount phải ở kernel, không phải subprocess.\n"
            f"Chi tiết: {e}"
        ) from e
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError(
            "Mount Drive thất bại (không thấy /content/drive/MyDrive). "
            "Chạy lại cell mount trong notebook và Connect khi Colab hỏi."
        )


def ensure_xformers() -> str:
    try:
        import xformers

        ver = getattr(xformers, "__version__", "?")
        print(f"[t4] xformers {ver}", flush=True)
        return ver
    except ImportError:
        print("[t4] Chưa có xformers — pip install...", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "xformers"],
        )
        import xformers

        ver = getattr(xformers, "__version__", "?")
        print(f"[t4] xformers {ver} (vừa cài)", flush=True)
        return ver


def save_weights_to_drive(local: Path, drive: Path, *, label: str) -> None:
    """Copy cây weights local → Drive nếu Drive chưa có (tránh tải lại lần sau)."""
    import shutil

    if local.is_symlink():
        try:
            if local.resolve() == drive.resolve():
                print(f"[t4] {label}: local đã trỏ Drive — bỏ qua lưu.", flush=True)
                return
        except OSError:
            pass
    if not _tree_ok(local):
        print(f"[t4] {label}: local chưa OK — bỏ qua lưu Drive.", flush=True)
        return
    if _tree_ok(drive):
        print(f"[t4] {label}: Drive đã có → {drive}", flush=True)
        return
    drive.parent.mkdir(parents=True, exist_ok=True)
    if drive.exists() and not _tree_ok(drive):
        print(
            f"[t4] {label}: Drive path tồn tại nhưng thiếu file — không ghi đè: {drive}",
            flush=True,
        )
        return
    print(
        f"[t4] {label}: đang copy lên Drive (~GB, có thể lâu)...\n  {local} → {drive}",
        flush=True,
    )
    shutil.copytree(local, drive)
    print(f"[t4] {label}: đã lưu Drive → {drive}", flush=True)


def inspect_weights(*, drive_fp16: Path, drive_fp32: Path, local_fp16: Path, local_fp32: Path) -> None:
    """Log rõ đã có weights ở đâu — chưa tải/load gì."""
    rows = [
        ("Drive fp16", drive_fp16, _tree_ok(drive_fp16)),
        ("Drive fp32", drive_fp32, _tree_ok(drive_fp32)),
        ("Local fp16", local_fp16, _tree_ok(local_fp16)),
        ("Local fp32", local_fp32, _tree_ok(local_fp32)),
    ]
    print("[t4] Kiểm tra weights:", flush=True)
    for label, path, ok in rows:
        mark = "CÓ" if ok else "THIẾU"
        exists = "dir" if path.is_dir() else ("file" if path.exists() else "không tồn tại")
        print(f"  [{mark:5}] {label}: {path} ({exists})", flush=True)
    if _tree_ok(drive_fp16) or _tree_ok(local_fp16):
        print("[t4] → fp16 sẵn sàng — sẽ symlink/dùng, không cần tải Qualcomm.", flush=True)
    elif _tree_ok(drive_fp32) or _tree_ok(local_fp32):
        print("[t4] → chỉ có fp32 — sẽ convert → fp16 (lâu, tốn RAM).", flush=True)
    else:
        print(
            "[t4] → chưa có weights — sẽ TẢI Qualcomm (~10GB, vài–15 phút). "
            "VRAM vẫn ~0 trong lúc tải.",
            flush=True,
        )


def prepare_fp16_disk_weights(
    *,
    drive_fp16: Path,
    drive_fp32: Path,
    local_fp16: Path,
    local_fp32: Path,
    allow_download: bool,
    allow_convert: bool,
    save_to_drive: bool,
) -> Path:
    """Drive fp16 → local; thiếu thì tải Qualcomm + convert; tùy chọn lưu lại Drive."""
    inspect_weights(
        drive_fp16=drive_fp16,
        drive_fp32=drive_fp32,
        local_fp16=local_fp16,
        local_fp32=local_fp32,
    )

    if _tree_ok(local_fp16):
        print(f"[t4] fp16 local đã OK — bỏ qua prepare: {local_fp16}", flush=True)
        return local_fp16

    cmd = [
        sys.executable,
        "-u",
        str(SCRIPTS / "prepare_colab_weights.py"),
        "--configs",
        CONFIG_NAME,
        "--drive-fp16",
        str(drive_fp16),
        "--drive-fp32",
        str(drive_fp32),
        "--local-fp16",
        str(local_fp16),
        "--local-fp32",
        str(local_fp32),
    ]
    if allow_download:
        cmd.append("--allow-qualcomm-download")
    else:
        cmd.append("--no-qualcomm-download")
    if not allow_convert:
        cmd.append("--no-convert")

    print("[t4] Chuẩn bị weights (Drive trước → thiếu thì tải/convert)…", flush=True)
    print(f"[t4] Lệnh: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode == 0 and _tree_ok(local_fp16):
        print(f"[t4] fp16 disk OK: {local_fp16}", flush=True)
        if save_to_drive:
            save_weights_to_drive(local_fp16, drive_fp16, label="fp16")
            if _tree_ok(local_fp32):
                save_weights_to_drive(local_fp32, drive_fp32, label="fp32")
        return local_fp16

    # Fallback: tải Qualcomm nếu prepare chưa được phép / thất bại
    if allow_download and not _tree_ok(local_fp32):
        print("[t4] Fallback tải Qualcomm fp32…", flush=True)
        r2 = subprocess.run(
            [
                sys.executable,
                "-u",
                str(SCRIPTS / "download_swiftedit_weights.py"),
                "--stream",
                "--clean",
            ],
            cwd=ROOT,
        )
        if r2.returncode != 0:
            raise RuntimeError("Tải Qualcomm thất bại.")

    if not _tree_ok(local_fp16):
        if not _tree_ok(local_fp32):
            raise RuntimeError(
                "Thiếu weights fp16 và fp32.\n"
                f"  Drive fp16: {drive_fp16}\n"
                f"  Drive fp32: {drive_fp32}\n"
                "Upload `swiftedit_weights_fp16` lên Drive (khuyên), "
                "hoặc chạy lại với --allow-download."
            )
        if not allow_convert:
            raise RuntimeError(
                "Có fp32 nhưng chưa có fp16 disk. "
                "Bỏ --no-convert hoặc upload Drive fp16."
            )
        print("[t4] Convert fp32 → fp16 (chậm, tốn RAM)…", flush=True)
        r3 = subprocess.run(
            [
                sys.executable,
                "-u",
                str(SCRIPTS / "convert_weights_fp16.py"),
                "--src",
                str(local_fp32),
                "--dst",
                str(local_fp16),
            ],
            cwd=ROOT,
        )
        if r3.returncode != 0 or not _tree_ok(local_fp16):
            raise RuntimeError("Convert fp16 thất bại.")

    if not _tree_ok(local_fp16):
        raise RuntimeError(f"Vẫn thiếu fp16 disk tại {local_fp16}")
    print(f"[t4] fp16 disk sẵn sàng: {local_fp16}", flush=True)
    if save_to_drive:
        save_weights_to_drive(local_fp16, drive_fp16, label="fp16")
        if _tree_ok(local_fp32):
            save_weights_to_drive(local_fp32, drive_fp32, label="fp32")
    return local_fp16


def build_app(weights_fp16: Path):
    sys.path.insert(0, str(ROOT / "SwiftEdit"))
    os.environ.setdefault("SWIFTEDIT_TIMING", "0")

    from infer import EditCache, edit_image, get_device
    from models import AuxiliaryModel, InverseModel, IPSBV2Model

    device = get_device()
    if not str(device).startswith("cuda"):
        raise RuntimeError(f"fp16_disk_xformers cần CUDA, hiện device={device}")

    ensure_xformers()
    dtype = "fp16"
    channels_last = True
    use_xformers = True

    print(
        f"[t4] Nạp {CONFIG_NAME}: device={device} dtype={dtype} "
        f"channels_last={channels_last} xformers={use_xformers} "
        f"weights={weights_fp16}",
        flush=True,
    )
    print("[t4] (VRAM sẽ tăng ở bước này — đợi InverseModel + IPSBV2…)", flush=True)
    _log_vram()
    t0 = time.perf_counter()
    print("[t4]   → InverseModel…", flush=True)
    inverse_model = InverseModel(
        str(weights_fp16 / "inverse_ckpt-120k"),
        device=device,
        dtype=dtype,
        channels_last=channels_last,
        use_xformers=use_xformers,
    )
    _log_vram()
    print("[t4]   → AuxiliaryModel…", flush=True)
    aux_model = AuxiliaryModel(device=device, dtype=dtype)
    _log_vram()
    print("[t4]   → IPSBV2Model…", flush=True)
    ip_sb_model = IPSBV2Model(
        str(weights_fp16 / "sbv2_0.5"),
        str(weights_fp16 / "ip_adapter_ckpt-90k" / "ip_adapter.bin"),
        aux_model,
        device=device,
        with_ip_mask_controller=True,
        dtype=dtype,
        channels_last=channels_last,
        use_xformers=use_xformers,
    )
    load_s = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"[t4] Nạp xong {load_s:.1f}s | peak alloc ~{peak_mb:.0f} MB", flush=True)
    _log_vram()
    if peak_mb < 500:
        raise RuntimeError(
            f"Load model bất thường: peak VRAM chỉ ~{peak_mb:.0f} MB "
            "(kỳ vọng vài GB). Kiểm tra đường dẫn weights / CUDA."
        )

    cache = EditCache()
    runtime_banner = (
        f"**Config:** `{CONFIG_NAME}` · **device:** `{device}` · "
        f"**dtype:** fp16 + channels_last + xFormers MEA + EditCache  \n"
        f"**Weights:** `{weights_fp16}` (disk FP16) · load {load_s:.1f}s · "
        f"peak ~{peak_mb:.0f} MB"
    )

    # hybrid_editing nằm cạnh script — cùng UI multi-candidate như Mac.
    sys.path.insert(0, str(SCRIPTS))
    from hybrid_editing import (  # noqa: E402
        Candidate,
        commit_candidate,
        crop_mask,
        ensure_session,
        hybrid_composite,
        paste_square,
        square_roi_from_mask,
        undo_session,
    )

    def run_edit(image_path, src_p, edit_p, scale_edit, scale_non_edit, mask_threshold, use_cache):
        """Selftest / API đơn giản (không ROI) — giữ tương thích --selftest."""
        if not image_path:
            raise gr.Error("Vui lòng tải lên ảnh nguồn.")
        if not edit_p or not edit_p.strip():
            raise gr.Error("Vui lòng nhập Edit prompt.")
        active_cache = cache if use_cache else None
        model_path, lb_meta = _prepare_model_image(image_path)
        hit = (
            use_cache
            and active_cache is not None
            and active_cache._img_path == model_path
            and active_cache._src_p == src_p
        )

        _sync(device)
        t_edit = time.perf_counter()
        res = edit_image(
            model_path,
            src_p or "",
            edit_p,
            inverse_model,
            aux_model,
            ip_sb_model,
            scale_edit=scale_edit,
            scale_non_edit=scale_non_edit,
            mask_threshold=mask_threshold,
            cache=active_cache,
        )
        _sync(device)
        dt = time.perf_counter() - t_edit

        note = (
            "cache hit (cùng ảnh + source prompt)"
            if hit
            else ("cache nạp mới" if use_cache else "không dùng cache")
        )
        orig_w, orig_h = lb_meta["orig_size"]
        info = (
            f"{runtime_banner}  \n"
            f"**Thời gian:** {dt:.2f}s · {note}  \n"
            f"**Kích thước gốc:** {orig_w}×{orig_h}"
        )
        return _unletterbox(tensor_to_pil(res), lb_meta), info

    def reset_edit_session(image_path, src_p):
        if not image_path:
            return None, None, "Tải ảnh để bắt đầu phiên chỉnh sửa."
        session = ensure_session(None, image_path, src_p or "")
        w, h = session.master.size
        return (
            session,
            image_editor_value(session.master),
            f"{runtime_banner}  \n**Ảnh hiện tại:** {w}×{h} · lượt 0",
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
            mask = (
                np.asarray(
                    Image.fromarray((mask * 255).astype(np.uint8)).resize(
                        session.master.size,
                        Image.Resampling.NEAREST,
                    ),
                    dtype=np.float32,
                )
                / 255.0
            )
        try:
            roi = square_roi_from_mask(
                mask,
                padding_ratio=float(roi_padding_percent) / 100.0,
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        model_path, source_crop = _prepare_roi_image(session, roi)
        source_mask = crop_mask(mask, roi)
        model_user_mask = (
            np.asarray(
                source_mask.resize((EDIT_SIZE, EDIT_SIZE), Image.Resampling.NEAREST),
                dtype=np.float32,
            )
            / 255.0
        )
        batch_cache = EditCache()
        candidates: list = []
        candidate_images = [None, None, None]
        batch_seed = 250101049 + session.turn * 10000 + session.batch_index * 3
        session.batch_index += 1
        session.candidates = []
        use_latent_strategy = latent_strategy == "latent"
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
                    f"đang sinh candidate {index + 2}/3…"
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
                f"{runtime_banner}  \n"
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
            f"{runtime_banner}  \n"
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
            f"{runtime_banner}  \n**Đã quay lại lượt {session.turn}.**"
            if had_history
            else f"{runtime_banner}  \n**Không có lượt trước để Undo.**"
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
        bg_img, mask = extract_editor_mask(editor_value)
        if bg_img is None:
            raise gr.Error("Vui lòng tải ảnh và khoanh vùng cần xóa.")
        if mask is None or mask.sum() < 1:
            raise gr.Error("Chưa tô mask — dùng cọ khoanh vật thể cần xóa.")
        edit_p = (edit_p or "").strip() or "empty background"

        boxed, lb_meta = _letterbox_image(bg_img.convert("RGB"), EDIT_SIZE)
        mask512 = _letterbox_mask(mask, lb_meta, EDIT_SIZE)
        tmp = Path(tempfile.gettempdir()) / "swiftedit_t4_demo" / "removal_src.png"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        boxed.save(tmp)

        _sync(device)
        t_rm = time.perf_counter()
        res = edit_image(
            str(tmp),
            src_p or "",
            edit_p,
            inverse_model,
            aux_model,
            ip_sb_model,
            scale_edit=scale_edit,
            scale_non_edit=scale_non_edit,
            mask_threshold=mask_threshold,
            user_mask=mask512,
            cache=None,
        )
        _sync(device)
        dt = time.perf_counter() - t_rm

        mask_prev = _unletterbox(
            Image.fromarray((mask512 * 255).astype(np.uint8)), lb_meta
        )
        hints = _removal_prompt_hints(src_p, edit_p)
        orig_w, orig_h = lb_meta["orig_size"]
        info = (
            f"{runtime_banner}  \n"
            f"**Thời gian:** {dt:.2f}s  \n"
            f"**Kích thước gốc:** {orig_w}×{orig_h} · "
            f"vùng khoanh {100 * mask512.mean():.1f}%  \n"
            f"{hints}\n"
            f"_Lưu ý: SwiftEdit xóa tốt vật **rời nhỏ/vừa**; "
            f"**chữ trên banner** thường không sạch._"
        )
        return _unletterbox(tensor_to_pil(res), lb_meta), mask_prev, info

    with gr.Blocks(
        title="SwiftEdit-RT Demo (T4)",
        theme=gr.themes.Soft(),
        js=BRUSH_REFRESH_JS,
    ) as demo:
        gr.Markdown(
            "# SwiftEdit-RT — Chỉnh sửa & xóa vật thể bằng prompt (one-step)\n"
            f"Colab T4: **fp16 disk + channels_last + xFormers + EditCache** "
            f"(đang chạy trên `{device}`).\n\n"
            f"{runtime_banner}"
        )
        with gr.Tabs():
            with gr.Tab("Chỉnh sửa bằng prompt"):
                edit_session = gr.State(value=None)
                with gr.Row():
                    with gr.Column(scale=1):
                        inp_image = gr.Image(label="Ảnh nguồn", type="filepath", height=320)
                        gr.Markdown(
                            "**Chọn vùng sửa:** dùng cọ tô mask trên *Ảnh hiện tại*. "
                            "Hệ thống tự tạo crop vuông có context bao quanh mask, "
                            "sau đó chỉ ghép kết quả vào đúng vùng đã tô."
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
                            sl_edit = gr.Slider(
                                0.0, 1.0, value=0.2, step=0.05,
                                label="scale_edit (vùng chỉnh sửa)",
                            )
                            sl_non = gr.Slider(
                                0.0, 2.0, value=1.0, step=0.05,
                                label="scale_non_edit (giữ nền)",
                            )
                            sl_mask = gr.Slider(
                                0.0, 1.0, value=0.5, step=0.05,
                                label="mask_threshold",
                            )
                            roi_padding = gr.Slider(
                                0, 100, value=25, step=5,
                                label="Context padding quanh mask (%)",
                            )
                            mask_blur = gr.Slider(
                                0, 20, value=4, step=1,
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
                            type="numpy",
                            height=360,
                            brush=gr.Brush(
                                colors=["rgba(255, 0, 0, 0.5)"],
                                default_size=28,
                            ),
                            layers=False,
                            transforms=[],
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
                            rm_sl_edit = gr.Slider(
                                0.0, 1.0, value=0.15, step=0.05,
                                label="scale_edit (vùng xóa; thấp = ít giữ vật gốc)",
                            )
                            rm_sl_non = gr.Slider(
                                0.0, 2.0, value=1.2, step=0.05,
                                label="scale_non_edit (giữ nền)",
                            )
                            rm_sl_mask = gr.Slider(
                                0.0, 1.0, value=0.5, step=0.05,
                                label="mask_threshold",
                            )
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

    demo.queue(max_size=8, default_concurrency_limit=1)
    return demo, run_edit, run_removal


def _start_ngrok(port: int) -> str | None:
    """Tạo tunnel ngrok nếu có token (env NGROK_AUTHTOKEN hoặc Colab secret)."""
    token = os.environ.get("NGROK_AUTHTOKEN") or os.environ.get("NGROK_TOKEN")
    if not token and _in_colab():
        try:
            from google.colab import userdata

            token = userdata.get("NGROK_AUTHTOKEN")
        except Exception:
            token = None
    if not token:
        print(
            "[t4] ngrok: không có NGROK_AUTHTOKEN — bỏ qua "
            "(Colab Secrets hoặc env).",
            flush=True,
        )
        return None
    try:
        from pyngrok import ngrok
    except ImportError:
        print("[t4] ngrok: pip install pyngrok…", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyngrok"])
        from pyngrok import ngrok

    ngrok.set_auth_token(token)
    tunnel = ngrok.connect(port, "http")
    url = tunnel.public_url if hasattr(tunnel, "public_url") else str(tunnel)
    print(f"[t4] ngrok Public URL: {url}", flush=True)
    return url


def _launch_gradio(demo, *, port: int, share_mode: str) -> int:
    """share_mode: gradio | ngrok | both | none"""
    use_gradio_share = share_mode in ("gradio", "both")
    use_ngrok = share_mode in ("ngrok", "both")

    print(
        f"[t4] Launch Gradio port={port} share_mode={share_mode}",
        flush=True,
    )
    if use_gradio_share:
        print(
            "[t4] Đang tạo Gradio share (*.gradio.live) — chờ vài chục giây…",
            flush=True,
        )

    launch_kwargs = dict(
        share=use_gradio_share,
        server_port=port,
        server_name="0.0.0.0",
        show_error=True,
    )
    try:
        out = demo.launch(**launch_kwargs)
    except TypeError:
        out = demo.launch(
            share=use_gradio_share,
            server_port=port,
            server_name="0.0.0.0",
        )

    local_url = share_url = None
    if isinstance(out, (tuple, list)):
        if len(out) >= 2:
            local_url = out[1]
        if len(out) >= 3:
            share_url = out[2]
    if local_url:
        print(f"[t4] Local URL: {local_url}", flush=True)
    if share_url:
        print(f"[t4] Gradio Public URL: {share_url}", flush=True)
    elif use_gradio_share:
        print(
            "[t4] Chưa lấy share_url từ return — xem dòng 'Running on public URL' phía trên.",
            flush=True,
        )

    if use_ngrok:
        _start_ngrok(port)

    print(
        "[t4] App đang chạy. Dừng: Ctrl+C / Interrupt kernel (■).",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gradio Colab T4 — fp16_disk_xformers + EditCache (Drive-first)"
    )
    parser.add_argument("--drive-fp16", type=Path, default=DEFAULT_DRIVE_FP16)
    parser.add_argument("--drive-fp32", type=Path, default=DEFAULT_DRIVE_FP32)
    parser.add_argument("--local-fp16", type=Path, default=LOCAL_FP16)
    parser.add_argument("--local-fp32", type=Path, default=LOCAL_FP32)
    parser.add_argument(
        "--allow-download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Nếu Drive/local thiếu → tải Qualcomm (mặc định: có)",
    )
    parser.add_argument(
        "--convert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cho phép convert fp32→fp16 trên Colab (mặc định: có)",
    )
    parser.add_argument(
        "--save-to-drive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sau tải/convert → copy weights lên Drive (mặc định: có)",
    )
    parser.add_argument(
        "--share",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Bật Gradio share (mặc định Colab: bật). Ghi đè --share-mode nếu False.",
    )
    parser.add_argument(
        "--share-mode",
        choices=("gradio", "ngrok", "both", "none"),
        default=None,
        help="Cách public URL: gradio | ngrok | both | none (mặc định: gradio trên Colab)",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--selftest",
        type=Path,
        default=None,
        help="Chạy 1 edit rồi thoát (không mở server)",
    )
    args = parser.parse_args()
    _import_runtime()

    allow_download = bool(args.allow_download)
    save_to_drive = bool(args.save_to_drive)
    allow_convert = bool(args.convert)

    # Resolve share_mode
    share_mode = args.share_mode
    if share_mode is None:
        env_mode = os.environ.get("SWIFTEDIT_SHARE_MODE", "").strip().lower()
        if env_mode in ("gradio", "ngrok", "both", "none"):
            share_mode = env_mode
        elif _in_colab():
            share_mode = "gradio"
        else:
            share_mode = "none"
    if args.share is False:
        share_mode = "none"
    elif args.share is True and share_mode == "none":
        share_mode = "gradio"

    _step(1, 4, "Kiểm tra CUDA")
    ensure_cuda()
    _log_vram()

    _step(2, 4, "Mount Drive (nếu cần) + kiểm tra / chuẩn bị weights")
    mount_drive_if_needed(args.drive_fp16, args.drive_fp32)
    weights = prepare_fp16_disk_weights(
        drive_fp16=args.drive_fp16,
        drive_fp32=args.drive_fp32,
        local_fp16=args.local_fp16,
        local_fp32=args.local_fp32,
        allow_download=allow_download,
        allow_convert=allow_convert,
        save_to_drive=save_to_drive,
    )
    print(f"[t4] Weights sẵn sàng → {weights}", flush=True)

    _step(3, 4, "Load model lên GPU (VRAM phải tăng)")
    demo, run_edit, _run_removal = build_app(weights)
    print("[t4] Load model OK — mới được phép launch app.", flush=True)

    if args.selftest is not None:
        img, info = run_edit(
            str(args.selftest),
            "a slanted mountain bicycle on the road in front of a building",
            "a slanted rusty mountain motorcycle in front of a fence",
            0.2,
            1.0,
            0.5,
            True,
        )
        out = ROOT / "results" / "app_t4_xformers_selftest.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        print(f"[selftest] OK → {out}\n{info}", flush=True)
        return 0

    _step(4, 4, f"Launch Gradio + share ({share_mode})")
    return _launch_gradio(demo, port=args.port, share_mode=share_mode)


if __name__ == "__main__":
    raise SystemExit(main())
