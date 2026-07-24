#!/usr/bin/env python3
"""Demo Gradio cho Google Colab T4 — config `fp16_disk_xformers` + EditCache.

Khác `app_gradio.py` (Mac MPS / fp16 compute trên weights FP32):
  - Bắt buộc CUDA (T4)
  - Weights **FP16 trên disk** (`swiftedit_weights_fp16`)
  - Bật **xFormers Memory-Efficient Attention**
  - Ưu tiên **Google Drive**; thiếu thì tải Qualcomm (+ convert fp16 nếu cần)
  - Mặc định **lưu lại Drive** sau tải/convert (`--save-to-drive`)

Chạy trên Colab (sau khi clone repo + GPU T4):
  from google.colab import drive
  drive.mount('/content/drive')
  !pip install -q 'gradio>=5,<6' 'huggingface-hub<1.0' xformers
  !python scripts/app_gradio_t4_xformers.py --share

Tùy chọn:
  python scripts/app_gradio_t4_xformers.py \\
      --drive-fp16 /content/drive/MyDrive/CS2309/swiftedit_weights_fp16 \\
      --share --port 7860
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
print("[t4] Đang import torch/gradio/xformers (có thể 30–90s, VRAM chưa tăng)…", flush=True)

import gradio as gr
import numpy as np
import torch
from PIL import Image

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
    if not torch.cuda.is_available():
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
        hints.append("source prompt quá ngắn — mô tả toàn bộ ảnh")
    if edit.lower() in _VAGUE_EDIT_PROMPTS:
        hints.append("edit prompt quá chung — mô tả nền thay thế cụ thể")
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

    def run_edit(image_path, src_p, edit_p, scale_edit, scale_non_edit, mask_threshold, use_cache):
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

    def run_removal(editor_value, src_p, edit_p, scale_edit, scale_non_edit, mask_threshold):
        bg_img, mask = extract_editor_mask(editor_value)
        if bg_img is None:
            raise gr.Error("Vui lòng tải ảnh và khoanh vùng cần xóa.")
        if mask is None or mask.sum() < 1:
            raise gr.Error("Chưa tô mask — dùng cọ khoanh vật thể cần xóa.")
        if not edit_p or not edit_p.strip():
            raise gr.Error("Vui lòng nhập mô tả nền sau khi xóa.")

        boxed, lb_meta = _letterbox_image(bg_img, EDIT_SIZE)
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
            f"{hints}"
        )
        return _unletterbox(tensor_to_pil(res), lb_meta), mask_prev, info

    with gr.Blocks(title="SwiftEdit T4 — fp16_disk_xformers", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# SwiftEdit — Colab T4 (`fp16_disk_xformers`)\n"
            "FP16 **disk** + **xFormers MEA** + **EditCache**. "
            "Weights ưu tiên Google Drive; thiếu thì tải về.\n\n"
            f"{runtime_banner}"
        )
        with gr.Tabs():
            with gr.Tab("Chỉnh sửa bằng prompt"):
                with gr.Row():
                    with gr.Column(scale=1):
                        inp_image = gr.Image(label="Ảnh nguồn", type="filepath", height=320)
                        inp_src = gr.Textbox(
                            label="Source prompt",
                            placeholder="vd: a mountain bicycle on the road",
                        )
                        inp_edit = gr.Textbox(
                            label="Edit prompt",
                            placeholder="vd: a rusty motorcycle on the road",
                        )
                        with gr.Accordion("Tùy chọn", open=False):
                            sl_edit = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="scale_edit")
                            sl_non = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="scale_non_edit")
                            sl_mask = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="mask_threshold")
                            use_cache = gr.Checkbox(value=True, label="Dùng EditCache")
                        btn = gr.Button("Chỉnh sửa", variant="primary")
                    with gr.Column(scale=1):
                        out_img = gr.Image(label="Kết quả", height=320)
                        out_info = gr.Markdown()
                gr.Examples(
                    examples=[[p[0], p[1]] for p in EXAMPLE_PROMPTS],
                    inputs=[inp_src, inp_edit],
                    label="Ví dụ prompt",
                )
                btn.click(
                    run_edit,
                    inputs=[inp_image, inp_src, inp_edit, sl_edit, sl_non, sl_mask, use_cache],
                    outputs=[out_img, out_info],
                )

            with gr.Tab("Xóa vật thể (khoanh vùng)"):
                gr.Markdown("Tải ảnh → **cọ tô** vật cần xóa → mô tả nền thay thế.")
                with gr.Row():
                    with gr.Column(scale=1):
                        rm_editor = gr.ImageEditor(
                            label="Khoanh vùng cần xóa",
                            type="numpy",
                            height=360,
                            brush=gr.Brush(
                                colors=["rgba(255, 0, 0, 0.5)"],
                                default_size=28,
                            ),
                            layers=False,
                            transforms=[],
                        )
                        rm_src = gr.Textbox(label="Source prompt (ảnh gốc)")
                        rm_edit = gr.Textbox(label="Edit prompt (nền sau khi xóa)")
                        with gr.Accordion("Tùy chọn", open=False):
                            rm_sl_edit = gr.Slider(
                                0.0, 1.0, value=0.15, step=0.05, label="scale_edit"
                            )
                            rm_sl_non = gr.Slider(
                                0.0, 2.0, value=1.2, step=0.05, label="scale_non_edit"
                            )
                            rm_sl_mask = gr.Slider(
                                0.0, 1.0, value=0.5, step=0.05, label="mask_threshold"
                            )
                        rm_btn = gr.Button("Xóa vật thể", variant="primary")
                    with gr.Column(scale=1):
                        rm_out = gr.Image(label="Ảnh sau khi xóa", height=320)
                        rm_mask = gr.Image(label="Mask vùng khoanh", height=200)
                        rm_info = gr.Markdown()
                gr.Examples(
                    examples=[[p[0], p[1]] for p in REMOVAL_EXAMPLE_PROMPTS],
                    inputs=[rm_src, rm_edit],
                    label="Ví dụ prompt xóa",
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
