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
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

EXAMPLE_PROMPTS = [
    ["a slanted mountain bicycle on the road in front of a building",
     "a slanted rusty mountain motorcycle in front of a fence"],
]


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
    return Image.fromarray(bg[..., :3].astype(np.uint8)), mask


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
        hit = use_cache and active_cache is not None and active_cache._img_path == image_path \
            and active_cache._src_p == src_p

        _sync(device)
        t0 = time.perf_counter()
        res = edit_image(
            image_path, src_p or "", edit_p,
            inverse_model, aux_model, ip_sb_model,
            scale_edit=scale_edit, scale_non_edit=scale_non_edit,
            mask_threshold=mask_threshold, cache=active_cache,
        )
        _sync(device)
        dt = time.perf_counter() - t0

        note = "cache hit (cùng ảnh + source prompt)" if hit else (
            "cache nạp mới" if use_cache else "không dùng cache")
        info = (
            f"**Thời gian:** {dt:.2f}s  \n"
            f"**Thiết bị:** `{device}` | **dtype:** `{dtype}`"
            f"{' + channels_last' if channels_last else ''} (VAE fp32)  \n"
            f"**Cache:** {note}"
        )
        return tensor_to_pil(res), info

    def run_removal(editor_value, src_p, edit_p, scale_edit, scale_non_edit, mask_threshold):
        bg, mask = extract_editor_mask(editor_value)
        if bg is None:
            raise gr.Error("Vui lòng tải ảnh và khoanh vùng vật thể cần xóa.")
        if mask is None or mask.sum() < 1:
            raise gr.Error("Chưa khoanh vùng nào. Hãy tô lên vật thể cần xóa.")
        # edit_image cần đường dẫn file -> lưu ảnh nền ra temp.
        tmp = Path(tempfile.mkdtemp()) / "removal_src.png"
        bg.resize((512, 512)).save(tmp)

        _sync(device)
        t0 = time.perf_counter()
        res = edit_image(
            str(tmp), src_p or "", edit_p or "empty background",
            inverse_model, aux_model, ip_sb_model,
            scale_edit=scale_edit, scale_non_edit=scale_non_edit,
            mask_threshold=mask_threshold, user_mask=mask,
        )
        _sync(device)
        dt = time.perf_counter() - t0

        # Ảnh mask preview (resize về 512)
        mask_prev = Image.fromarray((mask * 255).astype(np.uint8)).resize((512, 512))
        info = (
            f"**Thời gian:** {dt:.2f}s  \n"
            f"**Thiết bị:** `{device}` | **dtype:** `{dtype}`"
            f"{' + channels_last' if channels_last else ''} (VAE fp32)  \n"
            f"**Tỉ lệ vùng khoanh:** {100*mask.mean():.1f}% ảnh  \n"
            f"_Lưu ý: SwiftEdit là editor ngữ nghĩa, xóa tốt vật nhỏ/vừa; "
            f"vật lớn chiếm phần lớn khung có thể còn sót._"
        )
        return tensor_to_pil(res), mask_prev, info

    with gr.Blocks(title="SwiftEdit-RT Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# SwiftEdit-RT — Chỉnh sửa & xóa vật thể bằng prompt (one-step)\n"
            f"Inference tăng tốc bằng **fp16 + channels_last + cache** "
            f"(đang chạy `{dtype}` trên `{device}`)."
        )
        with gr.Tabs():
            with gr.Tab("Chỉnh sửa bằng prompt"):
                with gr.Row():
                    with gr.Column(scale=1):
                        inp_image = gr.Image(label="Ảnh nguồn", type="filepath", height=320)
                        inp_src = gr.Textbox(label="Source prompt (mô tả ảnh gốc)",
                                             placeholder="vd: a mountain bicycle in front of a building")
                        inp_edit = gr.Textbox(label="Edit prompt (thay đổi mong muốn)",
                                              placeholder="vd: a rusty motorcycle in front of a fence")
                        with gr.Accordion("Tùy chọn nâng cao", open=False):
                            sl_edit = gr.Slider(0.0, 1.0, value=0.2, step=0.05,
                                                label="scale_edit (vùng chỉnh sửa)")
                            sl_non = gr.Slider(0.0, 2.0, value=1.0, step=0.05,
                                               label="scale_non_edit (giữ nền)")
                            sl_mask = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                                label="mask_threshold")
                            cb_cache = gr.Checkbox(value=True,
                                                   label="Dùng cache (nhanh khi đổi edit prompt cùng ảnh)")
                        btn = gr.Button("Chỉnh sửa ảnh", variant="primary")
                    with gr.Column(scale=1):
                        out_image = gr.Image(label="Ảnh kết quả", height=320)
                        out_info = gr.Markdown()
                gr.Examples(
                    examples=[[p[0], p[1]] for p in EXAMPLE_PROMPTS],
                    inputs=[inp_src, inp_edit],
                    label="Ví dụ prompt",
                )
                btn.click(
                    run_edit,
                    inputs=[inp_image, inp_src, inp_edit, sl_edit, sl_non, sl_mask, cb_cache],
                    outputs=[out_image, out_info],
                )

            with gr.Tab("Xóa vật thể (khoanh vùng)"):
                gr.Markdown(
                    "Tải ảnh, **dùng cọ tô lên vật thể cần xóa**, rồi mô tả ảnh gốc + nền "
                    "sau khi xóa. Xóa tốt nhất với vật **nhỏ/vừa**; vật rất lớn có thể còn sót."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        rm_editor = gr.ImageEditor(
                            label="Khoanh vùng vật thể cần xóa (dùng cọ)",
                            type="numpy", height=360,
                            brush=gr.Brush(colors=["#ff0000"], default_size=28),
                            layers=False, transforms=[],
                        )
                        rm_src = gr.Textbox(label="Mô tả ảnh gốc (source prompt)",
                                            placeholder="vd: a bicycle on the road in front of a building")
                        rm_edit = gr.Textbox(
                            label="Mô tả vùng sau khi xóa (nền thay thế)",
                            placeholder="vd: empty asphalt road in front of a building",
                            value="empty background",
                        )
                        with gr.Accordion("Tùy chọn nâng cao", open=False):
                            rm_sl_edit = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                                                   label="scale_edit (0 = bỏ giữ vật thể)")
                            rm_sl_non = gr.Slider(0.0, 2.0, value=1.2, step=0.05,
                                                  label="scale_non_edit (giữ nền)")
                            rm_sl_mask = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                                   label="mask_threshold")
                        rm_btn = gr.Button("Xóa vật thể", variant="primary")
                    with gr.Column(scale=1):
                        rm_out = gr.Image(label="Ảnh sau khi xóa", height=320)
                        rm_mask = gr.Image(label="Mask vùng khoanh", height=200)
                        rm_info = gr.Markdown()
                rm_btn.click(
                    run_removal,
                    inputs=[rm_editor, rm_src, rm_edit, rm_sl_edit, rm_sl_non, rm_sl_mask],
                    outputs=[rm_out, rm_mask, rm_info],
                )
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
            0.0, 1.2, 0.5,
        )
        out = ROOT / "results" / "app_removal_selftest.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        print(f"[selftest-removal] OK -> {out}\n{info}")
        return 0

    demo.launch(share=args.share, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
