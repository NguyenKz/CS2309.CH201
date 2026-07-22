"""State và compositing cho workflow chỉnh ảnh nhiều lượt.

Module này không phụ thuộc Gradio/model để có thể unit-test nhanh.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


GLOBAL_MASK_THRESHOLD = 0.45


@dataclass(frozen=True)
class SquareROI:
    x: int
    y: int
    size: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.size, self.y + self.size


@dataclass
class Candidate:
    image: Image.Image
    model_image: Image.Image
    mask: Image.Image
    clean_latent: Any
    seed: int
    mode: str
    source_prompt: str
    edit_prompt: str


@dataclass
class EditSnapshot:
    image: Image.Image
    source_prompt: str
    clean_latent: Any
    turn: int


@dataclass
class EditSession:
    source_key: str
    original: Image.Image
    master: Image.Image
    source_prompt: str = ""
    clean_latent: Any = None
    turn: int = 0
    batch_index: int = 0
    candidates: list[Candidate] = field(default_factory=list)
    history: list[EditSnapshot] = field(default_factory=list)


def image_source_key(image_path: str | Path) -> str:
    path = Path(image_path)
    stat = path.stat()
    raw = f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode()
    return hashlib.sha256(raw).hexdigest()


def parse_square_roi(value: str, image_size: tuple[int, int]) -> SquareROI:
    """Đọc ROI do canvas gửi và clamp thành box vuông hợp lệ trong ảnh."""
    if not value:
        raise ValueError("Chưa chọn vùng vuông trên ảnh hiện tại.")
    try:
        data = json.loads(value)
        x = int(round(float(data["x"])))
        y = int(round(float(data["y"])))
        size = int(round(float(data["size"])))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Tọa độ vùng vuông không hợp lệ; hãy vẽ lại.") from exc
    width, height = image_size
    size = min(size, width, height)
    if size < 16:
        raise ValueError("Vùng chọn quá nhỏ; cạnh box cần ít nhất 16 px.")
    x = min(max(0, x), width - size)
    y = min(max(0, y), height - size)
    return SquareROI(x=x, y=y, size=size)


def square_roi_from_mask(
    mask: np.ndarray,
    padding_ratio: float = 0.25,
    min_size: int = 64,
) -> SquareROI:
    """Tạo crop vuông bao quanh mask, có context padding và clamp trong ảnh."""
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask.squeeze()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Chưa tô mask trên ảnh hiện tại.")
    height, width = mask.shape
    box_w = int(xs.max() - xs.min() + 1)
    box_h = int(ys.max() - ys.min() + 1)
    content_side = max(box_w, box_h)
    padding = int(round(content_side * max(0.0, float(padding_ratio))))
    side = max(int(min_size), content_side + 2 * padding)
    side = min(side, width, height)
    center_x = (float(xs.min()) + float(xs.max()) + 1) / 2
    center_y = (float(ys.min()) + float(ys.max()) + 1) / 2
    x = int(round(center_x - side / 2))
    y = int(round(center_y - side / 2))
    x = min(max(0, x), width - side)
    y = min(max(0, y), height - side)
    return SquareROI(x=x, y=y, size=side)


def crop_mask(mask: np.ndarray, roi: SquareROI) -> Image.Image:
    array = np.asarray(mask, dtype=np.float32)
    if array.ndim > 2:
        array = array.squeeze()
    cropped = array[roi.y : roi.y + roi.size, roi.x : roi.x + roi.size]
    return Image.fromarray(
        (np.clip(cropped, 0, 1) * 255).astype(np.uint8),
        mode="L",
    )


def crop_square(image: Image.Image, roi: SquareROI) -> Image.Image:
    return image.convert("RGB").crop(roi.box)


def paste_square(master: Image.Image, patch: Image.Image, roi: SquareROI) -> Image.Image:
    result = master.convert("RGB").copy()
    if patch.size != (roi.size, roi.size):
        patch = patch.resize((roi.size, roi.size), Image.Resampling.LANCZOS)
    result.paste(patch, (roi.x, roi.y))
    return result


def model_mask_to_roi(
    mask: np.ndarray,
    roi: SquareROI,
    image_size: tuple[int, int],
) -> tuple[Image.Image, Image.Image]:
    """Resize mask latent về crop và đặt mask đó vào canvas full-resolution."""
    mask = np.asarray(mask, dtype=np.float32).squeeze()
    crop_mask = Image.fromarray(
        (np.clip(mask, 0, 1) * 255).astype(np.uint8),
        mode="L",
    ).resize((roi.size, roi.size), Image.Resampling.NEAREST)
    full_mask = Image.new("L", image_size, 0)
    full_mask.paste(crop_mask, (roi.x, roi.y))
    return crop_mask, full_mask


def new_session(image_path: str | Path, source_prompt: str = "") -> EditSession:
    image = Image.open(image_path).convert("RGB")
    return EditSession(
        source_key=image_source_key(image_path),
        original=image.copy(),
        master=image.copy(),
        source_prompt=source_prompt.strip(),
    )


def ensure_session(
    session: EditSession | None,
    image_path: str | Path,
    source_prompt: str = "",
) -> EditSession:
    key = image_source_key(image_path)
    if session is None or session.source_key != key:
        return new_session(image_path, source_prompt)
    return session


def model_mask_to_original(mask: np.ndarray, letterbox_meta: dict) -> Image.Image:
    """Ánh xạ mask canvas model về đúng kích thước ảnh master."""
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim > 2:
        mask = mask.squeeze()
    model_h, model_w = mask.shape
    left, top = letterbox_meta["pad"]
    content_w, content_h = letterbox_meta["content_size"]
    original_size = tuple(letterbox_meta["orig_size"])
    canvas_size = max(
        left * 2 + content_w,
        top * 2 + content_h,
    )
    scale_x = model_w / canvas_size
    scale_y = model_h / canvas_size
    mask_left = int(round(left * scale_x))
    mask_top = int(round(top * scale_y))
    mask_right = int(round((left + content_w) * scale_x))
    mask_bottom = int(round((top + content_h) * scale_y))
    cropped = mask[
        max(0, mask_top) : min(model_h, mask_bottom),
        max(0, mask_left) : min(model_w, mask_right),
    ]
    pil_mask = Image.fromarray((np.clip(cropped, 0, 1) * 255).astype(np.uint8), mode="L")
    return pil_mask.resize(original_size, Image.Resampling.NEAREST)


def feather_mask(mask: Image.Image, dilation: int = 5, blur: float = 2.0) -> Image.Image:
    """Nới mask và làm mềm mép để ghép vùng edit ít lộ biên."""
    result = mask.convert("L")
    if dilation > 0:
        kernel = max(3, 2 * int(dilation) + 1)
        result = result.filter(ImageFilter.MaxFilter(kernel))
    if blur > 0:
        result = result.filter(ImageFilter.GaussianBlur(float(blur)))
    return result


def resolve_edit_mode(requested_mode: str, mask_coverage: float) -> str:
    if requested_mode in {"local", "global"}:
        return requested_mode
    return "global" if mask_coverage > GLOBAL_MASK_THRESHOLD else "local"


def hybrid_composite(
    master: Image.Image,
    edited: Image.Image,
    mask: Image.Image,
    mode: str,
    dilation: int = 5,
    blur: float = 2.0,
) -> Image.Image:
    """Ghép local edit; global edit thay toàn frame theo đúng kích thước master."""
    master = master.convert("RGB")
    edited = edited.convert("RGB")
    if edited.size != master.size:
        edited = edited.resize(master.size, Image.Resampling.LANCZOS)
    if mode == "global":
        return edited
    alpha = feather_mask(mask, dilation=dilation, blur=blur)
    if alpha.size != master.size:
        alpha = alpha.resize(master.size, Image.Resampling.NEAREST)
    return Image.composite(edited, master, alpha)


def set_candidates(session: EditSession, candidates: list[Candidate]) -> EditSession:
    session.candidates = candidates
    session.batch_index += 1
    return session


def commit_candidate(session: EditSession, index: int) -> EditSession:
    if not 0 <= index < len(session.candidates):
        raise IndexError("Candidate không tồn tại; hãy tạo hoặc regen trước.")
    candidate = session.candidates[index]
    session.history.append(
        EditSnapshot(
            image=session.master.copy(),
            source_prompt=session.source_prompt,
            clean_latent=session.clean_latent,
            turn=session.turn,
        )
    )
    session.master = candidate.image.copy()
    session.source_prompt = candidate.edit_prompt
    session.clean_latent = candidate.clean_latent
    session.turn += 1
    session.candidates = []
    return session


def undo_session(session: EditSession) -> EditSession:
    if not session.history:
        return session
    previous = session.history.pop()
    session.master = previous.image
    session.source_prompt = previous.source_prompt
    session.clean_latent = previous.clean_latent
    session.turn = previous.turn
    session.candidates = []
    return session
