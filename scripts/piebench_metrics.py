"""PIE-Bench metrics (tương thích PnPInversion evaluate.py, bỏ structure_distance/DINO)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class PieBenchMetrics:
    def __init__(self, device: str | torch.device) -> None:
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.multimodal import CLIPScore
        from torchmetrics.regression import MeanSquaredError

        self.device = torch.device(device)
        self.clip = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.mse = MeanSquaredError().to(self.device)

    @staticmethod
    def _to_tensor(img: Image.Image | np.ndarray, device: torch.device) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.tensor(arr).permute(2, 0, 1).to(device)

    def _masked_pair(
        self,
        src: Image.Image,
        tgt: Image.Image,
        mask_pred: np.ndarray | None,
        mask_gt: np.ndarray | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_arr = np.array(src).astype(np.float32) / 255.0
        tgt_arr = np.array(tgt).astype(np.float32) / 255.0
        if mask_pred is not None:
            src_arr = src_arr * np.array(mask_pred).astype(np.float32)
        if mask_gt is not None:
            tgt_arr = tgt_arr * np.array(mask_gt).astype(np.float32)
        src_t = torch.tensor(src_arr).permute(2, 0, 1).to(self.device)
        tgt_t = torch.tensor(tgt_arr).permute(2, 0, 1).to(self.device)
        return src_t, tgt_t

    def psnr_unedit(self, src: Image.Image, tgt: Image.Image, edit_mask: np.ndarray) -> float:
        bg = 1.0 - edit_mask
        if bg.sum() == 0:
            return float("nan")
        bg3 = bg[:, :, np.newaxis].repeat(3, axis=2)
        src_t, tgt_t = self._masked_pair(src, tgt, bg3, bg3)
        return float(
            self.psnr(
                src_t.unsqueeze(0),
                tgt_t.unsqueeze(0),
            ).cpu()
        )

    def mse_unedit(self, src: Image.Image, tgt: Image.Image, edit_mask: np.ndarray) -> float:
        bg = 1.0 - edit_mask
        if bg.sum() == 0:
            return float("nan")
        bg3 = bg[:, :, np.newaxis].repeat(3, axis=2)
        src_t, tgt_t = self._masked_pair(src, tgt, bg3, bg3)
        return float(self.mse(src_t.contiguous(), tgt_t.contiguous()).cpu())

    def clip_whole(self, img: Image.Image, prompt: str) -> float:
        arr = np.array(img)
        t = torch.tensor(arr).permute(2, 0, 1).to(self.device)
        return float(self.clip(t, prompt).detach().cpu())

    def clip_edited(self, img: Image.Image, prompt: str, edit_mask: np.ndarray) -> float:
        if edit_mask.ndim == 3:
            m = edit_mask[:, :, 0]
        else:
            m = edit_mask
        if m.sum() == 0:
            return float("nan")
        arr = np.uint8(np.array(img) * m[:, :, np.newaxis].repeat(3, axis=2))
        t = torch.tensor(arr).permute(2, 0, 1).to(self.device)
        return float(self.clip(t, prompt).detach().cpu())

    def evaluate_edit(
        self,
        src_image: Image.Image,
        tgt_image: Image.Image,
        edit_mask: np.ndarray,
        src_prompt: str,
        edit_prompt: str,
    ) -> dict[str, float]:
        size = (512, 512)
        src_image = src_image.convert("RGB").resize(size)
        tgt_image = tgt_image.convert("RGB").resize(size)
        if edit_mask.shape[:2] != size[::-1]:
            from PIL import Image as PILImage

            edit_mask = np.array(
                PILImage.fromarray((edit_mask * 255).astype(np.uint8)).resize(size)
            ).astype(np.float32)
            edit_mask = (edit_mask > 127).astype(np.float32)
        return {
            "psnr_unedit": self.psnr_unedit(src_image, tgt_image, edit_mask),
            "mse_unedit": self.mse_unedit(src_image, tgt_image, edit_mask),
            "clip_whole": self.clip_whole(tgt_image, edit_prompt),
            "clip_edited": self.clip_edited(tgt_image, edit_prompt, edit_mask),
        }
