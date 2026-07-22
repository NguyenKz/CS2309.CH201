# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os, time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from models import *
from timing import StageTimer


def prepare_user_mask(user_mask, out_hw, device, threshold=0.5):
    """Chuẩn bị mask người dùng vẽ thành tensor nhị phân [H,W] ở độ phân giải latent.

    user_mask: ndarray/tensor 2D (H,W) hoặc 3D (H,W,C) giá trị {0..255} hoặc [0,1].
    Vùng được vẽ (giá trị > threshold) = 1 = vùng sẽ chỉnh sửa/xóa.
    """
    m = torch.as_tensor(np.asarray(user_mask), device=device, dtype=torch.float32)
    if m.dim() == 3:
        m = m[..., -1] if m.shape[-1] in (1, 2, 4) else m.mean(dim=-1)  # alpha nếu RGBA
    if m.max() > 1.0:
        m = m / 255.0
    m = F.interpolate(m[None, None], size=out_hw, mode="bilinear", align_corners=False)[0, 0]
    return (m > threshold).to(torch.float32)

#
# Configure this path to where you have stored the local copy of the weights:
#
SWIFTEDIT_WEIGHTS_ROOT = 'swiftedit_weights'


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_binary(pix, threshold=0.5):
    if float(pix) > threshold:
        return 1.0
    return 0.0


def make_generator(device, seed: int) -> torch.Generator:
    """Generator cho VAE sample — MPS dùng CPU generator (ổn định hơn)."""
    gen_device = "cpu" if str(device).startswith("mps") else device
    gen = torch.Generator(device=gen_device)
    gen.manual_seed(int(seed))
    return gen


def apply_job_seed(seed: int | None, device) -> torch.Generator | None:
    """Cố định RNG cho một job eval (None = giữ hành vi cũ không seed)."""
    if seed is None:
        return None
    torch.manual_seed(int(seed))
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    return make_generator(device, int(seed))


class EditCache:
    """Cache các tensor chỉ phụ thuộc ảnh nguồn / source prompt.

    Dùng cho kịch bản realtime: cùng 1 ảnh + source prompt, đổi nhiều edit prompt.
    Tự invalidate khi đổi ảnh (latent + CLIP image embed) hoặc đổi source prompt
    (source text embed của inverse model + generation model).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._img_path = None
        self._src_p = None
        self.latents = None              # VAE encode ảnh nguồn (image-only)
        self.src_inv_embed = None        # source prompt embed — inverse model
        self.gen_embed_cache: dict = {}  # cho gen_img: image_prompt_embeds + src_text_embed

    def ensure(self, img_path, src_p) -> None:
        """Invalidate phần cache không còn hợp lệ khi đổi ảnh / source prompt."""
        if img_path != self._img_path:
            self._img_path = img_path
            self.latents = None
            self.gen_embed_cache.pop("image_prompt_embeds", None)
        if src_p != self._src_p:
            self._src_p = src_p
            self.src_inv_embed = None
            self.gen_embed_cache.pop("src_text_embed", None)


@torch.no_grad()
def edit_image(
    img_path,
    src_p,
    edit_p,
    inverse_model,
    aux_model,
    ip_sb_model,
    scale_ta=1,
    scale_edit=0.2,
    scale_non_edit=1,
    clamp_rate=3.0,
    mask_threshold=0.5,
    cache=None,
    user_mask=None,
    seed=None,
    source_latent=None,
    latent_jitter_strength=0.0,
    return_details=False,
):
    """
        Save keysteps to file.
            + img_path: path to the source image.
            + src_p: Source Prompt that describes source image (could leave it empty).
            + edit_p: Edit Prompt that describes your desired changes.
            + cache: EditCache tùy chọn — tái dùng latent/embedding khi cùng ảnh+source prompt.
            + user_mask: mask người dùng vẽ (2D/3D). Nếu có, ghi đè self-guided mask —
              dùng cho xóa/chỉnh vật thể theo vùng khoanh tay.
            + seed: RNG cố định cho job (VAE encode .sample). None = không set (legacy).
            + source_latent: latent sạch đã scale từ lượt trước; bỏ VAE encode khi có.
            + latent_jitter_strength: nhiễu nhỏ có seed để tạo candidate từ source_latent.
            + return_details: trả dict gồm image, clean_latent và mask cho multi-turn app.
    """
    device = inverse_model.device
    vae_gen = apply_job_seed(seed, device)
    timer = StageTimer(device, label=f"{src_p}->{edit_p}")
    mid_timestep = torch.ones((1,), dtype=torch.int64, device=device) * 500
    final_timestep = torch.ones((1,), dtype=torch.int64, device=device) * 999

    if cache is not None:
        cache.ensure(img_path, src_p)

    # Input Image
    pil_img_cond = Image.open(img_path).resize((512, 512))

    processed_image = to_tensor(pil_img_cond).unsqueeze(0).to(device) * 2 - 1

    # Predict inverted noise — latent VAE chỉ phụ thuộc ảnh nguồn -> cache được
    with timer.stage("vae_encode"):
        if source_latent is not None:
            latents = source_latent.to(device=device, dtype=inverse_model.vae.dtype)
            if latent_jitter_strength > 0:
                # MPS dùng CPU generator để seed ổn định; tạo noise CPU rồi chuyển device.
                jitter_device = "cpu" if str(device).startswith("mps") else latents.device
                jitter = torch.randn(
                    latents.shape,
                    generator=vae_gen,
                    device=jitter_device,
                    dtype=latents.dtype,
                ).to(latents.device)
                latents = latents + float(latent_jitter_strength) * jitter
        elif cache is not None and cache.latents is not None:
            latents = cache.latents
        else:
            latents = inverse_model.vae.encode(
                processed_image.to(inverse_model.vae.dtype)
            ).latent_dist.sample(generator=vae_gen)
            latents = latents * inverse_model.vae.config.scaling_factor
            if cache is not None:
                cache.latents = latents
        dub_latents = torch.cat([latents] * 2, dim=0)

    with timer.stage("inv_text_encode"):
        if cache is not None:
            # Source prompt embed cache được; chỉ encode lại edit prompt.
            if cache.src_inv_embed is None:
                src_id = tokenize_captions(inverse_model.tokenizer, [src_p]).to(device)
                cache.src_inv_embed = inverse_model.text_encoder(src_id)[0]
            edit_id = tokenize_captions(inverse_model.tokenizer, [edit_p]).to(device)
            edit_embed = inverse_model.text_encoder(edit_id)[0]
            encoder_hidden_state = torch.cat([cache.src_inv_embed, edit_embed], dim=0).to(
                dtype=inverse_model.weight_dtype
            )
        else:
            input_id = tokenize_captions(inverse_model.tokenizer, [src_p, edit_p]).to(device)
            encoder_hidden_state = inverse_model.text_encoder(input_id)[0].to(
                dtype=inverse_model.weight_dtype
            )

    with timer.stage("unet_inverse"):
        # compute_dtype thay cho .dtype (sai sau khi nén 4-bit weight-only).
        unet_inv_dtype = getattr(inverse_model, "compute_dtype", None) or inverse_model.unet_inverse.dtype
        predict_inverted_code = inverse_model.unet_inverse(
            dub_latents.to(unet_inv_dtype), mid_timestep, encoder_hidden_state.to(unet_inv_dtype)
        ).sample.float()  # về fp32 cho mask + input_sb (alpha_t/sigma_t fp32) ổn định

    # Estimate editing mask
    with timer.stage("mask_estimate"):
        inverted_noise_1, inverted_noise_2 = predict_inverted_code.chunk(2)
        subed = (inverted_noise_1 - inverted_noise_2).abs_().mean(dim=[0, 1])
        max_v = (subed.mean() * clamp_rate).item()
        mask12 = subed.clamp(0, max_v) / max_v
        # Nhị phân hóa vectorized ngay trên device (thay .cpu().apply_() — vòng lặp Python từng pixel)
        mask12 = (mask12 > mask_threshold).to(dtype=subed.dtype)
        # Ghi đè bằng mask người dùng khoanh (xóa/chỉnh vật thể theo vùng) nếu có.
        if user_mask is not None:
            mask12 = prepare_user_mask(
                user_mask, mask12.shape, device, mask_threshold
            ).to(dtype=subed.dtype)

    # Edit images
    input_sb = ip_sb_model.alpha_t * latents + ip_sb_model.sigma_t * inverted_noise_1
    mask_controller = MaskController(
        mask12, scale_text_hiddenstate=scale_ta, scale_ip_fg=scale_edit, scale_ip_bg=scale_non_edit
    )
    ip_sb_model.set_controller(mask_controller, where=["mid_blocks", "up_blocks"])
    gen_result = ip_sb_model.gen_img(
        pil_image=pil_img_cond,
        prompts=[src_p, edit_p],
        noise=input_sb,
        return_noise_image=False,
        return_clean_latent=return_details,
        timer=timer,
        embed_cache=(cache.gen_embed_cache if cache is not None else None),
    )
    if return_details:
        res_gen_img, _, clean_latent = gen_result
    else:
        res_gen_img, _ = gen_result

    timer.dump(extra={"img_path": img_path})

    if return_details:
        return {
            "image": res_gen_img,
            "clean_latent": clean_latent.detach(),
            "mask": mask12.detach(),
            "source_latent": latents.detach(),
        }
    return res_gen_img


if __name__ == "__main__":

    device = get_device()
    print(f"Using device: {device}")

    # Define model
    inverse_ckpt = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "inverse_ckpt-120k")
    inverse_model = InverseModel(inverse_ckpt, device=device)
    aux_model = AuxiliaryModel(device=device)

    path_unet_sb = (os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "sbv2_0.5"))
    ip_ckpt = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "ip_adapter_ckpt-90k/ip_adapter.bin")
    ip_sb_model = IPSBV2Model(
        path_unet_sb, ip_ckpt, aux_model, device=device, with_ip_mask_controller=True
    )

    # Input — ví dụ mèo cam → mèo đen (đổi ảnh trong assets/imgs_demo nếu cần)
    img_path = "./assets/imgs_demo/woman_face.jpg"
    src_p = "woman"
    edit_p = "Taylor Swift"

    # img_path = "./assets/imgs_demo/02.jpg"
    # src_p = "dog"
    # edit_p = "dog with mouth opened"

    start_time = time.time()
    result = edit_image(img_path, src_p, edit_p, inverse_model, aux_model, ip_sb_model)
    print(f"Edit {src_p}->{edit_p} in {time.time()-start_time}")
    save_image(result, f"result_{src_p}->{edit_p}.png")
