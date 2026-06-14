# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os, time

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from models import *
from timing import StageTimer

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
    else:
        return 0.0


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
):
    """
        Save keysteps to file.
            + img_path: path to the source image.
            + src_p: Source Prompt that describes source image (could leave it empty).
            + edit_p: Edit Prompt that describes your desired changes.
            + cache: EditCache tùy chọn — tái dùng latent/embedding khi cùng ảnh+source prompt.
    """
    device = inverse_model.device
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
        if cache is not None and cache.latents is not None:
            latents = cache.latents
        else:
            latents = inverse_model.vae.encode(
                processed_image.to(inverse_model.vae.dtype)
            ).latent_dist.sample()
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
        unet_inv_dtype = inverse_model.unet_inverse.dtype
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

    # Edit images
    input_sb = ip_sb_model.alpha_t * latents + ip_sb_model.sigma_t * inverted_noise_1
    mask_controller = MaskController(
        mask12, scale_text_hiddenstate=scale_ta, scale_ip_fg=scale_edit, scale_ip_bg=scale_non_edit
    )
    ip_sb_model.set_controller(mask_controller, where=["mid_blocks", "up_blocks"])
    res_gen_img, _ = ip_sb_model.gen_img(
        pil_image=pil_img_cond,
        prompts=[src_p, edit_p],
        noise=input_sb,
        return_noise_image=False,
        timer=timer,
        embed_cache=(cache.gen_embed_cache if cache is not None else None),
    )

    timer.dump(extra={"img_path": img_path})

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
