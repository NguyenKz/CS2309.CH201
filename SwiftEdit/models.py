# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from contextlib import contextmanager

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPVisionModelWithProjection,
)

from src.mask_ip_controller import *
from src.attention_processor import AttnProcessor2_0 as AttnProcessor
from src.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor
from src.mask_attention_processor import IPAttnProcessor2_0WithIPMaskController

class _NullTimer:
    """Timer giả: dùng khi gen_img được gọi mà không truyền StageTimer."""

    @contextmanager
    def stage(self, name):
        yield


def resolve_dtype(dtype):
    """Chuẩn hóa 'fp16'/'bf16'/'fp32' (hoặc torch.dtype) về torch.dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    return {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, torch.float32)


def module_dtype(module):
    """dtype của tham số đầu tiên trong module (cho module không có .dtype)."""
    return next(module.parameters()).dtype


def quantize_unet(unet, quant, device, compute_dtype):
    """Lượng tử hóa (weight-only) các lớp Linear của UNet để giảm VRAM.

    Áp dụng SAU khi đã nạp trọng số (khớp luồng IP-adapter: trọng số UNet được nạp
    sau `from_pretrained` nên không dùng được quantization_config lúc load).

    quant:
        None / "" / "none" -> không lượng tử hóa (trả nguyên).
        "fp8"  -> torchao Float8WeightOnlyConfig (e4m3). Cần GPU hỗ trợ; trên Turing/T4
                  có thể không có kernel fp8 -> raise (caller bắt và bỏ qua config).
        "fp4"  -> bitsandbytes Linear4bit quant_type="fp4" (chạy được trên Turing/T4).

    Chỉ áp cho `nn.Linear` (Conv giữ nguyên — weight-only quant không đụng Conv).
    VAE luôn fp32 và được xử lý bên ngoài, không truyền vào đây.
    """
    if not quant or quant == "none":
        return unet

    if quant == "fp8":
        from torchao.quantization import Float8WeightOnlyConfig, quantize_

        quantize_(unet, Float8WeightOnlyConfig())
        return unet

    if quant == "fp4":
        import bitsandbytes as bnb

        def _swap(mod):
            for name, child in list(mod.named_children()):
                if isinstance(child, torch.nn.Linear):
                    new = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=compute_dtype,
                        quant_type="fp4",
                    )
                    # Params4bit nhận trọng số fp16/fp32 trên CPU; .to(cuda) sẽ nén thật.
                    new.weight = bnb.nn.Params4bit(
                        child.weight.data.detach().to("cpu", dtype=compute_dtype),
                        requires_grad=False,
                        quant_type="fp4",
                    )
                    if child.bias is not None:
                        new.bias = torch.nn.Parameter(
                            child.bias.data.detach().to("cpu", dtype=compute_dtype)
                        )
                    setattr(mod, name, new)
                else:
                    _swap(child)

        _swap(unet)
        unet.to(device)  # kích hoạt nén 4-bit của bitsandbytes
        return unet

    raise ValueError(f"quant không hỗ trợ: {quant!r} (chỉ nhận None/'fp8'/'fp4')")


def tokenize_captions(tokenizer, captions):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(
        self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4
    ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(
            clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class InverseModel:
    """
        Inversion Network that bring source image latents to noisy latents.
    """
    def __init__(
        self, 
        pretrained_model_name_path, 
        model_name="stabilityai/sd-turbo",
        dtype="fp32",
        device="cuda",
        channels_last=False,
        quant=None,
    ):
        self.weight_dtype = resolve_dtype(dtype)

        self.device = device
        self.model_name = model_name
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.model_name, subfolder="scheduler")
        # VAE giữ fp32: SD VAE fp16 dễ ra NaN/ảnh đen.
        self.vae = AutoencoderKL.from_pretrained(self.model_name, subfolder="vae").to(
            self.device, dtype=torch.float32
        )

        self.unet_inverse = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_path, subfolder="unet_ema"
        ).to(self.device, dtype=self.weight_dtype)
        if channels_last:
            self.unet_inverse = self.unet_inverse.to(memory_format=torch.channels_last)

        # dtype tính toán của unet (sau nén 4-bit, .dtype không còn đúng -> lưu rõ ràng).
        self.compute_dtype = self.weight_dtype
        # Lượng tử hóa weight-only (fp8/fp4) SAU khi đã nạp + ép dtype/format.
        self.unet_inverse = quantize_unet(
            self.unet_inverse, quant, self.device, self.compute_dtype
        )

        self.unet_inverse.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_name, subfolder="text_encoder"
        ).to(self.device, dtype=self.weight_dtype)

        T = torch.ones((1,), dtype=torch.int64, device=self.device)
        T = T * (self.noise_scheduler.config.num_train_timesteps - 1)
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        self.corrupt_alpha_t = (alphas_cumprod[int(T / 4)] ** 0.5).view(-1, 1, 1, 1)
        self.corrupt_sigma_t = ((1 - alphas_cumprod[int(T / 4)]) ** 0.5).view(-1, 1, 1, 1)

        del alphas_cumprod

class AuxiliaryModel:
    """
        A few auxiliary and supported models (text encoder, noise scheduler, tokenizer, ...) as separate modules.
    """
    def __init__(
        self,
        # stabilityai/* SD2.1 repos were deprecated/private on HF (2025+); mirror has same weights
        model_name="Manojb/stable-diffusion-2-1-base",
        image_encoder_path="h94/IP-Adapter",
        device="cuda",
        dtype="fp32",
    ):
        self.device = device
        self.weight_dtype = resolve_dtype(dtype)
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        # VAE giữ fp32 (decode ổn định, tránh NaN/ảnh đen ở fp16).
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(
            self.device, dtype=self.weight_dtype
        )

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path, subfolder="models/image_encoder"
        ).to(device, dtype=self.weight_dtype)
        self.image_encoder.requires_grad_(False)

        self.clip_image_processor = CLIPImageProcessor()


class IPSBV2Model(torch.nn.Module):
    """
        SwiftBrushv2 model with incorporated IP-Adapter.
    """
    def __init__(
        self,
        pretrained_model_name_path,
        ip_model_path,
        aux_model,
        device="cuda",
        with_ip_mask_controller=False,
        dtype="fp32",
        channels_last=False,
        quant=None,
    ):
        super().__init__()
        self.device = device
        self.weight_dtype = resolve_dtype(dtype)
        self.channels_last = channels_last
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_path
        ).to(self.device)
        self.unet.eval()
        self.aux_model = aux_model

        self.timestep = torch.ones((1,), dtype=torch.int64, device=self.device)
        self.timestep = self.timestep * (
            self.aux_model.noise_scheduler.config.num_train_timesteps - 1
        )

        self.image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.aux_model.image_encoder.config.projection_dim,
            clip_extra_context_tokens=4,
        ).to(self.device)

        self.with_ip_mask_controller = with_ip_mask_controller

        # init adapter modules
        attn_procs = {}
        unet_sd = self.unet.state_dict()
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor().to(device)
            else:
                # this is for cross-attention
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                if self.with_ip_mask_controller:
                    attn_procs[name] = IPAttnProcessor2_0WithIPMaskController(
                        hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                    ).to(device)
                else:
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                    ).to(device)
                attn_procs[name].load_state_dict(weights)

        self.unet.set_attn_processor(attn_procs)
        self.adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values())

        # prepare stuff
        alphas_cumprod = self.aux_model.noise_scheduler.alphas_cumprod.to(self.device)
        self.alpha_t = (alphas_cumprod[self.timestep] ** 0.5).view(-1, 1, 1, 1)
        self.sigma_t = ((1 - alphas_cumprod[self.timestep]) ** 0.5).view(-1, 1, 1, 1)
        del alphas_cumprod

        self.load_state_dict(
            torch.load(ip_model_path, map_location="cpu", weights_only=True)
        )
        # self.load_ip_adapter(path_ckpt_ip)

        # Áp dtype/memory-format SAU khi load weight (fp32) để chỉ ép kiểu 1 lần.
        # alpha_t/sigma_t là tensor thuộc tính (không phải param) -> giữ fp32 cho math ổn định.
        if self.weight_dtype != torch.float32:
            self.unet = self.unet.to(dtype=self.weight_dtype)
            self.image_proj_model = self.image_proj_model.to(dtype=self.weight_dtype)
        if self.channels_last:
            self.unet = self.unet.to(memory_format=torch.channels_last)

        # dtype dùng cho input UNet khi chạy. Sau khi nén 4-bit, self.unet.dtype không còn
        # phản ánh đúng dtype tính toán (param thành uint8) -> lưu rõ ràng để gen_img dùng.
        self.compute_dtype = self.weight_dtype
        # Lượng tử hóa weight-only (fp8/fp4) SAU khi đã nạp IP-adapter + ép dtype/format.
        self.unet = quantize_unet(self.unet, quant, self.device, self.compute_dtype)

    def load_ip_adapter(self, path_ckpt_ip):

        sd = torch.load(path_ckpt_ip, map_location="cpu")
        image_proj_sd = {}
        ip_sd = {}
        for k in sd:
            if k.startswith("unet"):
                pass
            elif k.startswith("image_proj_model"):
                image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
            elif k.startswith("adapter_modules"):
                ip_sd[k.replace("adapter_modules.", "")] = sd[k]

        self.image_proj_model.load_state_dict(image_proj_sd)
        self.adapter_modules.load_state_dict(ip_sd)

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        enc_dtype = self.aux_model.image_encoder.dtype
        proj_dtype = module_dtype(self.image_proj_model)
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.aux_model.clip_image_processor(
                images=pil_image, return_tensors="pt"
            ).pixel_values
            clip_image_embeds = self.aux_model.image_encoder(
                clip_image.to(self.device, dtype=enc_dtype)
            ).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=enc_dtype)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds.to(proj_dtype))
        return image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor) or isinstance(
                attn_processor, IPAttnProcessor2_0WithIPMaskController
            ):
                attn_processor.scale = scale

    def set_controller(
        self, controller, where=["down_blocks", "mid_block", "up_blocks"], type_controller=None
    ):

        for name_attn_processor, attn_processor in self.unet.attn_processors.items():
            if isinstance(attn_processor, IPAttnProcessor2_0WithIPMaskController):
                # only set at particular blocks
                for from_where in where:
                    if from_where in name_attn_processor:
                        attn_processor.controller = controller

    @torch.no_grad()
    def gen_img(
        self,
        pil_image=None,
        prompts=None,
        noise=None,
        scale=1.0,
        return_noise_image=False,
        timer=None,
        embed_cache=None,
    ):
        timer = timer or _NullTimer()
        # embed_cache: dict tùy chọn để tái dùng embed phụ thuộc ảnh/source prompt
        # giữa nhiều edit trên cùng ảnh. None -> không cache (dict tạm, bị bỏ).
        cache = embed_cache if embed_cache is not None else {}

        self.set_scale(scale)
        if prompts is None:
            prompts = ["best quality, high quality"]
        num_samples = len(prompts)

        # CLIP image embed — chỉ phụ thuộc ảnh nguồn -> cache được
        with timer.stage("gen_image_embeds"):
            if "image_prompt_embeds" in cache:
                image_prompt_embeds = cache["image_prompt_embeds"]
            else:
                image_prompt_embeds = self.get_image_embeds(pil_image=pil_image)
                cache["image_prompt_embeds"] = image_prompt_embeds
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with timer.stage("gen_text_encode"):
            # Source prompt embed (hàng 0) cache được; chỉ encode lại edit prompt.
            if "src_text_embed" in cache and num_samples == 2:
                edit_id = tokenize_captions(self.aux_model.tokenizer, [prompts[1]]).to(self.device)
                edit_embed = self.aux_model.text_encoder(edit_id)[0]
                prompt_embeds_ = torch.cat([cache["src_text_embed"], edit_embed], dim=0)
            else:
                input_id = tokenize_captions(self.aux_model.tokenizer, prompts).to(self.device)
                prompt_embeds_ = self.aux_model.text_encoder(input_id)[0]
                if num_samples == 2:
                    cache["src_text_embed"] = prompt_embeds_[0:1]
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)

        # Feed inverted noise to ip-unet generation
        with timer.stage("gen_unet"):
            noise = torch.cat([noise] * num_samples, dim=0).float()
            # compute_dtype thay cho self.unet.dtype (sai sau khi nén 4-bit).
            unet_dtype = getattr(self, "compute_dtype", None) or self.unet.dtype
            model_pred = self.unet(
                noise.to(unet_dtype), self.timestep, prompt_embeds.to(unet_dtype)
            ).sample

            if model_pred.shape[1] == noise.shape[1] * 2:
                model_pred, _ = torch.split(model_pred, noise.shape[1], dim=1)

            # Hậu xử lý ở fp32 (alpha_t/sigma_t fp32) để ổn định số học, tránh NaN fp16.
            model_pred = model_pred.float()
            pred_original_sample = (noise - self.sigma_t * model_pred) / self.alpha_t

            if self.aux_model.noise_scheduler.config.thresholding:
                pred_original_sample = self.aux_model.noise_scheduler._threshold_sample(
                    pred_original_sample
                )
            elif self.aux_model.noise_scheduler.config.clip_sample:
                clip_sample_range = self.aux_model.noise_scheduler.config.clip_sample_range
                pred_original_sample = pred_original_sample.clamp(-clip_sample_range, clip_sample_range)

        with timer.stage("gen_vae_decode"):
            pred_original_sample = pred_original_sample / self.aux_model.vae.config.scaling_factor
            image = (
                self.aux_model.vae.decode(pred_original_sample.to(dtype=torch.float32)).sample.float() + 1
            ) / 2

        noise_image = None
        if return_noise_image:
            with timer.stage("gen_vae_decode_noise"):
                noise_image = noise / self.aux_model.vae.config.scaling_factor
                noise_image = (
                    self.aux_model.vae.decode(noise_image.to(dtype=self.aux_model.vae.dtype)).sample.float()
                    + 1
                ) / 2
        return image, noise_image
