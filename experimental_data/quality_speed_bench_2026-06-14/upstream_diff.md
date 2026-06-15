# Diff SwiftEdit gốc (Qualcomm-AI-research) vs bản đã sửa

Upstream: https://github.com/Qualcomm-AI-research/SwiftEdit.git (clone 230bff1)
Ngày so: 2026-06-15 10:57

## infer.py
```diff
5a6
> import numpy as np
6a8
> import torch.nn.functional as F
11a14
> from timing import StageTimer
12a16,30
> 
> def prepare_user_mask(user_mask, out_hw, device, threshold=0.5):
>     """Chuẩn bị mask người dùng vẽ thành tensor nhị phân [H,W] ở độ phân giải latent.
> 
>     user_mask: ndarray/tensor 2D (H,W) hoặc 3D (H,W,C) giá trị {0..255} hoặc [0,1].
>     Vùng được vẽ (giá trị > threshold) = 1 = vùng sẽ chỉnh sửa/xóa.
>     """
>     m = torch.as_tensor(np.asarray(user_mask), device=device, dtype=torch.float32)
>     if m.dim() == 3:
>         m = m[..., -1] if m.shape[-1] in (1, 2, 4) else m.mean(dim=-1)  # alpha nếu RGBA
>     if m.max() > 1.0:
>         m = m / 255.0
>     m = F.interpolate(m[None, None], size=out_hw, mode="bilinear", align_corners=False)[0, 0]
>     return (m > threshold).to(torch.float32)
> 
17a36,44
> 
> def get_device():
>     if torch.cuda.is_available():
>         return "cuda"
>     if torch.backends.mps.is_available():
>         return "mps"
>     return "cpu"
> 
> 
24a52,79
> class EditCache:
>     """Cache các tensor chỉ phụ thuộc ảnh nguồn / source prompt.
> 
>     Dùng cho kịch bản realtime: cùng 1 ảnh + source prompt, đổi nhiều edit prompt.
>     Tự invalidate khi đổi ảnh (latent + CLIP image embed) hoặc đổi source prompt
>     (source text embed của inverse model + generation model).
>     """
> 
>     def __init__(self, enabled: bool = True):
>         self.enabled = enabled
>         self._img_path = None
>         self._src_p = None
>         self.latents = None              # VAE encode ảnh nguồn (image-only)
>         self.src_inv_embed = None        # source prompt embed — inverse model
>         self.gen_embed_cache: dict = {}  # cho gen_img: image_prompt_embeds + src_text_embed
> 
>     def ensure(self, img_path, src_p) -> None:
>         """Invalidate phần cache không còn hợp lệ khi đổi ảnh / source prompt."""
>         if img_path != self._img_path:
>             self._img_path = img_path
>             self.latents = None
>             self.gen_embed_cache.pop("image_prompt_embeds", None)
>         if src_p != self._src_p:
>             self._src_p = src_p
>             self.src_inv_embed = None
>             self.gen_embed_cache.pop("src_text_embed", None)
> 
> 
37a93,94
>     cache=None,
>     user_mask=None,
43a101,103
>             + cache: EditCache tùy chọn — tái dùng latent/embedding khi cùng ảnh+source prompt.
>             + user_mask: mask người dùng vẽ (2D/3D). Nếu có, ghi đè self-guided mask —
>               dùng cho xóa/chỉnh vật thể theo vùng khoanh tay.
45,46c105,108
<     mid_timestep = torch.ones((1,), dtype=torch.int64, device="cuda") * 500
<     final_timestep = torch.ones((1,), dtype=torch.int64, device="cuda") * 999
---
>     device = inverse_model.device
>     timer = StageTimer(device, label=f"{src_p}->{edit_p}")
>     mid_timestep = torch.ones((1,), dtype=torch.int64, device=device) * 500
>     final_timestep = torch.ones((1,), dtype=torch.int64, device=device) * 999
47a110,112
>     if cache is not None:
>         cache.ensure(img_path, src_p)
> 
51c116
<     processed_image = to_tensor(pil_img_cond).unsqueeze(0).to("cuda") * 2 - 1
---
>     processed_image = to_tensor(pil_img_cond).unsqueeze(0).to(device) * 2 - 1
53,58c118,129
<     # Predict inverted noise
<     latents = inverse_model.vae.encode(
<         processed_image.to(inverse_model.weight_dtype)
<     ).latent_dist.sample()
<     latents = latents * inverse_model.vae.config.scaling_factor
<     dub_latents = torch.cat([latents] * 2, dim=0)
---
>     # Predict inverted noise — latent VAE chỉ phụ thuộc ảnh nguồn -> cache được
>     with timer.stage("vae_encode"):
>         if cache is not None and cache.latents is not None:
>             latents = cache.latents
>         else:
>             latents = inverse_model.vae.encode(
>                 processed_image.to(inverse_model.vae.dtype)
>             ).latent_dist.sample()
>             latents = latents * inverse_model.vae.config.scaling_factor
>             if cache is not None:
>                 cache.latents = latents
>         dub_latents = torch.cat([latents] * 2, dim=0)
60,63c131,146
<     input_id = tokenize_captions(inverse_model.tokenizer, [src_p, edit_p]).to("cuda")
<     encoder_hidden_state = inverse_model.text_encoder(input_id)[0].to(
<         dtype=inverse_model.weight_dtype
<     )
---
>     with timer.stage("inv_text_encode"):
>         if cache is not None:
>             # Source prompt embed cache được; chỉ encode lại edit prompt.
>             if cache.src_inv_embed is None:
>                 src_id = tokenize_captions(inverse_model.tokenizer, [src_p]).to(device)
>                 cache.src_inv_embed = inverse_model.text_encoder(src_id)[0]
>             edit_id = tokenize_captions(inverse_model.tokenizer, [edit_p]).to(device)
>             edit_embed = inverse_model.text_encoder(edit_id)[0]
>             encoder_hidden_state = torch.cat([cache.src_inv_embed, edit_embed], dim=0).to(
>                 dtype=inverse_model.weight_dtype
>             )
>         else:
>             input_id = tokenize_captions(inverse_model.tokenizer, [src_p, edit_p]).to(device)
>             encoder_hidden_state = inverse_model.text_encoder(input_id)[0].to(
>                 dtype=inverse_model.weight_dtype
>             )
65,67c148,152
<     predict_inverted_code = inverse_model.unet_inverse(
<         dub_latents, mid_timestep, encoder_hidden_state
<     ).sample.to("cuda", dtype=inverse_model.weight_dtype)
---
>     with timer.stage("unet_inverse"):
>         unet_inv_dtype = inverse_model.unet_inverse.dtype
>         predict_inverted_code = inverse_model.unet_inverse(
>             dub_latents.to(unet_inv_dtype), mid_timestep, encoder_hidden_state.to(unet_inv_dtype)
>         ).sample.float()  # về fp32 cho mask + input_sb (alpha_t/sigma_t fp32) ổn định
70,74c155,166
<     inverted_noise_1, inverted_noise_2 = predict_inverted_code.chunk(2)
<     subed = (inverted_noise_1 - inverted_noise_2).abs_().mean(dim=[0, 1])
<     max_v = (subed.mean() * clamp_rate).item()
<     mask12 = subed.clamp(0, max_v) / max_v
<     mask12 = mask12.detach().cpu().apply_(lambda pix: to_binary(pix, mask_threshold)).to("cuda")
---
>     with timer.stage("mask_estimate"):
>         inverted_noise_1, inverted_noise_2 = predict_inverted_code.chunk(2)
>         subed = (inverted_noise_1 - inverted_noise_2).abs_().mean(dim=[0, 1])
>         max_v = (subed.mean() * clamp_rate).item()
>         mask12 = subed.clamp(0, max_v) / max_v
>         # Nhị phân hóa vectorized ngay trên device (thay .cpu().apply_() — vòng lặp Python từng pixel)
>         mask12 = (mask12 > mask_threshold).to(dtype=subed.dtype)
>         # Ghi đè bằng mask người dùng khoanh (xóa/chỉnh vật thể theo vùng) nếu có.
>         if user_mask is not None:
>             mask12 = prepare_user_mask(
>                 user_mask, mask12.shape, device, mask_threshold
>             ).to(dtype=subed.dtype)
83c175,180
<         pil_image=pil_img_cond, prompts=[src_p, edit_p], noise=input_sb
---
>         pil_image=pil_img_cond,
>         prompts=[src_p, edit_p],
>         noise=input_sb,
>         return_noise_image=False,
>         timer=timer,
>         embed_cache=(cache.gen_embed_cache if cache is not None else None),
85a183,184
>     timer.dump(extra={"img_path": img_path})
> 
90a190,192
>     device = get_device()
>     print(f"Using device: {device}")
> 
93,94c195,196
<     inverse_model = InverseModel(inverse_ckpt)
<     aux_model = AuxiliaryModel()
---
>     inverse_model = InverseModel(inverse_ckpt, device=device)
>     aux_model = AuxiliaryModel(device=device)
98c200,202
<     ip_sb_model = IPSBV2Model(path_unet_sb, ip_ckpt, aux_model, with_ip_mask_controller=True)
---
>     ip_sb_model = IPSBV2Model(
>         path_unet_sb, ip_ckpt, aux_model, device=device, with_ip_mask_controller=True
>     )
100,101c204
<     # Input
< 
---
>     # Input — ví dụ mèo cam → mèo đen (đổi ảnh trong assets/imgs_demo nếu cần)
```

## models.py
```diff
3a4,5
> from contextlib import contextmanager
> 
18a21,45
> class _NullTimer:
>     """Timer giả: dùng khi gen_img được gọi mà không truyền StageTimer."""
> 
>     @contextmanager
>     def stage(self, name):
>         yield
> 
> 
> def resolve_dtype(dtype):
>     """Chuẩn hóa 'fp16'/'bf16'/'fp32' (hoặc torch.dtype) về torch.dtype."""
>     if isinstance(dtype, torch.dtype):
>         return dtype
>     return {
>         "fp16": torch.float16,
>         "float16": torch.float16,
>         "bf16": torch.bfloat16,
>         "bfloat16": torch.bfloat16,
>     }.get(dtype, torch.float32)
> 
> 
> def module_dtype(module):
>     """dtype của tham số đầu tiên trong module (cho module không có .dtype)."""
>     return next(module.parameters()).dtype
> 
> 
61c88,89
<         device="cuda"
---
>         device="cuda",
>         channels_last=False,
63,68c91
<         if dtype == "fp16":
<             self.weight_dtype = torch.float16
<         elif dtype == "bf16":
<             self.weight_dtype = torch.bfloat16
<         else:
<             self.weight_dtype = torch.float32
---
>         self.weight_dtype = resolve_dtype(dtype)
72a96
>         # VAE giữ fp32: SD VAE fp16 dễ ra NaN/ảnh đen.
79a104,105
>         if channels_last:
>             self.unet_inverse = self.unet_inverse.to(memory_format=torch.channels_last)
102c128,129
<         model_name="stabilityai/stable-diffusion-2-1-base",
---
>         # stabilityai/* SD2.1 repos were deprecated/private on HF (2025+); mirror has same weights
>         model_name="Manojb/stable-diffusion-2-1-base",
104a132
>         dtype="fp32",
106a135
>         self.weight_dtype = resolve_dtype(dtype)
107a137
>         # VAE giữ fp32 (decode ổn định, tránh NaN/ảnh đen ở fp16).
112c142
<             self.device, dtype=torch.float32
---
>             self.device, dtype=self.weight_dtype
117c147
<         ).to(device, dtype=torch.float32)
---
>         ).to(device, dtype=self.weight_dtype)
133a164,165
>         dtype="fp32",
>         channels_last=False,
136a169,170
>         self.weight_dtype = resolve_dtype(dtype)
>         self.channels_last = channels_last
199c233,235
<         self.load_state_dict(torch.load(ip_model_path))
---
>         self.load_state_dict(
>             torch.load(ip_model_path, map_location="cpu", weights_only=True)
>         )
201a238,245
>         # Áp dtype/memory-format SAU khi load weight (fp32) để chỉ ép kiểu 1 lần.
>         # alpha_t/sigma_t là tensor thuộc tính (không phải param) -> giữ fp32 cho math ổn định.
>         if self.weight_dtype != torch.float32:
>             self.unet = self.unet.to(dtype=self.weight_dtype)
>             self.image_proj_model = self.image_proj_model.to(dtype=self.weight_dtype)
>         if self.channels_last:
>             self.unet = self.unet.to(memory_format=torch.channels_last)
> 
219a264,265
>         enc_dtype = self.aux_model.image_encoder.dtype
>         proj_dtype = module_dtype(self.image_proj_model)
227c273
<                 clip_image.to(self.device, dtype=torch.float32)
---
>                 clip_image.to(self.device, dtype=enc_dtype)
230,231c276,277
<             clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float32)
<         image_prompt_embeds = self.image_proj_model(clip_image_embeds)
---
>             clip_image_embeds = clip_image_embeds.to(self.device, dtype=enc_dtype)
>         image_prompt_embeds = self.image_proj_model(clip_image_embeds.to(proj_dtype))
258a305,307
>         return_noise_image=False,
>         timer=None,
>         embed_cache=None,
259a309,312
>         timer = timer or _NullTimer()
>         # embed_cache: dict tùy chọn để tái dùng embed phụ thuộc ảnh/source prompt
>         # giữa nhiều edit trên cùng ảnh. None -> không cache (dict tạm, bị bỏ).
>         cache = embed_cache if embed_cache is not None else {}
262,264d314
<         num_samples = len(prompts)
<         
<         # Prepare prompt + image embeds
266a317
>         num_samples = len(prompts)
268,271c319,328
<         image_prompt_embeds = self.get_image_embeds(pil_image=pil_image)
<         bs_embed, seq_len, _ = image_prompt_embeds.shape
<         image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
<         image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
---
>         # CLIP image embed — chỉ phụ thuộc ảnh nguồn -> cache được
>         with timer.stage("gen_image_embeds"):
>             if "image_prompt_embeds" in cache:
>                 image_prompt_embeds = cache["image_prompt_embeds"]
>             else:
>                 image_prompt_embeds = self.get_image_embeds(pil_image=pil_image)
>                 cache["image_prompt_embeds"] = image_prompt_embeds
>             bs_embed, seq_len, _ = image_prompt_embeds.shape
>             image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
>             image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
273,275c330,341
<         input_id = tokenize_captions(self.aux_model.tokenizer, prompts).to(self.device)
<         prompt_embeds_ = self.aux_model.text_encoder(input_id)[0]
<         prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
---
>         with timer.stage("gen_text_encode"):
>             # Source prompt embed (hàng 0) cache được; chỉ encode lại edit prompt.
>             if "src_text_embed" in cache and num_samples == 2:
>                 edit_id = tokenize_captions(self.aux_model.tokenizer, [prompts[1]]).to(self.device)
>                 edit_embed = self.aux_model.text_encoder(edit_id)[0]
>                 prompt_embeds_ = torch.cat([cache["src_text_embed"], edit_embed], dim=0)
>             else:
>                 input_id = tokenize_captions(self.aux_model.tokenizer, prompts).to(self.device)
>                 prompt_embeds_ = self.aux_model.text_encoder(input_id)[0]
>                 if num_samples == 2:
>                     cache["src_text_embed"] = prompt_embeds_[0:1]
>             prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
278,279c344,349
<         noise = torch.cat([noise] * num_samples, dim=0)
<         model_pred = self.unet(noise, self.timestep, prompt_embeds).sample
---
>         with timer.stage("gen_unet"):
>             noise = torch.cat([noise] * num_samples, dim=0).float()
>             unet_dtype = self.unet.dtype
>             model_pred = self.unet(
>                 noise.to(unet_dtype), self.timestep, prompt_embeds.to(unet_dtype)
>             ).sample
281,282c351,352
<         if model_pred.shape[1] == noise.shape[1] * 2:
<             model_pred, _ = torch.split(model_pred, noise.shape[1], dim=1)
---
>             if model_pred.shape[1] == noise.shape[1] * 2:
>                 model_pred, _ = torch.split(model_pred, noise.shape[1], dim=1)
284c354,356
<         pred_original_sample = (noise - self.sigma_t * model_pred) / self.alpha_t
---
>             # Hậu xử lý ở fp32 (alpha_t/sigma_t fp32) để ổn định số học, tránh NaN fp16.
>             model_pred = model_pred.float()
>             pred_original_sample = (noise - self.sigma_t * model_pred) / self.alpha_t
286,292c358,364
<         if self.aux_model.noise_scheduler.config.thresholding:
<             pred_original_sample = self.aux_model.noise_scheduler._threshold_sample(
<                 pred_original_sample
<             )
<         elif self.aux_model.noise_scheduler.config.clip_sample:
<             clip_sample_range = self.aux_model.noise_scheduler.config.clip_sample_range
<             pred_original_sample = pred_original_sample.clamp(-clip_sample_range, clip_sample_range)
---
>             if self.aux_model.noise_scheduler.config.thresholding:
>                 pred_original_sample = self.aux_model.noise_scheduler._threshold_sample(
>                     pred_original_sample
>                 )
>             elif self.aux_model.noise_scheduler.config.clip_sample:
>                 clip_sample_range = self.aux_model.noise_scheduler.config.clip_sample_range
>                 pred_original_sample = pred_original_sample.clamp(-clip_sample_range, clip_sample_range)
294,297c366,370
<         pred_original_sample = pred_original_sample / self.aux_model.vae.config.scaling_factor
<         image = (
<             self.aux_model.vae.decode(pred_original_sample.to(dtype=torch.float32)).sample.float() + 1
<         ) / 2
---
>         with timer.stage("gen_vae_decode"):
>             pred_original_sample = pred_original_sample / self.aux_model.vae.config.scaling_factor
>             image = (
>                 self.aux_model.vae.decode(pred_original_sample.to(dtype=torch.float32)).sample.float() + 1
>             ) / 2
299,304c372,379
<         noise_image = noise / self.aux_model.vae.config.scaling_factor
<         noise_image = (
<             self.aux_model.vae.decode(noise_image.to(dtype=self.aux_model.vae.dtype)).sample.float()
<             + 1
<         ) / 2
< 
---
>         noise_image = None
>         if return_noise_image:
>             with timer.stage("gen_vae_decode_noise"):
>                 noise_image = noise / self.aux_model.vae.config.scaling_factor
>                 noise_image = (
>                     self.aux_model.vae.decode(noise_image.to(dtype=self.aux_model.vae.dtype)).sample.float()
>                     + 1
>                 ) / 2
```

## src/__init__.py
```diff
```

## src/attention_processor.py
```diff
```

## src/mask_attention_processor.py
```diff
```

## src/mask_ip_controller.py
```diff
39a40
>             mask = mask.to(sim.dtype)  # khớp dtype để chạy fp16
66c67
<             mask = mask.reshape(-1, 1)  # (hw, 1)
---
>             mask = mask.reshape(-1, 1).to(out_source.dtype)  # (hw, 1), khớp dtype fp16
87c88
<             mask = mask.reshape(-1, 1)  # (hw, 1)
---
>             mask = mask.reshape(-1, 1).to(out_source.dtype)  # (hw, 1), khớp dtype fp16
139c140
<             mask = mask.reshape(-1, 1)  # (hw, 1)
---
>             mask = mask.reshape(-1, 1).to(out_source.dtype)  # (hw, 1), khớp dtype fp16
172c173
<             mask = mask.reshape(-1, 1)  # (hw, 1)
---
>             mask = mask.reshape(-1, 1).to(out_source.dtype)  # (hw, 1), khớp dtype fp16
```

