# Báo cáo thời gian inference SwiftEdit

- **Ngày đo:** 2026-06-14 06:55 UTC
- **Số mẫu:** 20
- **Nguồn log:** `results/timing_bench20.log`
- **Pipeline:** một lần gọi `edit_image()` (512×512, 1-step SwiftEdit)

## Môi trường đo

| Mục | Giá trị |
|-----|---------|
| Thiết bị tính toán | `mps` (Metal Performance Shaders) |
| Máy | Apple M4 |
| RAM | 24 GB |
| Hệ điều hành | macOS 26.5 |
| Python | 3.12.10 |
| PyTorch | 2.12.0 |
| transformers | 4.57.6 |
| diffusers | 0.35.2 |

### Model sử dụng (SwiftEdit)

| Thành phần | Checkpoint / nguồn | Ghi chú |
|-----------|--------------------|---------|
| Inverse UNet | `swiftedit_weights/inverse_ckpt-120k (subfolder unet_ema)` | base: stabilityai/sd-turbo, fp32 |
| Generation UNet (SwiftBrush v2) | `swiftedit_weights/sbv2_0.5` | one-step generator |
| IP-Adapter | `swiftedit_weights/ip_adapter_ckpt-90k/ip_adapter.bin` | image prompt adapter |
| VAE / Text encoder / Tokenizer | `Manojb/stable-diffusion-2-1-base` | SD 2.1 base (mirror) |
| CLIP image encoder | `h94/IP-Adapter (models/image_encoder)` | CLIP ViT-H |

## Tổng quan

| Chỉ số | Giá trị (ms) | Giá trị (s) |
|--------|-------------:|------------:|
| Trung bình | 69017.99 | 69.02 |
| Độ lệch chuẩn | 15809.17 | — |
| Min / Max | 15785.97 / 82914.44 | 15.79 / 82.91 |
| P50 / P95 | 74913.58 / 82914.44 | 74.91 / 82.91 |

**Throughput ước lượng:** ~0.014 ảnh/giây (chỉ inference, không tính load model).

> **Warmup (MPS):** 2 mẫu đầu thường nhanh/chậm bất thường do biên dịch kernel lần đầu. Steady-state (18 mẫu còn lại): trung bình **73.64 s**, P50 **75.36 s**.

## Thời gian từng công đoạn (ms)

| Công đoạn | Mô tả | Mean | Std | % tổng | Min | Max | P50 |
|-----------|-------|-----:|----:|-------:|----:|----:|----:|
| `vae_encode` | VAE encode (ảnh → latent) | 992.73 | 428.99 | 1.4% | 548.76 | 2254.22 | 874.86 |
| `inv_text_encode` | Text encoder (inverse, src+edit) | 5162.37 | 2274.16 | 7.5% | 305.07 | 8032.44 | 5689.91 |
| `unet_inverse` | UNet inverse (ước lượng noise) | 14849.11 | 3541.99 | 21.5% | 5754.43 | 22973.05 | 15099.53 |
| `mask_estimate` | Ước lượng mask chỉnh sửa | 12.15 | 9.99 | 0.0% | 3.31 | 38.08 | 8.02 |
| `gen_image_embeds` | IP-Adapter image embeds | 16221.78 | 4885.38 | 23.5% | 1670.11 | 22586.85 | 16971.88 |
| `gen_text_encode` | Text encoder (generation) | 754.74 | 762.36 | 1.1% | 81.88 | 3206.81 | 429.69 |
| `gen_unet` | UNet 1-step (sinh ảnh) | 14958.36 | 3680.76 | 21.7% | 1311.09 | 19875.71 | 15492.78 |
| `gen_vae_decode` | VAE decode (latent → ảnh) | 15989.97 | 4317.28 | 23.2% | 4077.4 | 21463.72 | 16479.78 |

## Phân tích nhanh

- **Hai lần UNet forward** (`unet_inverse` + `gen_unet`): ~29807 ms/trung bình (~43.2% tổng thời gian).
- **VAE** (`vae_encode` + `gen_vae_decode`): encode + decode latent.
- **Text/IP embeds**: hai lần text encoder + IP-Adapter image encoder.
- **Mask estimate**: nhẹ so với UNet.

## Chất lượng chỉnh sửa (PIE-Bench metrics)

Tính trên 20 mẫu cùng đợt (nguồn: `results/piebench/metrics.csv`). PSNR/MSE chỉ tính trên các mẫu có vùng nền (mask không phủ toàn ảnh).

| Metric | Ý nghĩa | Mean | Min | Max | Số mẫu |
|--------|---------|-----:|----:|----:|------:|
| `clip_whole` | CLIP-Whole (toàn ảnh ↔ edit prompt) (cao ↑) | 23.02 | 16.47 | 32.6 | 20 |
| `clip_edited` | CLIP-Edited (vùng sửa ↔ edit prompt) (cao ↑) | 21.46 | 11.73 | 32.6 | 20 |
| `psnr_unedit` | PSNR vùng nền (giữ nền) (cao ↑) | 14.01 | 8.58 | 20.17 | 9 |
| `mse_unedit` | MSE vùng nền (giữ nền) (thấp ↓) | 0.06 | 0.01 | 0.14 | 9 |

## Chi tiết từng mẫu

| # | Label | Total (ms) | unet_inverse | gen_unet |
|--:|-------|----------:|-------------:|---------:|
| 1 | a slanted mountain bicycle on the road in front of a building->a slanted rusty mountain motorcycle in front of a fence | 15785.97 | 7138.85 | 1311.09 |
| 2 | a round cake with orange frosting on a wooden plate->a square cake with strawberry frosting on a plastic plate | 39057.59 | 5754.43 | 12951.12 |
| 3 | fishes and kelp in the ocean->sharks and flowers in the ocean | 67254.66 | 14004.1 | 14889.65 |
| 4 | a cat wearing a pink hat->a tiger wearing a pink scarf | 75816.24 | 22973.05 | 14320.61 |
| 5 | a dog wearing space suit->a dog wearing space suit and scarf with flowers in mouth | 75079.41 | 15941.74 | 16485.41 |
| 6 | a cat->a cat with a gold chain and a star on its head | 82914.44 | 16457.42 | 19875.71 |
| 7 | a wathet bird sitting on a branch of yellow flowers->a wathet bird sitting on a branch | 79408.78 | 17126.83 | 17442.4 |
| 8 | a cat wearing headphones on a gray background->a cat on a gray background | 81596.75 | 15342.39 | 16854.85 |
| 9 | a cartoon painting of a fox with big eyes in a car->a cartoon painting of a fox with small eyes in a big car | 74747.76 | 13961.68 | 14147.48 |
| 10 | a colorful parrot with its wings spread out->a colorful parrot with its wings fold in | 76370.67 | 16174.07 | 15358.43 |
| 11 | a cartoon dog laying down on the ground->a cartoon dog with tail down jumping on the ground | 66977.55 | 14485.8 | 16817.43 |
| 12 | a standing pixel robot dog->a sitting and tilting its head pixel robot dog wagging its tail | 75633.24 | 18244.55 | 15188.77 |
| 13 | a black raven with red eyes sits on a tree stump in the rain->a white raven with green eyes sits on a tree stump in the rain | 75765.43 | 15052.3 | 15485.34 |
| 14 | otter in the water with pink hearts->white otter in the water with green hearts | 74410.27 | 14076.31 | 15500.22 |
| 15 | a drawing of a brown bear sitting down->a drawing of a knitted brown bear toy sitting down | 79145.58 | 16904.51 | 16363.19 |
| 16 | a colorful bird sitting on a branch with a green background->a colorful wooden bird toy sitting on a branch with a green background | 68492.9 | 15283.26 | 15531.18 |
| 17 | a little carton sheep in a white background->a little carton sheep in a forest background | 68344.74 | 14282.89 | 14140.93 |
| 18 | watercolor painting of a deer with flowers on its head standing in white background->watercolor painting of a deer with flowers on its head standing in dark background | 60362.71 | 15146.77 | 11882.98 |
| 19 | a cat and a bunny->a realism style cat and a expressionism style bunny | 65635.96 | 14160.34 | 16296.03 |
| 20 | a bird standing on tree branch->a realism style painting of bird standing on tree branch | 77559.07 | 14470.99 | 18324.29 |
