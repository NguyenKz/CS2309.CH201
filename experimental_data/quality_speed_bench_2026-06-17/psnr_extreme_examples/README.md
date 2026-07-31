# PSNR extreme examples (FP16 ↔ FP32)

Cặp output cùng `(job_id, prompt_idx)` từ quality speed bench — dùng minh họa Feedback 2 (PSNR ~48.5 là so FP16 vs FP32, không vs ảnh nguồn).

Metric từ `../quality_raw.csv`, config `improved_fp16_cache`.

| Thư mục | job | PSNR | SSIM | LPIPS | MSE |
|---|---|---|---|---|---|
| `low_psnr/` | `111000000004_0` | 34.97 | 0.9889 | 0.0036 | 0.000319 |
| `mid_sample/` | `000000000000_0` | 46.48 | 0.9979 | 0.0007 | 0.000022 |
| `high_psnr/` | `111000000006_0` | 56.73 | 0.9990 | 0.0001 | 0.000002 |

Mỗi thư mục: `fp32_*.png` (baseline) và `fp16_*.png` (improved).

So online: [Img2Go Compare Images](https://www.img2go.com/compare-image). Đo trong project: `torchmetrics`. Chi tiết: `feedback.md` mục **F2**.
