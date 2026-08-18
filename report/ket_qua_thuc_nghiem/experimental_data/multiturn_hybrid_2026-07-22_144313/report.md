# Benchmark Hybrid Multi-turn

- Ảnh: `/Users/nguyenkz/Documents/code/CS2309.CH201/data/PIE-Bench-smoke/annotation_images/0_random_140/smoke_woman.jpg` (512×512)
- Device: `mps`; số lượt: 5; tổng thời gian: 17.81s
- Thí nghiệm cô lập suy giảm VAE: không semantic edit; mask giữa ảnh chiếm 25%.

## Kết quả lượt cuối

| Mode | PSNR ngoài mask | SSIM | LPIPS |
|---|---:|---:|---:|
| Naive | 30.0924 | 0.835280 | 0.05753010883927345 |
| Hybrid | inf | 0.959185 | 0.011863231658935547 |

Hybrid giữ nguyên pixel ngoài mask; Naive đưa toàn ảnh qua VAE ở mọi lượt.
