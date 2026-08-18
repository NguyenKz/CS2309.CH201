# So sánh với quality_speed_bench_2026-06-17

Bench cũ đo **runtime cast/quant** (disk vẫn fp32). Run này đo thêm **disk fp16** + load `torch_dtype`.

| Run này | Bench 2026-06-17 | Ghi chú |
|---|---|---|
| baseline_fp32 | baseline_fp32 | reference ảnh |
| fp16_disk | improved_fp16_cache | cùng dtype chạy; khác: weights trên disk + cache (bench cũ có EditCache) |
| fp4_from_fp16 | improved_fp4_cache | chỉ CUDA; skip trên Mac |

## Số liệu tham chiếu bench cũ (Colab T4, 600 edits)

| Config | Speedup | VRAM MB | PSNR vs fp32 |
|---|---:|---:|---:|
| improved_fp16_cache | 1.70× | 8446 | 48.56 |
| improved_fp4_cache | 1.68× | 7515 | 21.67 |

## Số liệu run này (xem quality_summary.csv)

- **baseline_fp32**: PSNR_vs_fp32_mean=99.0, peak_load_mb=12366.27, n=4
- **fp16_disk**: PSNR_vs_fp32_mean=51.425, peak_load_mb=6567.03, n=4

Disk save fp16 vs fp32: **49.5%**

Khi viết báo cáo: nêu rõ Phase A = Mac/MPS (memory khác CUDA); Phase B = T4 cho VRAM + fp4.

