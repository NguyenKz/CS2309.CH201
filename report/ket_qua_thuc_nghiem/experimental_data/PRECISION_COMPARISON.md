# Bảng so sánh precision (tổng hợp)

Sinh bởi `scripts/compare_precision_runs.py`. Case 1–4 từ bench T4 2026-06-17;
các dòng `precision_disk_vram_*` từ Phase A/B (disk fp16 / fp4_from_fp16).

Chi tiết kiểm kê: [`PRECISION_CASES.md`](./PRECISION_CASES.md).

| Case | Config | Nguồn | n | s/edit | VRAM/peak MB | PSNR | Disk | Status |
|---|---|---|---:|---:|---:|---:|---|---|
| 1 Full FP32 | `baseline_fp32` | quality_speed_bench_2026-06-17 | 600 | 2.91 | 14596 | None | fp32 ~10GB | DONE |
| 2 FP16 compute (+cache) | `improved_fp16_cache` | quality_speed_bench_2026-06-17 | 600 | 1.71 | 8446 | 48.56 | disk vẫn fp32 | DONE |
| 3 FP8 runtime | `improved_fp8_cache` | quality_speed_bench_2026-06-17 | 600 | 1.52 | 7819 | 6.01 | disk vẫn fp32 | DONE (fail chất lượng) |
| 4 FP4 runtime | `improved_fp4_cache` | quality_speed_bench_2026-06-17 | 600 | 1.74 | 7515 | 21.67 | disk vẫn fp32 | DONE |
| disk-run precision_disk_vram_2026-07-19 / baseline_fp32 | `baseline_fp32` | precision_disk_vram_2026-07-19 | 4 | 31.396 | 12366.27 | 99.0 | fp16 tree 4.9421 GiB | DONE |
| disk-run precision_disk_vram_2026-07-19 / fp16_disk | `fp16_disk` | precision_disk_vram_2026-07-19 | 4 | 5.342 | 6567.03 | 51.425 | fp16 tree 4.9421 GiB | DONE |
| disk-run precision_disk_vram_2026-07-19_promptfix / baseline_fp32 | `baseline_fp32` | precision_disk_vram_2026-07-19_promptfix | 2 | 20.886 | 12366.27 | 99.0 | fp16 tree 4.9421 GiB | DONE |
| disk-run precision_disk_vram_2026-07-19_promptfix / fp16_disk | `fp16_disk` | precision_disk_vram_2026-07-19_promptfix | 2 | 5.464 | 6567.03 | 48.768 | fp16 tree 4.9421 GiB | DONE |

## Thiếu gì?

- Case **FP16 disk trên T4** với quy mô gần bench cũ (hoặc subset rõ ràng).
- Case **FP4 from fp16 disk** trên T4.
- Sau khi có `bundle.zip` Colab: thả vào `experimental_data/` rồi chạy lại script này.

