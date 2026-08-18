# Precision disk / memory / quality

- Timestamp (UTC): `2026-07-19T07:33:24.244841+00:00`
- Device: `mps` | torch `2.12.0` | git `ad83671`
- Configs: baseline_fp32, fp16_disk
- Images × edits: 2 × 2

## Disk (UNet + IP tree)

| Label | GiB | Exists |
|---|---:|:---:|
| swiftedit_weights_fp32 | 9.7886 | True |
| swiftedit_weights_fp16 | 4.9421 | True |
| **tiết kiệm fp16 vs fp32** | **49.5%** | |

## Memory after load

| Config | peak_alloc_mb | driver_used_mb | load_s |
|---|---:|---:|---:|
| baseline_fp32 | 12366 | 13008 | 25.87 |
| fp16_disk | 6567 | 7163 | 15.57 |

## Quality vs baseline_fp32 (cùng ảnh + prompt)

| Config | n | seconds_mean | PSNR↑ mean | PSNR min | map → bench cũ |
|---|---:|---:|---:|---:|---|
| baseline_fp32 | 4 | 31.396 | 99.0 | 99.0 | baseline_fp32 (quality_speed_bench_2026-06-17) |
| fp16_disk | 4 | 5.342 | 51.425 | 48.494 | gần improved_fp16_cache nhưng weights fp16 trên disk + torch_dtype load |

## File xuất để tải về / so báo cáo

- `disk.csv`, `memory.csv`, `quality.csv`, `quality_summary.csv`
- `run_meta.json`, `report.md`, `COMPARE_WITH_PREV.md`
- `edited_images/<config>/`
- `bundle.zip` — nén toàn bộ thư mục này

So với chạy cũ: mở `experimental_data/quality_speed_bench_2026-06-17/report.md`.

