# Báo cáo precision cuối (hợp nhất)

- Sinh bởi `scripts/compare_precision_runs.py`
- `jobs_hash` dùng: `None`
- Số run gộp: **2**
- Ảnh FP32 làm baseline PSNR: **6** job

## Checklist coverage

| Config | Có data? | Nguồn run |
|---|:---:|---|
| `baseline_fp32` | YES | precision_disk_vram_2026-07-19_promptfix |
| `improved_fp16_cache` | NO | — |
| `improved_fp8_cache` | NO | — |
| `improved_fp4_cache` | NO | — |
| `fp16_disk` | YES | precision_disk_vram_2026-07-19_promptfix |
| `fp4_from_fp16` | NO | — |

## Hiệu năng / resource / chất lượng (vs FP32)

| Config | n | s/edit | cache hit | VRAM peak MB | PSNR↑ vs FP32 | PSNR nguồn | Disk tree |
|---|---:|---:|---:|---:|---:|---|---|
| `baseline_fp32` | 2 | 20.886 | None | 12366.27 | 99.0 | in-run | 9.7886 GiB |
| `fp16_disk` | 2 | 5.464 | None | 6567.03 | 48.768 | in-run | 4.9421 GiB |

## Speedup vs baseline_fp32

- `baseline_fp32`: **1.00×**
- `fp16_disk`: **3.82×**

## Còn thiếu

- `improved_fp16_cache` — chạy Colab `--configs improved_fp16_cache` (cùng MAX_JOBS / jobs_june17), tải bundle về
- `improved_fp8_cache` — chạy Colab `--configs improved_fp8_cache` (cùng MAX_JOBS / jobs_june17), tải bundle về
- `improved_fp4_cache` — chạy Colab `--configs improved_fp4_cache` (cùng MAX_JOBS / jobs_june17), tải bundle về
- `fp4_from_fp16` — chạy Colab `--configs fp4_from_fp16` (cùng MAX_JOBS / jobs_june17), tải bundle về

## Cách bổ sung data

1. Notebook: chọn **một** (hoặc vài) config trong `EVAL_CONFIGS`.
2. Cell ensure weights (tự convert nếu `*_weight`).
3. Eval → tải `bundle.zip` → giải nén vào `experimental_data/`.
4. Chạy lại script này.

Không dùng số June17 cũ nếu bạn không tin — chỉ dùng các `precision_run_*` mới cùng `jobs_hash`.

