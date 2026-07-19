# Kiểm kê case precision — CS2309 SwiftEdit

> Cập nhật: 2026-07-19 (workflow chọn version + merge local).

## Hệ quy chiếu thống nhất

- Dataset / jobs: [`data/jobs_june17.json`](../data/jobs_june17.json) (200×3)
- Mỗi lần Colab **chỉ chạy configs bạn chọn** → 1 `bundle.zip`
- So sánh cuối **local** khi đủ data: `python scripts/compare_precision_runs.py`
- Baseline chất lượng: ảnh `baseline_fp32` (có thể chạy riêng 1 session `--configs fp32`)

## Catalog (picker)

| Alias | Config | Disk | EditCache |
|-------|--------|------|-----------|
| `fp32` | `baseline_fp32` | fp32 | off |
| `fp16` | `improved_fp16_cache` | fp32 | on |
| `fp8` | `improved_fp8_cache` | fp32 | on |
| `fp4` | `improved_fp4_cache` | fp32 | on |
| `fp16_weight` | `fp16_disk` | **fp16** | on |
| `fp4_weight` | `fp4_from_fp16` | **fp16** → quant fp4 | on |

Cell notebook:

1. **Chọn** `SELECT` (True/False từng alias)
2. **Ensure weights** — chỉ convert/link khi có `*_weight`
3. **Eval** → tải `bundle.zip`

## Export mỗi run (`precision_run_<stamp>_<configs>/`)

| File | Nội dung |
|------|----------|
| `inputs/jobs.json` + `inputs/images/` | input (prompt + ảnh nguồn) |
| `edited_images/<config>/` | output PNG theo `job_id` |
| `quality.csv` | duration, cache hit/miss, VRAM/job, PSNR nếu có fp32 trong cùng run |
| `memory.csv` | peak sau load / warmup |
| `disk.csv` | dung lượng cây weights |
| `run_meta.json` / `inventory.json` | `jobs_hash` để join các run |
| `bundle.zip` | toàn bộ thư mục |

## Số liệu June17 cũ

Giữ tham khảo trong `quality_speed_bench_2026-06-17/` nhưng **không** trộn vào báo cáo cuối nếu bạn không tin (script merge chỉ đọc `precision_run_*` mới cùng `jobs_hash`).

## Bug dtype đã fix

`Half != float` tại IP `to_k_ip` khi load fp16 disk — đã ép dtype trong `SwiftEdit/models.py`. Cần pull code mới trước khi eval.

## Tối ưu tải weights (Colab)

| Trường hợp | Hành vi |
|------------|---------|
| Chỉ `*_weight` + Drive fp16 | Symlink Drive — **không** tải Qualcomm |
| Cần fp32 + Drive fp32 | Symlink Drive |
| Bắt buộc tải Qualcomm | `curl…\|tar` stream — **không** giữ `.part` + `.tar.gz` (tránh ~50GB→xóa 40GB) |

Script: `scripts/prepare_colab_weights.py` · notebook set `SKIP_QUALCOMM_IN_SETUP=True`.
