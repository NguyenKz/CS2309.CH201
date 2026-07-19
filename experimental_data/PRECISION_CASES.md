# Kiểm kê case precision — CS2309 SwiftEdit

> Cập nhật: 2026-07-19 (workflow 3 lần: fp32 → fp16_weight → fp4_weight).

## Hệ quy chiếu thống nhất

- Dataset / jobs: [`data/jobs_june17.json`](../data/jobs_june17.json) (200×3)
- Mỗi lần Colab **chỉ chạy 1 config** → 1 `bundle.zip`
- So sánh cuối **local**: `python scripts/compare_precision_runs.py`
- Baseline chất lượng: ảnh `baseline_fp32` trong bundle lần 1 (cùng `jobs_hash`)

## Workflow Colab khuyến nghị (T4 16GB)

| Lần | SELECT | Prepare | Ghi chú |
|-----|--------|---------|---------|
| 1 | `fp32` | verify sau setup tải Qualcomm | Reference PSNR + thời gian/VRAM |
| 2 | `fp16_weight` | có fp32 → **convert** cây fp16 (một lần) | Disk ~½; VRAM ~bằng fp16 compute |
| 3 | `fp4_weight` | **reuse** fp16 nếu đã có | Load fp16 disk → quant **fp4** (bitsandbytes, CUDA) |

Cùng `MAX_JOBS=600`. Không bật nhiều config một lúc trên T4.

Notebook: [`notebooks/CS2309_SwiftEdit_precision_disk_vram.ipynb`](../notebooks/CS2309_SwiftEdit_precision_disk_vram.ipynb).

## Catalog (picker)

| Alias | Config | Disk | EditCache |
|-------|--------|------|-----------|
| `fp32` | `baseline_fp32` | fp32 | off |
| `fp16` | `improved_fp16_cache` | fp32 | on |
| `fp8` | `improved_fp8_cache` | fp32 | on |
| `fp4` | `improved_fp4_cache` | fp32 | on |
| `fp16_weight` | `fp16_disk` | **fp16** | on |
| `fp4_weight` | `fp4_from_fp16` | **fp16** → quant fp4 | on |

## Ma trận case

| # | Case | Status |
|---|------|--------|
| 1 | Full FP32 | Chạy picker `fp32` (đừng trộn số June17 cũ nếu không tin) |
| 2 | FP16 compute + cache | Alias `fp16` (disk vẫn fp32) |
| 3 | FP8 + cache | Alias `fp8` — T4 thường chất lượng kém |
| 4 | FP4 compute + cache | Alias `fp4` — quant từ disk fp32 |
| 5 | FP16 **disk** | Alias `fp16_weight` — VRAM ~ fp16; tối ưu disk/load |
| 6 | FP4 from fp16 disk | Alias `fp4_weight` — pipeline sẵn; chờ bundle full 600 để ghi số liệu cuối |

Kỳ vọng case 6 vs case 5: VRAM thấp hơn (weight-only fp4); disk dùng chung cây fp16; chất lượng/PSNR trade-off (so ảnh fp32 qua merge local).

## Export mỗi run (`precision_run_<stamp>_<configs>/`)

| File | Nội dung |
|------|----------|
| `inputs/jobs.json` + `inputs/images/` | input (prompt + ảnh nguồn) |
| `edited_images/<config>/` | output PNG theo `job_id` |
| `quality.csv` | duration, cache hit/miss, VRAM/job, PSNR nếu có |
| `memory.csv` | peak sau load / warmup |
| `disk.csv` | dung lượng cây weights |
| `run_meta.json` / `inventory.json` | `jobs_hash` để join |
| `bundle.zip` | toàn bộ thư mục |

## Bug dtype đã fix

`Half != float` tại IP `to_k_ip` khi load fp16 disk — đã ép dtype trong `SwiftEdit/models.py`.

## Tải weights

- Lần 1: Qualcomm trong **setup** (curl nhanh, xóa `.part` sau extract).
- Lần 2–3 `*_weight`: không tải lại Qualcomm nếu đã có fp32; convert/reuse fp16 tại prepare.
- Script: [`scripts/prepare_colab_weights.py`](../scripts/prepare_colab_weights.py).
