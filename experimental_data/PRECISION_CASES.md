# Kiểm kê case precision — CS2309 SwiftEdit

> Cập nhật: 2026-07-19 (`fp16_weight_xformers` thêm; FP4 = ablation).  
> **Quyết định FP4:** dừng làm hướng tối ưu chính trên T4 — xem [`FP4_DECISION_AND_NEXT_PLAN.md`](./FP4_DECISION_AND_NEXT_PLAN.md).

## Hệ quy chiếu thống nhất

- Dataset / jobs: [`data/jobs_june17.json`](../data/jobs_june17.json) (200×3)
- Mỗi lần Colab **chỉ chạy 1 config** → 1 `bundle.zip`
- So sánh cuối **local**: `python scripts/compare_precision_runs.py`
- Baseline chất lượng: ảnh `baseline_fp32` trong bundle lần 1 (cùng `jobs_hash`)

## Workflow Colab khuyến nghị (T4 16GB)

| Lần | SELECT | Prepare | Ghi chú |
|-----|--------|---------|---------|
| 1 | `fp32` | verify sau setup tải Qualcomm | Reference PSNR + thời gian/VRAM |
| 2 | `fp16_weight` | có fp32 → **convert** cây fp16 (một lần) | Disk ~½; VRAM ~bằng fp16 compute — **giữ nguyên** |
| 3 | `fp16_weight_xformers` | **reuse** cùng cây fp16 | xFormers MEA; so trực tiếp với lần 2 |
| (tuỳ chọn) | `fp4_weight` | reuse fp16 | Ablation âm trên T4; không ưu tiên |

Cùng `MAX_JOBS=600`. Không bật nhiều config một lúc trên T4.

Notebook: [`notebooks/CS2309_SwiftEdit_precision_disk_vram.ipynb`](../notebooks/CS2309_SwiftEdit_precision_disk_vram.ipynb).

## Catalog (picker)

| Alias | Config | Disk | EditCache | xFormers |
|-------|--------|------|-----------|----------|
| `fp32` | `baseline_fp32` | fp32 | off | off |
| `fp16` | `improved_fp16_cache` | fp32 | on | off |
| `fp8` | `improved_fp8_cache` | fp32 | on | off |
| `fp4` | `improved_fp4_cache` | fp32 | on | off |
| `fp16_weight` | `fp16_disk` | **fp16** | on | off |
| `fp16_weight_xformers` | `fp16_disk_xformers` | **fp16** | on | **on** |
| `fp4_weight` | `fp4_from_fp16` | **fp16** → quant fp4 | on | off |

## Ma trận case

| # | Case | Status |
|---|------|--------|
| 1 | Full FP32 | Chạy picker `fp32` |
| 2 | FP16 compute + cache | Alias `fp16` (disk vẫn fp32) |
| 3 | FP8 + cache | Alias `fp8` — T4 thường chất lượng kém |
| 4 | FP4 compute + cache | Alias `fp4` — quant từ disk fp32 |
| 5 | FP16 **disk** | Alias `fp16_weight` — baseline disk/load (**không xóa**) |
| 6 | FP4 from fp16 disk | Alias `fp4_weight` — **ablation âm trên T4** |
| 7 | FP16 disk + xFormers MEA | Alias `fp16_weight_xformers` — cùng weight với case 5; đo Δ tốc độ/VRAM |

**Case 7:** inverse UNet dùng Diffusers xFormers; gen UNet: self-attn + nhánh không ARaM controller qua `xformers.ops.memory_efficient_attention` (không ghi đè IP processors). Cross-attn có mask controller vẫn einsum (ARaM).

**Case 6 (FP4):** Linear-only FP4 không nhanh hơn fp16 trên T4; baseline vận hành case 5 hoặc 7.

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
- Lần 2–3 `*_weight` / `fp16_weight_xformers`: không tải lại Qualcomm nếu đã có fp32; convert/reuse fp16 tại prepare.
- Script: [`scripts/prepare_colab_weights.py`](../scripts/prepare_colab_weights.py).
