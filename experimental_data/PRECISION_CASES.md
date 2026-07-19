# Kiểm kê case precision — CS2309 SwiftEdit

> Cập nhật: 2026-07-19. Mục tiêu: biết **đã có số liệu** vs **còn thiếu** trước khi chạy thêm trên Colab.

## Ma trận case

| # | Case | Cách làm | Dataset / quy mô | Tốc độ | VRAM | PSNR vs fp32 | Disk weights | Trạng thái |
|---|------|----------|------------------|--------|------|--------------|--------------|------------|
| 1 | Full **FP32** | native dtype | Colab T4, 200×3=600 | có | có | reference | fp32 ~10GB | **ĐỦ** — [`quality_speed_bench_2026-06-17`](./quality_speed_bench_2026-06-17/) |
| 2 | **FP16 compute** (cast trên GPU, disk vẫn fp32) | `dtype=fp16` + channels_last + cache | Colab T4, 600 | có (1.70×) | −42% | 48.6 dB | không giảm | **ĐỦ** — cùng thư mục 2026-06-17 (`improved_fp16_cache`) |
| 3 | **FP8** weight-only | torchao sau load | Colab T4, 600 | có | −46% | 6.0 dB (hỏng) | không giảm | **ĐỦ** (biết là fail) — 2026-06-17 |
| 4 | **FP4** weight-only (runtime) | bitsandbytes sau load fp32→fp16 | Colab T4, 600 | có | −48.5% | 21.7 dB | không giảm | **ĐỦ** — 2026-06-17 |
| 5 | **FP16 disk** (nén checkpoint) | convert safetensors fp16 + `torch_dtype` load | Mac MPS smoke 2 job | có (MPS) | peak load −47% (MPS) | ~51 dB (smoke) | **−49.5%** | **ĐỦ smoke** — [`precision_disk_vram_2026-07-19_promptfix`](./precision_disk_vram_2026-07-19_promptfix/); **THIẾU** cùng dataset 200×3 trên T4 |
| 6 | **FP4 from fp16 disk** | load weights_fp16 rồi `quant=fp4` | — | — | — | — | dùng tree fp16 | **THIẾU** (Colab chưa chạy xong / lỗi) |

Bổ sung Mac nhỏ: [`fp16_benchmark_2026-06-14`](./fp16_benchmark_2026-06-14/) — FP16 compute trên MPS (không phải disk).

## Kết luận kiểm kê

**Đã đủ để viết phần “precision compute / quant runtime”** (case 1–4) từ báo cáo 2026-06-17.

**Chưa đủ cho “nén checkpoint trên disk”:**

1. **Case 5 trên Colab T4, cùng (hoặc tương đương) dataset bench cũ** — để so VRAM CUDA + tốc độ với case 2.
2. **Case 6** `fp4_from_fp16` trên T4 — so với case 4 (fp4 từ weights fp32 disk).
3. **Báo cáo tổng hợp một bảng** gộp case 1–6 (script so sánh local — xem `scripts/compare_precision_runs.py`).

**Không cần chạy lại case 1 (FP32)** — dùng ảnh/CSV 2026-06-17 làm mốc (khi dataset khớp) hoặc so PSNR chỉ giữa các config trong cùng một run mới.

## Kế hoạch chạy tiếp (Colab) — tiết kiệm disk

```
configs mặc định: fp16_disk,fp4_from_fp16
(không baseline_fp32)
→ convert/lấy weights_fp16 → eval → bundle.zip → tải về
→ local: compare_precision_runs.py + viết report
```

## File bằng chứng chính

| Run | Vai trò |
|-----|---------|
| `quality_speed_bench_2026-06-17/` | Case 1–4 đầy đủ T4 |
| `precision_disk_vram_2026-07-19_promptfix/` | Case 5 smoke Mac (prompt đúng) |
| `precision_disk_vram_2026-07-19/` | Case 5 smoke **prompt sai** — chỉ để học lỗi, không dùng số liệu báo cáo chất lượng |
