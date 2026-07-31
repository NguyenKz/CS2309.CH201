# Edit quality — PieBench subset 20 (độ đo chỏi với PSNR↔FP32)

> Nguồn: `metrics.csv` trong thư mục này (Mac M4 MPS, 2026-06-14).  
> Đây là **chất lượng chỉnh sửa** (edited vs source / vs prompt), **không** phải fidelity precision FP16↔FP32.

## Protocol

| Độ đo | So sánh | Ý nghĩa |
|---|---|---|
| `clip_whole` | toàn ảnh edited ↔ `edit_prompt` | Semantics toàn cục |
| `clip_edited` | vùng mask edited ↔ `edit_prompt` | Semantics vùng sửa |
| `psnr_unedit` | edited ↔ **source** trên `(1 − mask)` | Giữ nền (chỉ mẫu có mask decode được) |
| `mse_unedit` |同上 | Sai số nền |
| `runtime_s` | wall-clock / ảnh | Tốc độ (Mac MPS, chưa tối ưu FP16+cache) |

## Tóm tắt số

| Metric | n | mean | min | max |
|---|---:|---:|---:|---:|
| CLIP-Whole ↑ | 20 | **23.02** | 16.47 | 32.60 |
| CLIP-Edited ↑ | 20 | **21.46** | 11.73 | 32.60 |
| PSNR nền (unedit) ↑ | 9 | **14.01** | 8.58 | 20.17 |
| MSE nền ↓ | 9 | 0.056 | 0.010 | 0.139 |
| Runtime (s) ↓ | 20 | 69.0 | 15.8 | 82.9 |

Ghi chú: 11/20 mẫu không có `psnr_unedit` (mask decode trống / không áp dụng) — chỉ báo trên 9 mẫu có mask.

## Đối chiếu với PSNR 48.6 (precision)

| Câu hỏi | Số điển hình | Protocol |
|---|---|---|
| FP16 có giống FP32 không? | PSNR **~48.6** (T4, n=600) | edit_fp16 ↔ edit_fp32 |
| Edit có giữ nền / khớp prompt không? | PSNR nền **~14**; CLIP-W **~23** | edited ↔ source / prompt |
| Edit có “copy ảnh gốc” không? | Spot-check edit↔source **~19 dB** | toàn ảnh, mẫu qsbench |

Hai nhóm số **chỏi nhau có chủ đích**: fidelity cao không chứng minh edit “hoàn hảo”; edit quality dùng CLIP + PSNR nền vs source.

## Paper Table 1 (tham chiếu, A100)

SwiftEdit paper (PIE-Bench đầy đủ): PSNR nền ~23.3, CLIP-Whole ~25.2, CLIP-Edited ~21.3, ~0.23 s.  
Subset 20 trên Mac MPS **không** phải tái hiện Table 1 đầy đủ — dùng để minh họa protocol metric edit-quality trong đề tài.
