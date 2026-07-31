# Audit PSNR ~48.6 dB — fidelity precision, không phải “giống ảnh gốc”

> Trả lời góp ý: số PSNR cao có thể bị hiểu nhầm như so với ảnh nguồn / “trick”. Mục này ghi rõ protocol, code path và spot-check.

| | |
|---|---|
| **Bench** | `experimental_data/quality_speed_bench_2026-06-17/` |
| **Producer** | `notebooks/CS2309_SwiftEdit_quality_speed_bench.ipynb` |
| **CSV** | `quality_raw.csv` (1800 dòng = 600 edit × 3 config improved) |
| **Reference** | `baseline_fp32` — ảnh **đã edit** FP32 cùng `(job_id, prompt_idx)` |

## 1. Code so cái gì?

Trong notebook:

```
REFERENCE = "baseline_fp32"
m = qm.compare(pc[REFERENCE], pc[c])  # ref = FP32 edit, test = improved edit
```

`QualityMetrics` dùng torchmetrics `PeakSignalNoiseRatio(data_range=1.0)` trên ảnh RGB `[0,1]`, cùng kích thước 512×512.

**Không** so với JPEG nguồn trong `sample_source/` / `source_images/`.

## 2. Vì sao ~48 dB hợp lý?

Công thức (data_range=1): `PSNR ≈ 10 · log10(1 / MSE)`.

- MSE mean FP16 ↔ FP32 ≈ 0.000020 → PSNR ≈ 47–48 dB.
- Hai output gần trùng pixel sau khi chỉ đổi precision (+ cache/channels_last) → số cao là **kỳ vọng**, không phải ghép nhầm hai file giống hệt.

## 3. Spot-check 12 ảnh mẫu (repo)

So trên `sample_edits/` + `sample_source/`:

| Cặp | Mean PSNR (12 mẫu) |
|---|---|
| FP16 edit ↔ FP32 edit (cùng job) | **~47.1 dB** |
| FP16 edit ↔ ảnh nguồn | **~19.2 dB** |

Kết luận: 48 dB **không** có nghĩa “ảnh edit giống ảnh gốc như chó vs chuột cùng một ảnh”. Edit vẫn đổi nội dung mạnh so với nguồn (~19 dB); chỉ gần như trùng với bản FP32 cùng job.

## 4. Hai bảng metric — không trộn

| Vai trò | Độ đo | So sánh |
|---|---|---|
| **Fidelity tối ưu** (đề tài RT) | PSNR / SSIM / LPIPS / MSE | output config ↔ output FP32 cùng job |
| **Edit quality** (paper / PieBench) | CLIP-Whole / CLIP-Edited; PSNR/MSE trên `(1−mask)` | edited ↔ prompt; edited ↔ **source** (vùng nền) |

PieBench subset 20 (Mac MPS): xem `experimental_data/piebench_subset20_2026-06-14/EDIT_QUALITY_SUMMARY.md`.

## 5. Rủi ro trình bày (đã sửa wording)

- Tránh gọi PSNR 48.6 là “chất lượng chỉnh sửa” không qualifier.
- Tránh cụm dễ đọc thành “so ảnh gốc” khi ý là so FP32.
- Luôn nêu **mean (min–max)** + SSIM/LPIPS để không chỉ một số mean.
