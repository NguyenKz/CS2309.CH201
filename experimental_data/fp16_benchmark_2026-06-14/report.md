# Benchmark fp16 / channels_last (SwiftEdit-RT) — 2026-06-14

Thực nghiệm tối ưu suy luận SwiftEdit bằng **half precision (fp16)** và **channels_last**,
không train lại. Đo tốc độ từng công đoạn + kiểm chứng chất lượng so với fp32.

## Môi trường

| Mục | Giá trị |
|-----|---------|
| Thiết bị | Apple M4 (Mac), MPS backend |
| dtype | fp32 (baseline) / fp16 / fp16 + channels_last |
| VAE | **luôn fp32** (SD VAE fp16 dễ NaN/ảnh đen) |
| Ảnh nguồn | `000000000000.jpg` (PIE-Bench subset) |
| Source prompt | "a slanted mountain bicycle on the road in front of a building" |
| Số edit/cfg | 3 (median; bỏ 2 warmup) |

## Per-stage (median ms/edit)

| Stage | fp32 | fp16 | fp16_cl | fp16 vs fp32 |
|-------|-----:|-----:|--------:|----:|
| vae_encode¹ | 1010.6 | 539.3 | 534.3 | (fp32 VAE — nhiễu nhiệt) |
| inv_text_encode | 1796.9 | 68.2 | 69.3 | ~26× |
| unet_inverse | 9335.8 | 1048.6 | 1069.6 | ~8.9× |
| mask_estimate | 2.9 | 2.0 | 1.4 | — |
| gen_image_embeds (CLIP ViT-H) | 7301.9 | 289.1 | 185.1 | ~25× |
| gen_text_encode | 490.6 | 48.0 | 47.1 | ~10× |
| gen_unet | 11383.6 | 1109.7 | 1137.4 | ~10× |
| gen_vae_decode¹ | 11140.4 | 2598.2 | 2435.4 | (fp32 VAE — nhiễu nhiệt) |
| **Tổng stage** | **42462.6** | **5703.1** | **5479.6** | |

¹ `vae_encode`/`gen_vae_decode` chạy **fp32 ở mọi config**. Chênh lệch ở 2 dòng này là
**nhiễu nhiệt thuần**, không phải do dtype — dùng làm thước đo mức throttling của baseline.

## Wall-clock end-to-end (median s/edit)

| Config | Median (s) | Speedup biểu kiến |
|--------|-----------:|------------------:|
| fp32 | 39.93 | 1.00× |
| fp16 | 5.88 | 6.79× |
| fp16 + channels_last | 5.48 | 7.29× |

## Chất lượng so với fp32

| Config | PSNR (dB) | MSE | NaN/đen | Nhận xét |
|--------|----------:|----:|--------:|---------|
| fp16 | 45.3 | ~0.0000 | 0/3 | Gần như trùng fp32 |
| fp16 + channels_last | 45.6 | ~0.0000 | 0/3 | Gần như trùng fp32 |

PSNR ~45 dB ≫ 30 dB → khác biệt với fp32 **không nhìn thấy được bằng mắt**; không có ảnh
hỏng (NaN/đen). Ảnh minh họa: `sample_images/{fp32,fp16,fp16_cl}_0.png`.

## Diễn giải trung thực về con số speedup

Baseline fp32 trên Mac M4 **rất nhạy nhiệt**: đo 1 edit lúc máy nguội fp32 ≈ 19.5s, nhưng
khi chạy chuỗi nhiều edit, fp32 bị thermal throttling và tăng dần (35 → 40 → 53s). fp16
ngược lại **ổn định ~5.5s** ở mọi edit.

Bằng chứng throttling: hai stage VAE (fp32 ở cả 2 config) chạy chậm **~3.9×** ở config fp32
so với config fp16, dù **cùng dtype**. Phần chênh này là nhiệt, không phải tính toán.

Vì vậy nên hiểu speedup theo 2 mức:

| Điều kiện | Speedup fp16 | Ý nghĩa |
|-----------|-------------:|---------|
| Máy nguội, 1 edit (bảo thủ) | **~3.3×** | Lợi ích thuần từ half-precision compute |
| Chạy liên tục (thực tế batch) | **~6–7×** | Cộng thêm: fp16 sinh nhiệt thấp, không throttle |

Cả hai đều là kết quả thật và đáng kể. Đây là tối ưu **tác động end-to-end lớn nhất** từ
trước tới nay (cache embedding chỉ giảm phần phụ thuộc ảnh/source; fp16 tăng tốc **toàn bộ**
UNet + encoder).

**channels_last**: thêm ~5% và làm thời gian phẳng hơn (5.48s đều tăm tắp). Lợi ích nhỏ trên
MPS nhưng không có rủi ro chất lượng (PSNR 45.6 dB) → nên bật kèm fp16.

## Kết luận

- fp16 tăng tốc SwiftEdit **~3.3× (bảo thủ) đến ~7× (chạy liên tục)** trên Mac M4/MPS.
- Chất lượng **không đổi đáng kể** (PSNR ~45 dB so fp32, không NaN/đen).
- VAE giữ fp32 để an toàn; mọi module còn lại chạy fp16.
- channels_last nên bật kèm (lợi ích nhỏ, ổn định hơn, không hại chất lượng).

## File trong thư mục

| File | Nội dung |
|------|----------|
| `report.md` | Báo cáo này |
| `timing_fp32.jsonl` / `timing_fp16.jsonl` / `timing_fp16_cl.jsonl` | Log timing thô từng config |
| `sample_images/` | Ảnh kết quả fp32 / fp16 / fp16_cl (cùng edit) để so mắt |

## Tái lập

```bash
cd /Users/nguyenkz/Documents/code/CS2309.CH201
source .venv/bin/activate
python scripts/create_piebench_subset.py --max-samples 20   # nếu chưa có ảnh

python scripts/bench_dtype.py \
  --image data/PIE-Bench-subset20/annotation_images/0_random_140/000000000000.jpg \
  --src-p "a slanted mountain bicycle on the road in front of a building" \
  --configs fp32 fp16 fp16_cl \
  --edits \
    "a slanted rusty mountain motorcycle in front of a fence" \
    "a slanted blue mountain bicycle on the road" \
    "a slanted mountain bicycle in front of a castle"

cat results/dtype_bench_report.md
```

## Code

- `SwiftEdit/models.py`: tham số `dtype` + `channels_last` cho `InverseModel`, `AuxiliaryModel`,
  `IPSBV2Model`; helper `resolve_dtype`/`module_dtype`; cast ranh giới dtype trong `gen_img`.
- `SwiftEdit/infer.py`: `vae.encode` dùng `vae.dtype`; `unet_inverse` cast input theo dtype unet;
  hậu xử lý về fp32 cho ổn định số học.
- `SwiftEdit/src/mask_ip_controller.py`: cast mask theo dtype runtime để chạy fp16.
- `scripts/bench_dtype.py`: script benchmark + đo chất lượng.
