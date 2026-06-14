# Cache latent + CLIP image embed + source prompt embed — 2026-06-14

Thực nghiệm đo lợi ích của **embedding cache** (SwiftEdit-RT) cho kịch bản realtime:
cùng 1 ảnh nguồn + source prompt, người dùng đổi nhiều edit prompt liên tiếp.

## Cache cái gì

| Thành phần | Phụ thuộc | Lệnh sinh |
|---|---|---|
| `latents` (VAE encode ảnh nguồn) | ảnh | `inverse_model.vae.encode(...)` |
| CLIP image embed (`image_prompt_embeds`) | ảnh | `ip_sb_model.get_image_embeds(...)` |
| source prompt embed (inverse model) | source prompt | `inverse_model.text_encoder([src_p])` |
| source prompt embed (generation model) | source prompt | `aux_model.text_encoder` hàng `src` |

Tự invalidate khi đổi ảnh (latent + CLIP embed) hoặc đổi source prompt (2 text embed).
Mỗi edit mới chỉ encode lại edit prompt + chạy 2× UNet + VAE decode (không cache được).

## Kết quả chính (Mac M4 / MPS)

- Tiết kiệm **~9.93 s/edit** ở các stage phụ thuộc ảnh/source (giảm 62% riêng phần cacheable).
- Đóng góp lớn nhất: `gen_image_embeds` (CLIP image encoder) **−11.5 s**, `vae_encode` **−0.95 s**.
- Embed cache **deterministic** (allclose với tính lại từ đầu) → không đổi chất lượng.
- Wall-clock end-to-end nhanh hơn nhưng nhỏ và bị nhiễu thermal throttling
  (2× UNet + VAE decode vẫn chiếm đa số thời gian).

Chi tiết: [`report.md`](./report.md). Số liệu thô: [`timing_raw.jsonl`](./timing_raw.jsonl).

## Tái lập

```bash
cd /Users/nguyenkz/Documents/code/CS2309.CH201
source .venv/bin/activate
python scripts/create_piebench_subset.py --max-samples 20   # nếu chưa có ảnh nguồn

# Benchmark (3 edit, interleave no-cache/cache để khử thermal throttling)
python scripts/bench_cache.py \
  --image data/PIE-Bench-subset20/annotation_images/0_random_140/000000000000.jpg \
  --src-p "a slanted mountain bicycle on the road in front of a building" \
  --edits \
    "a slanted rusty mountain motorcycle in front of a fence" \
    "a slanted blue mountain bicycle on the road" \
    "a slanted mountain bicycle in front of a castle"

# Đọc kết quả + so với số liệu đã commit
cat results/cache_bench_report.md
diff -u experimental_data/cache_benchmark_2026-06-14/report.md results/cache_bench_report.md || true
```

Bỏ `--edits` để dùng 5 edit mặc định. Chi tiết lệnh + giải thích từng bước: xem mục
**"Cách chạy & chứng minh"** trong [`report.md`](./report.md).

## Code

- `SwiftEdit/infer.py`: class `EditCache` + tham số `cache=` của `edit_image`.
- `SwiftEdit/models.py`: tham số `embed_cache=` của `IPSBV2Model.gen_img`.
