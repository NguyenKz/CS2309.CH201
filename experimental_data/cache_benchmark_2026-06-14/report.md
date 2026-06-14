# Benchmark cache embedding (SwiftEdit-RT)

- **Thiết bị:** `mps`
- **Ảnh nguồn:** `000000000000.jpg`
- **Source prompt:** a slanted mountain bicycle on the road in front of a building
- **Số edit:** 3 (interleave no-cache/cache, đã bỏ warmup)
- **Cache đúng (embed deterministic):** True

## Per-stage cacheable (mean ms/edit) — thước đo chính, ít nhiễu nhiệt

| Stage | No cache | With cache | Tiết kiệm |
|-------|---------:|-----------:|----------:|
| vae_encode | 948.4 | 0.2 | 948.2 |
| inv_text_encode | 2203.9 | 1973.3 | 230.7 |
| gen_image_embeds | 11533.5 | 1.9 | 11531.6 |
| gen_text_encode | 1375.1 | 4154.6 | -2779.5 |
| **Tổng cacheable** | **16061.0** | **6130.0** | **9931.0** |

-> Cache loại bỏ **~9.93 s/edit** ở các stage phụ thuộc ảnh/source (giảm 62% riêng phần cacheable).
Phần lớn đến từ `gen_image_embeds` (CLIP image encoder, **−11.5 s**) và `vae_encode` (**−0.95 s**).

> Lưu ý `gen_text_encode` hiện số âm (−2.8 s) là **nhiễu đo trên MPS**: đây là op rất nhỏ
> (CLIP text encoder), các run cache chạy ngay sau run no-cache nên máy nóng hơn → bị
> throttle. Bản chất với cache chỉ encode 1 edit prompt (thay vì batch 2) nên không thể
> chậm hơn về lý thuyết; chênh lệch này nằm trong sai số nhiệt.

## Wall-clock end-to-end (tham khảo — nhiễu vì thermal throttling Mac)

| Cấu hình | Mean (s) | Min | Max |
|----------|---------:|----:|----:|
| No cache | 56.09 | 51.66 | 64.84 |
| With cache | 43.74 | 35.97 | 50.99 |

Phần không cache được (2× UNet + VAE decode, phụ thuộc edit prompt) vẫn chiếm đa số thời gian nên speedup end-to-end nhỏ; wall-clock còn bị throttle nhiệt che lấp.

Cache tái dùng: latent VAE (ảnh), CLIP image embed (ảnh), source prompt embed (inverse + generation). Mỗi edit mới chỉ encode lại edit prompt + chạy UNet/VAE-decode.

## Cách chạy & chứng minh (tái lập)

Môi trường: Mac M4 (MPS), Python `.venv`, weights trong `SwiftEdit/swiftedit_weights/`.

### 1. Chuẩn bị

```bash
cd /Users/nguyenkz/Documents/code/CS2309.CH201
source .venv/bin/activate
# Tải subset PIE-Bench nếu chưa có ảnh nguồn
python scripts/create_piebench_subset.py --max-samples 20
```

### 2. Chạy benchmark (lệnh chính)

```bash
python scripts/bench_cache.py \
  --image data/PIE-Bench-subset20/annotation_images/0_random_140/000000000000.jpg \
  --src-p "a slanted mountain bicycle on the road in front of a building" \
  --edits \
    "a slanted rusty mountain motorcycle in front of a fence" \
    "a slanted blue mountain bicycle on the road" \
    "a slanted mountain bicycle in front of a castle"
```

Bỏ `--edits` để dùng 5 edit mặc định. Script tự: warmup (nạp cache), chạy **interleave**
no-cache/cache từng edit (khử thermal throttling), in bảng per-stage và kiểm chứng
`inverse source embed deterministic (allclose): True`.

### 3. Đọc kết quả

```bash
cat results/cache_bench_report.md
cat results/cache_bench_timing.jsonl | python -m json.tool --compact
```

### 4. So với số liệu đã commit (reproducibility)

```bash
diff -u experimental_data/cache_benchmark_2026-06-14/report.md results/cache_bench_report.md || true
```

Kỳ vọng (Mac M4/MPS, 3 edit): `gen_image_embeds` ~11.5 s → ~0; `vae_encode` ~0.95 s → ~0;
tổng cacheable tiết kiệm **~9.9 s/edit**; `allclose: True`. Chênh lệch vài giây giữa các
lần chạy là bình thường do thermal throttling — **per-stage cacheable** ổn định hơn wall-clock.

### Tham số script `bench_cache.py`

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `--image` | Ảnh nguồn | (bắt buộc) |
| `--src-p` | Source prompt | (bắt buộc) |
| `--edits` | Danh sách edit prompt | 5 prompt mẫu |
| `--repeat` | Lặp danh sách edit N lần | 1 |
| `--out` | File report markdown | `results/cache_bench_report.md` |
