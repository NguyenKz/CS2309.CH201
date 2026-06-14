# Thực nghiệm SwiftEdit — PIE-Bench subset 20 mẫu

Dữ liệu thực nghiệm cho lần kiểm thử ngày **2026-06-14**: đo thời gian từng công đoạn
và chất lượng chỉnh sửa của SwiftEdit trên 20 mẫu PIE-Bench (2 mẫu/loại × 10 loại edit).

## Môi trường

| Mục | Giá trị |
|-----|---------|
| Thiết bị | Apple M4 (Mac16,12), 24 GB RAM |
| Backend | PyTorch MPS (`mps`) |
| Hệ điều hành | macOS 26.5 |
| Python | 3.12.10 |
| PyTorch / transformers / diffusers | 2.12.0 / 4.57.6 / 0.35.2 |

## Model (stack SwiftEdit)

| Thành phần | Checkpoint / nguồn |
|-----------|--------------------|
| Inverse UNet | `inverse_ckpt-120k` (unet_ema), base `stabilityai/sd-turbo`, fp32 |
| Generation UNet | `sbv2_0.5` (SwiftBrush v2, one-step) |
| IP-Adapter | `ip_adapter_ckpt-90k/ip_adapter.bin` |
| VAE / Text encoder / Tokenizer | `Manojb/stable-diffusion-2-1-base` (SD 2.1) |
| CLIP image encoder | `h94/IP-Adapter` (ViT-H) |

## Dữ liệu

- Nguồn: HuggingFace `UB-CVML-Group/PIE_Bench_pp` (700 mẫu) → lấy subset 20 mẫu.
- Subset tạo bằng `scripts/create_piebench_subset.py` → `data/PIE-Bench-subset20/`.
- Ảnh 512×512, một lần gọi `edit_image()` (1-step SwiftEdit).

## Kết quả tóm tắt

**Tốc độ** (20 mẫu): trung bình **69.0 s/ảnh**; steady-state (bỏ 2 mẫu warmup MPS) **73.6 s**.
Bottleneck: 2× UNet (~43%) + IP image embeds (~24%) + VAE decode (~23%).

**Chất lượng** (PIE-Bench metrics):

| Metric | Mean | Số mẫu |
|--------|-----:|------:|
| CLIP-Whole | 23.02 | 20 |
| CLIP-Edited | 21.46 | 20 |
| PSNR vùng nền | 14.01 | 9 |
| MSE vùng nền | 0.06 | 9 |

PSNR/MSE chỉ tính được trên 9 mẫu có vùng nền (mask không phủ toàn ảnh).

## File trong thư mục

| File | Nội dung |
|------|----------|
| `metrics.csv` | Gộp runtime + timing chính + chất lượng theo từng mẫu |
| `timing_report.md` | Báo cáo thời gian từng công đoạn (đầy đủ) |
| `timing_report.csv` | Bảng timing dạng CSV |
| `timing_raw.jsonl` | Log thô từng mẫu (mỗi dòng 1 JSON) |
| `edited_images/` | 20 ảnh kết quả, chia theo loại edit |

## Cách chạy & chứng minh (tái lập)

Môi trường: Mac M4 (MPS), Python `.venv`, weights trong `SwiftEdit/swiftedit_weights/`.

### 1. Chuẩn bị

```bash
cd /Users/nguyenkz/Documents/code/CS2309.CH201
source .venv/bin/activate
# Tải subset 20 mẫu từ HuggingFace UB-CVML-Group/PIE_Bench_pp
python scripts/create_piebench_subset.py --max-samples 20
```

### 2. Chạy eval + đo timing (lệnh chính)

```bash
export SWIFTEDIT_TIMING_LOG=results/timing_bench20.log
python scripts/run_piebench_eval.py \
  --piebench-dir data/PIE-Bench-subset20 \
  --max-samples 20 \
  --no-resume
```

> `--no-resume` bắt buộc chạy lại inference (mặc định `resume` sẽ bỏ qua mẫu đã có ảnh).
> Bỏ `export ...` thì log timing ghi vào `results/timing.log`.

### 3. Tổng hợp báo cáo (timing + chất lượng)

```bash
python scripts/summarize_timing.py \
  --log results/timing_bench20.log \
  --metrics-csv results/piebench/metrics.csv \
  --out results/timing_report_20.md
```

> `--metrics-csv` mặc định đã là `results/piebench/metrics.csv` nên có thể bỏ.

### 4. Đọc kết quả + so với số liệu đã commit

```bash
cat results/timing_report_20.md
cat results/piebench/metrics.csv
diff -u experimental_data/piebench_subset20_2026-06-14/timing_report.md results/timing_report_20.md || true
```

Kỳ vọng (Mac M4/MPS): TB **~69 s/ảnh**; bottleneck 2× UNet (~43%) + IP image embeds (~24%)
+ VAE decode (~23%); CLIP-Whole ~23.0, CLIP-Edited ~21.5. Chênh lệch thời gian giữa các
lần chạy là bình thường (thermal throttling); thứ tự bottleneck giữ ổn định.
