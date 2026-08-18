# Demo UI Gradio — SwiftEdit-RT (2026-06-14)

Giao diện web cho SwiftEdit, tích hợp toàn bộ tối ưu đã làm trong hướng SwiftEdit-RT:
**fp16 + channels_last + EditCache**. Mục tiêu: trình diễn chỉnh sửa ảnh bằng prompt
ở tốc độ gần realtime trên Mac, có hiển thị thời gian từng lần chạy.

## Môi trường

| Mục | Giá trị |
|-----|---------|
| Thiết bị | Apple M4 (Mac), MPS backend |
| dtype mặc định | fp16 + channels_last (VAE giữ fp32) |
| Framework UI | Gradio 5.50 |
| Python | 3.12 (.venv) |
| Script | `scripts/app_gradio.py` |

> Lưu ý dependency: Gradio 6 kéo `huggingface-hub` lên 1.x làm vỡ `transformers 4.57`.
> Đã ghim `gradio>=5,<6` + `huggingface-hub<1.0` trong `requirements-mac.txt`.

## Tính năng

| Thành phần | Mô tả |
|-----------|-------|
| Upload ảnh nguồn | Kéo-thả/chọn file (đưa vào `edit_image` dạng filepath) |
| Source prompt | Mô tả ảnh gốc |
| Edit prompt | Mô tả thay đổi mong muốn |
| Slider nâng cao | `scale_edit`, `scale_non_edit`, `mask_threshold` |
| Checkbox cache | Bật/tắt `EditCache` (tăng tốc khi cùng ảnh + source prompt) |
| Panel kết quả | Ảnh edit + thời gian chạy + dtype + trạng thái cache (hit/nạp mới) |
| Examples | Prompt mẫu sẵn để thử nhanh |

## Kết quả kiểm thử (self-test, không mở server)

Chạy đúng code path của app qua `--selftest`:

```
device=mps dtype=fp16 channels_last=True — nạp model xong.
[selftest] OK -> results/app_selftest.png
Thời gian: 7.76s | mps | fp16 + channels_last (VAE fp32) | Cache: nạp mới
```

- Ảnh kết quả đúng (1 ảnh edit, không ghép grid): `sample_output.png`.
- ~7.76s cho **edit đầu tiên** (gồm chi phí compile MPS + nạp cache). Các edit sau trên
  cùng ảnh + source prompt nhanh hơn nhờ cache; bản thân fp16 đã cho ~5.5s/edit ở trạng
  thái ổn định (xem `experimental_data/fp16_benchmark_2026-06-14/`).

![Ảnh edit demo](./sample_output.png)

## Cách chạy

```bash
cd /Users/nguyenkz/Documents/code/CS2309.CH201
source .venv/bin/activate
pip install -r requirements-mac.txt          # nếu chưa có gradio

python scripts/app_gradio.py                  # fp16 (mặc định) trên MPS
# python scripts/app_gradio.py --dtype fp32   # so baseline
# python scripts/app_gradio.py --share        # link public tạm thời
```

Mở `http://127.0.0.1:7860`. Mẹo demo: giữ nguyên 1 ảnh + source prompt, đổi nhiều edit
prompt liên tiếp để thấy dòng "Cache: cache hit" và thời gian giảm.

## Kiểm thử không cần mở server (CI/sandbox)

```bash
python scripts/app_gradio.py --selftest \
  data/PIE-Bench-subset20/annotation_images/0_random_140/000000000000.jpg
```

In thời gian + lưu `results/app_selftest.png`. Hữu ích khi môi trường không bind được localhost.

## File trong thư mục

| File | Nội dung |
|------|----------|
| `report.md` | Báo cáo này |
| `sample_output.png` | Ảnh edit từ self-test (fp16) |

## Code liên quan

- `scripts/app_gradio.py`: app Gradio + chế độ `--selftest`.
- `SwiftEdit/infer.py`: `EditCache`, `edit_image(..., cache=)`.
- `SwiftEdit/models.py`: tham số `dtype`/`channels_last`.
- `requirements-mac.txt`: thêm `gradio>=5,<6`, `pandas`.
