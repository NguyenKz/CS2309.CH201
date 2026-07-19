# Lưu ý: run 2026-07-19 đầu (img1_edit0.png) DÙNG SAI PROMPT

Ảnh `smoke_woman.jpg` (chân dung phụ nữ) bị gán prompt xe đạp:

- source: `a slanted mountain bicycle...`
- edit: `a slanted rusty mountain motorcycle...`

→ Output “mặt kỳ / texture gỗ” là **failure case edit sai ngữ cảnh**, không phải lỗi fp16.
PSNR fp16 vs fp32 vẫn ~52 dB (= hai bên giống nhau, cùng hỏng theo prompt).

Đã sửa `run_precision_disk_vram_eval.py` để lấy prompt từ `mapping_file.json`
(`woman` → `Taylor Swift`, `dog` → `dog with mouth opened`).

Xem run mới: `experimental_data/precision_disk_vram_2026-07-19_promptfix/`
