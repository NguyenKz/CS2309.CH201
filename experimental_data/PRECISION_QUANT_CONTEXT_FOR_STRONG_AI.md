# Report — Phần nén FP4 trên SwiftEdit (Colab T4)

> **“PHẦN NÀY”** = phần **nén / quant xuống FP4** (`fp4` / `fp4_weight`), không phải toàn bộ pipeline precision.  
> **Trạng thái:** đã có phản biện ngoài + **quyết định đóng sổ** — xem [`FP4_DECISION_AND_NEXT_PLAN.md`](./FP4_DECISION_AND_NEXT_PLAN.md).  
> Ngày: 2026-07-19 · Repo CS2309.CH201 · GPU: Tesla T4 16GB

---

## 0. Kết luận ngắn (sau phản biện)

Weight-only FP4 (bitsandbytes) trên **Turing T4 không phải giải pháp nén đúng nghĩa** cho SwiftEdit:

- T4 không compute native FP4 → dequant on-the-fly sang FP16 rồi MatMul.
- UNet ảnh thường compute-bound → lợi băng thông bị triệt bởi dequant → tốc độ ≈ fp16.
- Chỉ `nn.Linear` → VRAM giảm ít; PSNR tụt mạnh (~21.7 vs ~48.6 fp16).
- **Dừng tối ưu FP4 trên T4.** Không lưu checkpoint FP4 trên disk.
- Baseline: **FP16 + EditCache** (`fp16` / `fp16_weight`).
- Hướng tiếp: **xFormers MEA → Token Merging (ToMe) → TensorRT FP16** (xếp hạng ROI trong plan).

---

## 1. Vấn đề quan sát ban đầu

Đã implement weight-only FP4 (`Linear4bit`) cho UNet inverse + gen. Lúc chạy `fp4_weight` trên Colab:

- GPU RAM ~10.5 / 15 GB
- ~1.3–1.7s/edit — không thấy nhanh hơn fp16
- Chất lượng kỳ vọng kém (bench cũ PSNR ~21.7 dB)

→ Mục tiêu nén (“nhẹ + nhanh”) không đạt.

---

## 2. Ngữ cảnh kỹ thuật tối thiểu

Quant chỉ áp `nn.Linear` qua `quantize_unet()` trong `SwiftEdit/models.py`. Không nén Conv, VAE (fp32), CLIP. `compute_dtype=fp16`. Không có weight FP4 trên disk — mỗi lần quant lại lúc load.

| Alias | Ý nghĩa |
|-------|---------|
| `fp4` | Disk fp32 → quant Linear→fp4 + EditCache |
| `fp4_weight` | Disk fp16 → quant Linear→fp4 + EditCache |

---

## 3. Số liệu

### Bench T4 June17 (600 edit/config)

| Config | s/edit | vs fp32 | VRAM peak (MB) | PSNR vs fp32 |
|--------|--------|---------|----------------|--------------|
| fp32 | 2.91 | 1.0× | ~14596 | — |
| fp16+cache | 1.71 | 1.70× | ~8446 | ~48.6 |
| fp4+cache | 1.74 | 1.68× | ~7515 | ~21.7 |

### Quan sát 2026-07-19 (`fp4_from_fp16`)

- GPU RAM 10.5 / 15.0 GB; job ~1.3–1.7s/edit — cùng band với cảm nhận fp16.

---

## 4. Câu hỏi đã hỏi AI / đã trả lời

1. Kết luận Linear-only FP4 trên Turing không phải nén đúng nghĩa? → **Có, dừng.**
2. Phương án thay thế ranked ROI? → xFormers → ToMe → TensorRT (chi tiết trong plan).
3. Viết ablation âm? → Có, câu mẫu trong plan §1.
4. Lưu checkpoint FP4? → **Không.**
5. Nén triệt để không train lại? → Không đi sâu FP4; chuyển graph/attention/token merge.

---

## 5. Code pointer

- `SwiftEdit/models.py` — `quantize_unet(..., quant="fp4")`
- `scripts/precision_catalog.py` — `fp4_weight` → `fp4_from_fp16`
- Plan tiếp theo: [`FP4_DECISION_AND_NEXT_PLAN.md`](./FP4_DECISION_AND_NEXT_PLAN.md)
