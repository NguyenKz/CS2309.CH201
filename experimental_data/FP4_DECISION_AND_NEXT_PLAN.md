# Quyết định FP4 + kế hoạch nén/tăng tốc tiếp theo (T4)

> Cập nhật: 2026-07-19  
> Phạm vi: phần **nén FP4** và hướng thay thế trên Tesla T4.  
> Nguồn: quan sát Colab + [report phản biện](./PRECISION_QUANT_CONTEXT_FOR_STRONG_AI.md) (AI ngoài).

---

## 1. Quyết định về FP4

**Dừng phát triển FP4 như hướng tối ưu chính trên T4.** Giữ code/config hiện có chỉ để ablation / số liệu âm trên báo cáo.

### Vì sao

| Quan sát | Ý nghĩa |
|----------|---------|
| T4 (Turing) không compute native FP4 | bitsandbytes FP4 + `compute_dtype=fp16` = đọc 4-bit rồi dequant → MatMul FP16 |
| UNet ảnh thường compute-bound | Lợi băng thông bị triệt bởi chi phí dequant → tốc độ ≈ fp16, không thắng |
| Chỉ quant `nn.Linear`, bỏ Conv/VAE/CLIP | VRAM tiết kiệm ít (~1GB so fp16; ~12% band so peak) |
| PSNR vs fp32 (~21.7) vs fp16 (~48.6) | Quality tụt mạnh không đổi được bằng “nén nhẹ” |
| GPU RAM Colab ~10.5/15 GB lúc `fp4_weight` | Không đạt mục tiêu nén “nhẹ + nhanh” |

### Không làm tiếp (trừ khi đổi GPU có native FP4 / có paper mới)

- Không lưu checkpoint FP4 trên disk (thêm debt, runtime/VRAM không cải thiện).
- Không tối ưu thêm `Linear4bit` trên T4 với kỳ vọng tăng tốc.

### Giữ trên báo cáo

Câu kết luận đề xuất:

> Trên Turing (Tesla T4), weight-only FP4 (bitsandbytes) cho lớp Linear không mang lại hiệu quả nén tổng thể. Phần cứng không hỗ trợ native FP4; chi phí dequant về FP16 triệt tiêu lợi băng thông. Chỉ Linear nên VRAM giảm hạn chế (~12%) trong khi chất lượng giảm mạnh (PSNR ~21.7 so với ~48.6 của fp16). Baseline tối ưu trên T4: **FP16 + EditCache** (và `fp16_weight` nếu cần giảm dung lượng disk).

Baseline khuyến nghị vận hành: **`fp16` / `fp16_weight` + EditCache**.

---

## 2. Xếp hạng giải pháp thay thế

Tiêu chí: **độ đơn giản** (thời gian tích hợp trên SwiftEdit Colab T4, ít đụng pipeline IP-Adapter) × **hiệu quả kỳ vọng** (tăng tốc và/hoặc giảm peak VRAM, chất lượng gần fp16).

Thang: Đơn giản 1–5 (5 = dễ nhất), Hiệu quả 1–5 (5 = tiềm năng cao nhất trên T4). **Điểm ROI** = Đơn giản + Hiệu quả (chỉ để xếp nội bộ đề tài).

| Hạng | Giải pháp | Đơn giản | Hiệu quả | ROI | Chất lượng kỳ vọng | Ghi chú repo |
|------|-----------|:--------:|:--------:|:---:|--------------------|--------------|
| **0 (đang có)** | FP16 + EditCache (+ `fp16_weight`) | 5 | 4 | **9** | Gần fp32 (PSNR cao) | Đã đo; baseline chính |
| **1** | xFormers Memory-Efficient Attention | 4–5 | 3–4 | **7–9** | Giữ (không đổi precision) | T4 hỗ trợ MEA; FlashAttention v2 không. Cần kiểm tra đã bật trong Diffusers UNet chưa |
| **2** | Token Merging (ToMe / `tomesd`) | 4 | 3–4 | **7–8** | Trade-off theo `ratio` | Patch nhanh; tune ratio đo PSNR |
| **3** | `torch.compile` (UNet) | 3 | 2–3 | **5–6** | Thấp rủi ro | Đã liệt kê trong `HUONG_PHAT_TRIEN`; warmup overhead trên Colab |
| **4** | TinyVAE / TAESD | 3 | 2–3 | **5–6** | Có thể đổi màu/chi tiết | Ablation riêng; nhắm stage VAE decode (~23% pipeline Mac) |
| **5** | TensorRT FP16 (2× UNet) | 1–2 | 4–5 | **5–7** | Gần giữ nếu FP16 | ROI dài hạn cao; effort lớn (ONNX/`torch-tensorrt`, IP-Adapter) |
| **— (đã loại trên T4)** | bitsandbytes FP4 Linear-only | 3 | 1 | **4** | Kém | Negative result — không ưu tiên |

### Cách đọc bảng

1. **Không bỏ baseline 0** — mọi hướng mới đo **so với fp16 + EditCache**.
2. **1 → 2 trước** (một buổi Colab): xác nhận xFormers + thử ToMe; ít rủi ro, ROI cao.
3. **5 TensorRT** chỉ khi cần vắt thêm sau khi 1–2 đã có số liệu; không làm song song FP4.
4. TinyVAE / compile = phụ, không thay thế quyết định dừng FP4.

---

## 3. Kế hoạch thực hiện (đề xuất)

### Giai đoạn A — Đóng sổ FP4 (ngắn)

- [x] Kết luận kiến trúc + ablation âm (file này + context).
- [ ] Hoàn tất / ghi rõ: bundle `fp4_weight` đủ việc báo cáo **hoặc** dùng số June17 `improved_fp4_cache` làm chứng cứ đã có; không bắt buộc chạy thêm nếu quota Colab hết.
- [ ] Không thêm luồng save/load checkpoint FP4.

### Giai đoạn B — Quick wins trên T4 (ưu tiên code tiếp theo)

| Bước | Việc | Tiêu chí xong |
|------|------|----------------|
| B1 | Audit attention backend | Ghi backend trước/sau |
| B2 | **`fp16_weight_xformers`** (`fp16_disk_xformers`) | Catalog + notebook + MEA wire — **đã code 2026-07-19**; chờ bundle Colab so `fp16_weight` |
| B3 | Inject ToMe (`tomesd`) với vài `ratio` | Bảng ratio → speed / VRAM / PSNR |
| B4 | Chọn combo ổn định làm “improved-RT” | Bundle + compare local |

**Chạy Colab B2:** SELECT `fp16_weight_xformers` (cùng cây `swiftedit_weights_fp16` như `fp16_weight`). So s/edit, VRAM peak, PSNR vs cùng `jobs_hash`.

### Giai đoạn C — Nặng hơn (chỉ nếu còn thời gian đề tài)

| Bước | Việc |
|------|------|
| C1 | PoC TensorRT/ONNX **một** UNet (không IP) → đo latency |
| C2 | Mở rộng gen UNet + IP nếu C1 có lợi ≥ ~20% |
| C3 | TinyVAE ablation riêng (không trộn vào mặc định RT) |

---

## 4. Metric bắt buộc mỗi lần thử (B2–B4)

So với cùng `jobs_hash` / `MAX_JOBS` nhỏ trước (smoke), rồi mới full:

| Metric | Mục tiêu so fp16+cache |
|--------|-------------------------|
| s/edit | Nhanh hơn hoặc ≈ (không chậm rõ) |
| VRAM peak | Thấp hơn rõ hoặc ≈ |
| PSNR vs `baseline_fp32` | Không tụt kiểu FP4 (~20 dB); ideally gần band fp16 (~45–50) |

---

## 5. Liên kết

- Context hỏi AI: [`PRECISION_QUANT_CONTEXT_FOR_STRONG_AI.md`](./PRECISION_QUANT_CONTEXT_FOR_STRONG_AI.md)
- Case precision: [`PRECISION_CASES.md`](./PRECISION_CASES.md)
- Bench cũ T4: [`quality_speed_bench_2026-06-17/report.md`](./quality_speed_bench_2026-06-17/report.md)
- Hướng RT tổng: [`../HUONG_PHAT_TRIEN.md`](../HUONG_PHAT_TRIEN.md)
