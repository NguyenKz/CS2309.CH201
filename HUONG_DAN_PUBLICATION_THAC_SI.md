# Hướng dẫn cân nhắc publication — đề tài SwiftEdit-RT & tốt nghiệp thạc sĩ UIT

> Tài liệu nội bộ ghi lại đánh giá trung thực về khả năng public paper từ công việc đề tài CS2309,
> phục vụ cân nhắc **điều kiện tốt nghiệp thạc sĩ UIT** (có bài báo → có thể giảm 10 tín chỉ).
>
> **Lưu ý:** Quy định chính thức về loại bài báo, hạng tạp chí/hội nghị, và mức miễn tín chỉ
> **phải xác nhận với Khoa / Phòng Đào tạo / GVHD** — file này chỉ là phân tích kỹ thuật và chiến lược,
> không thay quy chế UIT.

| | |
|---|---|
| **Đề tài** | SwiftEdit-RT — Realtime-Oriented Inference Acceleration |
| **Paper gốc** | [SwiftEdit (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_SwiftEdit_Lightning_Fast_Text-Guided_Image_Editing_via_One-Step_Diffusion_CVPR_2025_paper.pdf) |
| **Repo** | CS2309.CH201 |
| **Cập nhật** | 2026-06-15 |

---

## 1. Bối cảnh: tốt nghiệp thạc sĩ UIT và bài báo

Theo thông tin bạn cung cấp:

- Có **1 bài báo** (đúng loại/hạng theo quy định UIT) có thể **giảm 10 tín chỉ** trong chương trình thạc sĩ.
- Đề tài hiện tại là **mở rộng / tối ưu inference** trên SwiftEdit — không train lại model gốc.

**Việc cần làm ngay (hành chính):**

- [ ] Đọc quy chế tốt nghiệp thạc sĩ UIT (phiên bản áp dụng cho khóa của bạn).
- [ ] Hỏi GVHD / Khoa: bài **workshop**, **arXiv**, **tạp chí quốc gia**, **Scopus/WoS** — loại nào được tính?
- [ ] Hỏi rõ: bài **đã accept** hay chỉ **đã nộp**? Có cần **đồng tác giả GVHD** không?
- [ ] Hỏi deadline: nộp bài trước ngày nào để kịp xét tốt nghiệp khóa nào.

> Nhiều trường chỉ chấp nhận bài **đã được hội nghị/tạp chí chấp nhận (accepted)**, không tính arXiv hoặc bản nháp.
> Workshop CV thường được tính nếu nằm trong danh mục hội nghị của trường.

---

## 2. Đóng góp hiện có (tóm tắt thực chất)

Hướng chính: **SwiftEdit-RT** — tối ưu inference, không thay thuật toán editing gốc.

| Đóng góp | Mô tả ngắn | Đã đo / có bằng chứng | Mức “mới” khoa học |
|---|---|---|---|
| **EditCache** | Cache latent VAE + CLIP image embed + source prompt embed khi cùng ảnh, đổi nhiều edit prompt | Có: ~1.21× trên nền fp16 (cache-hit vs cache-miss); lossless, embed deterministic | Thấp — pattern engineering |
| **fp16 + channels_last** | Giảm precision UNet/text encoder; VAE giữ fp32 (tránh NaN/ảnh đen) | Có (T4): 1.50× vs fp32; PSNR 48.5 dB / SSIM 0.998 / LPIPS 0.0008 (600 ảnh) | Thấp — best practice |
| **VRAM** | Cho phép chạy trên T4 15GB | Có: 14.6 GB → 8.5 GB (−42.1%) | Thấp nhưng có giá trị thực tế |
| **StageTimer** | Profile latency từng module (`vae_encode`, `unet_inverse`, `gen_unet`, …) | Có (Mac M4 subset 20) | Thấp — instrumentation |
| **Vectorized mask** | Bỏ `.cpu().apply_()` trên self-guided mask | Có: 12.2 ms → 4.6 ms; mask giống hệt baseline | Rất thấp (ảnh hưởng end-to-end nhỏ) |
| **Bỏ decode noise_image** | Không decode ảnh nhiễu khi caller không dùng | Có patch | Rất thấp |
| **Benchmark quy mô** | 200 ảnh × 3 prompt trên T4; RUN_ID, CSV, run_meta, zip | Có: `experimental_data/quality_speed_bench_2026-06-14/` | Trung bình (nếu công bố đầy đủ) |
| **Kiểm chứng baseline fp32** | So diff với SwiftEdit upstream Qualcomm | Có: `fp32_baseline_verification.md`, `upstream_diff.md` | Tăng độ tin cậy, không phải novelty |
| **user_mask** | Mask người dùng khoanh vùng ghi đè self-guided mask | Có demo + giới hạn (vật lớn còn sót) | Thấp — extension nhỏ |
| **Gradio demo** | UI fp16 + cache + object removal | Có: `scripts/app_gradio.py` | Ứng dụng, không phải method mới |
| **PIE-Bench scripts** | `piebench_metrics.py`, `run_piebench_eval.py` | Có smoke + subset 20 (Mac) | Nền tảng, chưa đủ quy mô paper |
| **fp8 / fp4 quant** | Weight-only quant UNet (torchao / bitsandbytes) | **Đã đo T4 (2026-06-17):** fp8 1.92× nhưng PSNR 6 dB; fp4 VRAM −48.5%, PSNR 21.7 dB | Thấp — ablation; fp8 **fail** chất lượng |

**Kết luận một dòng:** Bạn có **bài systems / engineering study** khá chỉn (đo, tách lớp, tái lập), **không có thuật toán editing mới** so với SwiftEdit CVPR 2025.

---

## 3. Số liệu chính đã có (T4)

Nguồn mới nhất: [`experimental_data/quality_speed_bench_2026-06-17/report.md`](./experimental_data/quality_speed_bench_2026-06-17/report.md) — RUN_ID `20260617-0336-bb4785`

### 3.1. Tốc độ & VRAM (200×3, 4 config)

| Config | Mean (s) | Speedup | VRAM max | Giảm VRAM |
|---|---:|---:|---:|---:|
| fp32 | 2.91 | 1.00× | 14596 MB | — |
| fp16 + cache | 1.71 | 1.70× | 8446 MB | 42.1% |
| fp8 + cache | 1.52 | 1.92× | 7819 MB | 46.4% |
| fp4 + cache | 1.74 | 1.68× | 7515 MB | 48.5% |

### 3.2. Chất lượng vs fp32 (ground truth)

| Config | PSNR | SSIM | LPIPS | Đánh giá |
|---|---:|---:|---:|---|
| fp16 + cache | 48.6 dB | 0.998 | 0.0008 | Gần trùng fp32 |
| fp8 + cache | 6.0 dB | 0.020 | 0.968 | **Hỏng — không dùng được** |
| fp4 + cache | 21.7 dB | 0.776 | 0.150 | Giảm mạnh |

**Kết luận đã đo (2026-06-17):** **fp16 + cache** là khuyến nghị. fp8 nhanh hơn wall-clock nhưng vô dụng (PSNR 6 dB). fp4 chỉ khi bắt buộc tiết kiệm VRAM.

*(Bản cũ chỉ fp32 vs fp16: `quality_speed_bench_2026-06-14` — fp32 2.54s, kết quả fp16 tương đương.)*

**Hai câu hỏi chất lượng khác nhau (quan trọng khi viết paper):**

| Câu hỏi | Benchmark hiện tại trả lời | Còn thiếu cho paper |
|---|---|---|
| Tối ưu có làm hỏng output không? | **Có** — PSNR fp16 vs fp32 trên 600 ảnh | — |
| SwiftEdit-RT còn giữ chất lượng editing theo PIE-Bench không? | **Chưa đủ** — chỉ subset 20 trên Mac (CLIP) | Cần PIE-Bench 700 đầy đủ theo protocol gốc |

---

## 4. Đủ / chưa đủ theo từng mục tiêu publication

| Mục tiêu | Đủ chưa? | Ghi chú |
|---|---|---|
| Báo cáo đề tài CS2309 / luận văn thạc sĩ (không cần paper) | **Đủ** | Có thể viết luận văn từ tài liệu + số liệu hiện có |
| arXiv / technical report | **Đủ** | Framing: *systematic inference study*, không claim method mới |
| Workshop hội nghị CV/AI (~4 trang) | **Gần đủ** | Cần thêm ~2–4 tuần: PIE-Bench full, packaging, viết paper |
| Tạp chí / hội nghị quốc gia (nếu UIT chấp nhận) | **Có thể** | Tùy yêu cầu hạng; thường dễ hơn main track quốc tế |
| Main track CVPR / ICCV / ECCV | **Chưa đủ** | Thiếu novelty thuật toán + so SOTA đầy đủ |

### 4.1. Vì sao chưa đủ main track?

Reviewer sẽ hỏi:

1. **Novelty so với SwiftEdit?** — Chưa có thuật toán inversion/editing mới.
2. **fp16 + cache có gì surprising?** — Đã là practice phổ biến trong diffusion inference.
3. **1.7× trên T4 có ý nghĩa gì?** — SwiftEdit paper đã báo ~0.23s trên A100; cần framing rõ (GPU phổ thông, VRAM, interactive multi-prompt).
4. **So với TurboEdit / inpainting SOTA?** — Chưa có baseline ngoài.

### 4.2. Vì sao vẫn có cửa workshop / paper “hệ thống”?

- Bạn **không chỉ chạy lại** — có **đo từng lớp**, **tách đóng góp fp16 vs cache**, **benchmark tái lập** (RUN_ID, CSV, verification upstream).
- Story **interactive editing**: cùng ảnh, nhiều prompt — cache có lợi ích rõ, lossless.
- **VRAM −42%** giải quyết pain point thật (fp32 sát OOM trên T4 15GB).
- Có thể **open-source** gói tối ưu như contribution cộng đồng.

---

## 5. fp8 / fp4 — kết luận trung thực (cho báo cáo & paper)

Đã chuẩn bị trong notebook (`improved_fp8_cache`, `improved_fp4_cache`). Trên **Tesla T4 (Turing)**:

| Precision | Chạy được? | Tăng tốc trên T4? | Giảm VRAM? | Ảnh hưởng chất lượng |
|---|---|---|---|---|
| fp32 | Có (native) | Mốc 1.00× | Mốc | Ground truth |
| fp16 | Có (native) | **Có** (~1.5×) | Có (~42%) | Gần như không đổi (đã đo) |
| fp8 (weight-only, torchao) | Có thể; kernel fp8 hạn chế trên T4 | **Không** / có thể chậm hơn | Có | Nhỏ (cần đo) |
| fp4 (weight-only, bitsandbytes) | Có | **Không** / thường chậm hơn | **Nhiều nhất** | Lớn hơn fp8 (cần đo) |

**Cách viết trung thực trong paper:**

> Trên GPU Turing (T4), quantization fp8/fp4 **không mang lại tăng tốc phần cứng** như fp16;
> lợi ích chính là **giảm footprint bộ nhớ**. **fp16 + cache** là cấu hình cân bằng tốt nhất
> giữa tốc độ, VRAM và chất lượng trên phần cứng phổ thông.

**Khuyến nghị precision “tốt nhất” (đã đo 2026-06-17):**

| Tiêu chí | Thắng | Ghi chú |
|---|---|---|
| Nhanh nhất (wall-clock) | fp8 (1.92×) | **Vô dụng** — PSNR 6 dB |
| Tiết kiệm VRAM | fp4 (−48.5%) | PSNR 21.7 dB |
| Chất lượng | fp32 | Chậm + tốn VRAM |
| **Cân bằng tổng thể** | **fp16 + cache** | 1.70×, PSNR 48.6 dB, VRAM −42.1% |

---

## 6. Ba hướng paper khả thi (chọn 1)

### Hướng A — Workshop / systems (khớp nhất với việc đang làm) ⭐ Khuyến nghị

**Định vị:** *SwiftEdit-RT: A Reproducible Inference Acceleration Stack for One-Step Image Editing*

**Điểm mạnh:** Dùng gần hết công việc hiện có; timeline ngắn nhất; phù hợp điều kiện “có 1 bài” nếu UIT chấp nhận workshop.

**Cần bổ sung:**

- [x] Chạy fp8/fp4 trên Colab T4 — fp8 PSNR 6 dB (fail); fp4 PSNR 21.7 dB
- [ ] PIE-Bench **700 mẫu** đầy đủ: PSNR/MSE background, CLIP-Whole, CLIP-Edited, runtime (protocol PIE-Bench).
- [ ] Bảng ablation **từng lớp**: mask / cache / fp16 / (compile) / fp8 / fp4.
- [ ] So **upstream SwiftEdit** vs **SwiftEdit-RT** trên cùng GPU.
- [ ] Viết paper ~4 trang + figure (speedup, VRAM, quality, pipeline diagram).
- [ ] Nộp workshop (xem mục 8 — deadline).

**Venue gợi ý (cần đối chiếu danh mục UIT):**

- Workshop tại CVPR / ICCV / ECCV (Efficient Deep Learning, Real-time AI, On-device ML, …)
- Hội nghị quốc gia Công nghệ thông tin / Trí tuệ nhân tạo (nếu UIT chấp nhận và deadline sớm hơn)

---

### Hướng B — Application: object removal

**Định vị:** *Can One-Step Editors Replace Inpainting? A Study of SwiftEdit with User-Guided Masks*

**Cần thêm nhiều hơn:**

- [ ] Bộ ảnh removal có GT (COCO / PIE-Bench).
- [ ] So **LaMa** hoặc SD inpainting.
- [ ] Metric: confidence detector drop, LPIPS ngoài mask, human eval nhỏ.

**Khả năng:** workshop hoặc tạp chí ứng dụng — **effort cao hơn Hướng A**.

---

### Hướng C — Analysis: giới hạn SwiftEdit

**Định vị:** *When Does One-Step Editing Break? Global Style, Mask Quality, and Object Scale*

**Cần:** protocol đánh giá mới + nhiều case study (global weather, mask IoU, object size).

**Khả năng:** workshop phân tích — **effort trung bình–cao**, ít tái dùng số liệu tốc độ hiện có.

---

## 7. Checklist tối thiểu cho 1 bài workshop (Hướng A)

### 7.1. Thí nghiệm

- [x] Benchmark fp32 vs fp16+cache (200×3, T4)
- [x] Tách đóng góp fp16 vs cache
- [x] VRAM per-edit
- [x] Chất lượng fp16 vs fp32 (PSNR/SSIM/LPIPS)
- [x] fp8 / fp4 ablation (T4) — xem `quality_speed_bench_2026-06-17/`
- [ ] PIE-Bench 700 (metrics protocol gốc, không chỉ vs fp32)
- [ ] `torch.compile` ablation (optional nhưng tăng điểm)
- [ ] Latency breakdown (StageTimer) trên T4 — hiện chủ yếu có trên Mac

### 7.2. Nội dung paper

- [ ] Title + abstract (nhấn *system / reproducible / trade-off*, không claim SOTA editing)
- [ ] Introduction: motivation (realtime, T4 VRAM, multi-prompt demo)
- [ ] Related work: SwiftEdit, inference opt diffusion, caching
- [ ] Method: SwiftEdit-RT stack (sơ đồ 3 tầng: fp16 → cache → optional quant)
- [ ] Experiments: bảng speed, VRAM, quality; ablation
- [ ] Limitations: T4 vs A100; cache scope; fp8/fp4 không speedup trên Turing
- [ ] Conclusion + open-source link

### 7.3. Hành chính UIT

- [ ] Xác nhận workshop X có trong danh mục được tính miễn tín chỉ
- [ ] Thống nhất tác giả với GVHD (thứ tự, đồng tác giả)
- [ ] Giữ email / letter of acceptance làm hồ sơ tốt nghiệp

---

## 8. Outline paper 4 trang (Hướng A — draft cấu trúc)

### Title (gợi ý)

**SwiftEdit-RT: Layered Inference Acceleration for One-Step Text-Guided Image Editing**

### Abstract (ý chính, ~150 từ)

- SwiftEdit nhanh nhưng vẫn có overhead trên GPU phổ thông (T4).
- Đề xuất SwiftEdit-RT: fp16, channels_last, lossless EditCache, optional weight-only quant.
- Đo trên 200×3 PIE-Bench subset: 1.7× speedup, −42% VRAM, PSNR 48.5 dB vs fp32.
- Phân tích tầng: fp16 1.5×, cache thêm 1.21× trên fp16.
- Kết luận: fp16+cache là sweet spot trên T4; fp8/fp4 chủ yếu cho bộ nhớ.

### Section 1 — Introduction

- One-step editing vs multi-step latency.
- Gap: official repo chưa tối ưu cho Colab T4 / interactive demo.
- Contributions (bullet 3–4 ý, trung thực).

### Section 2 — Background

- SwiftEdit pipeline ngắn (inversion → mask → ARaM).
- PIE-Bench metrics.

### Section 3 — SwiftEdit-RT

- Sơ đồ 3 tầng tối ưu (không độc lập).
- EditCache semantics (invalidate khi đổi ảnh / source prompt).
- fp16 policy (VAE fp32).
- Optional fp8/fp4 (weight-only, sau load weights).

### Section 4 — Experiments

- Setup: T4, 200 images, 3 prompts, RUN_ID reproducibility.
- Table 1: speed + VRAM.
- Table 2: quality vs fp32.
- Table 3: ablation layers.
- Table 4 (nếu kịp): PIE-Bench 700 vs paper reference.
- Figure: comparison grid + bar chart speedup.

### Section 5 — Conclusion

- Khuyến nghị fp16+cache cho deployment.
- Future: Ada/Hopper fp8 speedup, torch.compile, upstream PR.

---

## 9. Timeline gợi ý (nếu nhắm workshop)

Giả định bắt đầu từ **2026-06-15**:

| Tuần | Việc |
|---|---|
| 1 | Chạy fp8/fp4 Colab; hoàn thiện ablation table |
| 2 | PIE-Bench 700 trên Colab (batch qua đêm) |
| 3 | Viết draft paper + figure |
| 4 | GVHD review; chỉnh sửa; nộp workshop |

**Rủi ro:** deadline workshop CVPR/ICCV thường **trước** hội nghị chính 2–3 tháng — cần tra **ngay** deadline venue cụ thể. Nếu trễ mùa quốc tế, cân nhắc **hội nghị quốc gia** sớm hơn (nếu UIT chấp nhận).

---

## 10. So sánh effort vs lợi ích tốt nghiệp

| Lựa chọn | Effort | Lợi ích học thuật | Phù hợp miễn 10 TC UIT |
|---|---|---|---|
| Chỉ luận văn, không paper | Thấp | Đủ tốt nghiệp nếu không cần miễn TC | Không |
| arXiv + luận văn | Thấp–TB | Có portfolio | **Tùy quy định UIT** (nhiều trường không tính) |
| Workshop quốc tế | TB | Tốt; có bằng accept | **Khả năng cao** nếu UIT liệt kê venue |
| Hội nghị/tạp chí quốc gia | TB | Đủ nếu UIT ưu tiên quốc gia | **Khả năng cao** — kiểm tra danh mục |
| Main track CVPR | Cao | Rất khó trong timeline thạc sĩ | Không khuyến nghị |

---

## 11. Kết luận chiến lược (tóm tắt)

1. **Công việc hiện tại đủ cho luận văn thạc sĩ** và có thể đủ cho **1 bài workshop / hệ thống** nếu bổ sung PIE-Bench full + viết paper đúng framing.
2. **Không nên nhắm main track CVPR** với scope hiện tại — thiếu novelty thuật toán.
3. **Hướng A (SwiftEdit-RT workshop)** là lộ trình **ngắn nhất** tới “có bài báo” phục vụ miễn 10 tín chỉ — **sau khi xác nhận quy định UIT**.
4. Luôn **báo cáo trung thực** fp8/fp4 trên T4 — đó vẫn là contribution (negative result có giá trị).
5. **Ưu tiên hành chính:** hỏi GVHD/Khoa trước khi đầu tư viết paper dài.

---

## 12. Tài liệu liên quan trong repo

| File | Nội dung |
|---|---|
| [`HUONG_PHAT_TRIEN.md`](./HUONG_PHAT_TRIEN.md) | Hướng phát triển + kết quả tối ưu đã đo |
| [`NHAT_KY.md`](./NHAT_KY.md) | Nhật ký tiến độ |
| [`SwiftEdit_DeTai_CS2309.md`](./SwiftEdit_DeTai_CS2309.md) | Đề tài chi tiết, RQ, đóng góp C1–C15 |
| [`experimental_data/quality_speed_bench_2026-06-17/report.md`](./experimental_data/quality_speed_bench_2026-06-17/report.md) | Benchmark T4 mới nhất (fp32/fp16/fp8/fp4) |
| [`notebooks/CS2309_SwiftEdit_quality_speed_bench.ipynb`](./notebooks/CS2309_SwiftEdit_quality_speed_bench.ipynb) | Notebook tái lập (+ fp8/fp4) |
| [`experimental_data/quality_speed_bench_2026-06-14/fp32_baseline_verification.md`](./experimental_data/quality_speed_bench_2026-06-14/fp32_baseline_verification.md) | Kiểm chứng baseline |

---

## 13. Ghi chú phiên bản

| Ngày | Thay đổi |
|---|---|
| 2026-06-17 | Cập nhật số liệu fp8/fp4 từ Colab T4 (RUN_ID 20260617-0336-bb4785) |
| 2026-06-15 | Tạo file — tổng hợp đánh giá publication, fp8/fp4, hướng UIT thạc sĩ |
