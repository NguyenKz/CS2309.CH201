# Lý thuyết — Danh sách task (Theory Tasks)

> Giai đoạn 1 — CS2309 SwiftEdit. Làm tuần tự từ Task 1 → Task 6.  
> Checklist tổng: [`README.md` § Giai đoạn 1](../README.md#giai-đoạn-1--lý-thuyết-tuần-12)

| Task | Tên | Trạng thái |
|------|-----|------------|
| **1** | [Đọc paper SwiftEdit](#task-1--đọc-paper-swiftedit) | 🔄 Đang làm |
| 2 | Nền tảng: SwiftBrushv2, DDIM, IP-Adapter | ⬜ Chưa |
| 3 | Ôn PieBench | ✅ Xong (ôn lại) |
| 4 | Pipeline + map code | ⬜ Chưa |
| 5 | Bảng Related Work | ⬜ Chưa |
| 6 | Viết báo cáo (draft lý thuyết) | ⬜ Chưa |

---

## Task 1 — Đọc paper SwiftEdit

| | |
|---|---|
| **Mục tiêu** | Nắm đủ nội dung cốt lõi paper để viết Overview, giải thích pipeline và hiểu thí nghiệm Table 1 |
| **Paper** | [SwiftEdit — CVPR 2025 (PDF)](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_SwiftEdit_Lightning_Fast_Text-Guided_Image_Editing_via_One-Step_Diffusion_CVPR_2025_paper.pdf) |
| **Đọc** | Abstract · Introduction · **Sec. 3** · **Sec. 4** · **Sec. 5** (bỏ qua Related Work chi tiết lúc này — để Task 5) |
| **Thời gian ước tính** | 2–4 giờ (chia 4 phiên: 1.1 → 1.2 → 1.3 → 1.4) |
| **Ghi chú của bạn** | Điền vào các mục *Ghi chú* bên dưới hoặc file riêng `tasks/notes/task1_paper_notes.md` |

### Tổng quan Task 1

Paper SwiftEdit giải quyết **chỉnh sửa ảnh theo văn bản** (text-guided image editing) với hai đóng góp chính:

1. **One-step Inversion** — mạng `F_theta` ánh xạ ảnh → inverted noise trong **một forward pass** (thay DDIM/Null-text đa bước).
2. **ARaM** (Attention Rescaling for Mask-aware Editing) — chỉnh cross-attention theo mask để sửa vùng cục bộ, giữ nền.

Khi đọc xong Task 1, bạn phải trả lời được:

- SwiftEdit khác pipeline cũ (P2P, Null-text) ở chỗ nào về **số bước** và **thời gian**?
- Inverted noise là gì, tính bằng gì?
- Self-guided mask sinh ra thế nào, không cần người vẽ?
- Table 1 paper báo cáo metric gì, SwiftEdit đứng ở đâu so TurboEdit / multi-step?

---

### Sub-task 1.1 — Abstract + Introduction

**Đọc:** Trang đầu paper — Abstract, phần 1. Introduction (và skim Related Work nếu có thời gian).

**Tập trung vào:**

| Câu hỏi | Gợi ý tìm trong paper |
|---------|------------------------|
| Bài toán đầu vào/đầu ra là gì? | Ảnh nguồn + source/edit prompt → ảnh đã sửa |
| Vì sao multi-step chậm? | Inversion 20–50+ step + editing 20–50+ step |
| Hai đóng góp chính của paper? | One-step inversion + ARaM |
| Backbone là gì? | SwiftBrushv2 (SBv2) — one-step text-to-image |
| Tốc độ paper báo cáo? | ~0.23s/ảnh trên A100; nhanh hơn ≥50× so multi-step |
| Dataset đánh giá chính? | PieBench (700 mẫu, 10 loại editing) |

**So sánh nhanh (điền sau khi đọc):**

| Phương pháp | Inversion | Editing | Runtime ~ (paper) |
|-------------|-----------|---------|-------------------|
| DDIM + P2P | nhiều step | nhiều step | ~26s |
| Null-text + P2P | nhiều step + optimize | nhiều step | ~134s |
| TurboEdit | few-step | few-step | ~1.3s |
| **SwiftEdit** | **1 step** | **1 step** | **~0.23s** |

**Deliverable 1.1:** 5–10 bullet ghi chú (tiếng Việt), ví dụ:

```
- SwiftEdit: one-step inversion + one-step editing trên SBv2
- Mục tiêu: instant editing, phù hợp realtime/on-device
- So P2P: giảm từ ~50+50 step xuống 1+1 step
- ...
```

**Ghi chú của tôi (1.1):**

```
(viết tại đây)
```

- [ ] 1.1 xong

---

### Sub-task 1.2 — Sec. 3: One-step Inversion Framework

**Đọc:** Section 3 — One-step Inversion (toàn bộ, kể cả training Stage 1 và Stage 2).

**Khái niệm bắt buộc:**

| Thuật ngữ | Ý nghĩa (tự điền sau khi đọc) |
|-----------|-------------------------------|
| Latent `z` | Ảnh nguồn sau VAE encode |
| Text condition `c_y` | Embedding của prompt (CLIP text encoder) |
| Image condition `c_x` | Điều kiện ảnh (nhánh IP-Adapter) |
| Inverted noise `eps_hat` | Noise “đảo ngược” từ ảnh, dùng làm điểm bắt đầu chỉnh sửa |
| `F_theta` | Mạng inversion — học ánh xạ `(z, c_y) → eps_hat` |

**Công thức cốt lõi (paper, dạng text):**

```
eps_hat = F_theta(z, c_y)
z_hat   = G_IP(eps_hat, c_y, c_x)
```

- `G_IP`: generator có IP-Adapter — sinh latent/ảnh từ inverted noise + prompt + ảnh điều kiện.

**Hai giai đoạn huấn luyện — điền bảng:**

| Stage | Dữ liệu | Loss / mục tiêu | Kết quả sau stage |
|-------|---------|-----------------|-------------------|
| Stage 1 | | | |
| Stage 2 | | | |

**Câu hỏi then chốt (phải trả lời bằng lời của bạn):**

1. **Tại sao DDIM inversion không phù hợp one-step model?**  
   *(Gợi ý: DDIM cần nhiều timestep; SBv2 chỉ có 1 bước denoise — không có quỹ đạo đa bước để đảo ngược.)*

2. Null-text Inversion tốn thời gian vì sao? (so với `F_theta`)

3. Stage 1 vs Stage 2 khác nhau thế nào? (synthetic vs real image)

**Deliverable 1.2:** Một đoạn 5–8 câu (tiếng Việt) giải thích *tại sao SwiftEdit cần học `F_theta` thay vì dùng DDIM*.

**Ghi chú của tôi (1.2):**

```
(viết tại đây)
```

- [ ] 1.2 xong

---

### Sub-task 1.3 — Sec. 4: ARaM + Self-guided Mask

**Đọc:** Section 4 — mask-aware editing, ARaM, self-guided mask.

**Pipeline logic (điền / xác nhận):**

```
Ảnh nguồn
  → VAE encode → z
  → F_theta(z, c_source)  → eps_hat_source
  → F_theta(z, c_edit)    → eps_hat_edit
  → M = normalize(|eps_hat_source - eps_hat_edit|)   ← self-guided mask
  → G_IP + ARaM(M, s_y, s_edit, s_non-edit) → ảnh edited
```

**Ba hệ số ARaM:**

| Hệ số | Tên trong code (gần đúng) | Tác dụng (điền sau khi đọc) |
|-------|---------------------------|------------------------------|
| `s_y` | scale text / prompt | |
| `s_edit` | `scale_edit` | Cường độ chỉnh trong vùng mask |
| `s_non-edit` | `scale_non_edit` | Giữ vùng nền (background) |

**Câu hỏi then chốt:**

1. Self-guided mask **không cần người vẽ** — vùng nào được coi là “cần sửa”?
2. ARaM tác động lên **thành phần nào** của model? (cross-attention)
3. Người dùng **có thể thay mask tự sinh** bằng mask ngoài không? (có — liên hệ `user_mask` trong đề tài)

**Liên hệ code repo (đọc sau paper, 10 phút):**

| Bước paper | File trong `SwiftEdit/` |
|------------|-------------------------|
| Pipeline chính | `infer.py` — `edit_image()` |
| Inversion + generation | `models.py` |
| ARaM / mask attention | `src/mask_ip_controller.py`, `src/mask_attention_processor.py` |

**Deliverable 1.3:** Bullet list 4–6 ý: mask sinh ra thế nào + ARaM làm gì + tên file code tương ứng.

**Ghi chú của tôi (1.3):**

```
(viết tại đây)
```

- [ ] 1.3 xong

---

### Sub-task 1.4 — Sec. 5: Experiments

**Đọc:** Section 5 — setup thí nghiệm, **Table 1** (PieBench), ablation (nếu có), hình minh họa qualitative.

**Table 1 — chép hoặc tóm tắt (điền số từ paper):**

| Method | PSNR ↑ | MSE ×10⁴ ↓ | CLIP-Whole ↑ | CLIP-Edited ↑ | Time (s) ↓ |
|--------|--------|------------|--------------|---------------|------------|
| DDIM + P2P | | | | | |
| NT-Inv + P2P | | | | | |
| TurboEdit | | | | | |
| ICD (SD 1.5) | | | | | |
| **SwiftEdit** | | | | | |

*(Số tham khảo từ Overview repo nếu chưa kịp chép: SwiftEdit PSNR 23.33, MSE 6.60, CLIP-Whole 25.16, CLIP-Edited 21.25, Time 0.23)*

**Câu hỏi then chốt:**

1. SwiftEdit **nhanh nhất** trong bảng — trade-off chất lượng thế nào so TurboEdit và NT-Inv+P2P?
2. Metric nào đo **giữ nền**? (PSNR, MSE trên vùng không edit)
3. Metric nào đo **đúng prompt**? (CLIP-Whole, CLIP-Edited với edit prompt)
4. Paper chạy trên GPU gì? (A100) — khác với Mac M4 / Colab T4 của đề tài

**Liên hệ đề tài CS2309 (đã làm thực nghiệm):**

| | Paper (A100) | Đề tài của bạn |
|---|--------------|----------------|
| Runtime | ~0.23s | Mac ~69s (20 mẫu fp32); Colab ~1.7–2.9s (fp16/fp32) |
| PieBench | 700 mẫu | Subset 20 mẫu Mac + benchmark 2400 edit Colab |
| CLIP-Whole | 25.16 | Mac subset TB **23.02** |

**Deliverable 1.4:** Table 1 đã điền đủ + 3 câu nhận xét: (1) tốc độ (2) CLIP (3) PSNR/background.

**Ghi chú của tôi (1.4):**

```
(viết tại đây)
```

- [ ] 1.4 xong

---

### Hoàn thành Task 1

Khi cả 4 sub-task đều `[x]`:

- [ ] Tổng hợp ghi chú 1.1–1.4 thành **1 trang tóm tắt** (có thể paste vào báo cáo §Overview)
- [ ] Báo agent: *"xong Task 1"* → đánh `[x]` README + ghi `NHAT_KY.md`
- [ ] Chuyển sang [Task 2 — Nền tảng kỹ thuật](#) *(sẽ bổ sung trong file này)*

**Tài liệu đọc kèm (không thay paper):**

- [`SwiftEdit_Overview.md` §1–3](../SwiftEdit_Overview.md) — bản tiếng Việt tóm tắt
- [`SwiftEdit_DeTai_CS2309.md` §1](../SwiftEdit_DeTai_CS2309.md#1-overview) — pipeline ASCII
- [`QA.md` §2, §4](../QA.md) — self-guided mask, source prompt

---

*Cập nhật: 2026-06-18 — Task 1 viết đầy đủ; Task 2–6 sẽ bổ sung sau.*
