# Q&A — Câu hỏi & Giải đáp khái niệm

> Ghi lại các câu hỏi về khái niệm, thuật ngữ, kỹ thuật trong quá trình làm đề tài **SwiftEdit / CS2309.CH201**.  
> Skill Cursor `update-qa` **tự thêm entry** sau khi bạn hỏi khái niệm (hoặc nhắn *"thêm vào QA.md"*).

Tài liệu liên quan: [Overview](./SwiftEdit_Overview.md) · [Đề tài](./SwiftEdit_DeTai_CS2309.md) · [README](./README.md)

---

## Cách dùng

1. Hỏi trong chat Cursor (vd: *"Diffusion inversion là gì?"*).
2. Agent trả lời và **tự ghi vào QA.md** (skill `update-qa`), hoặc nhắn: *"Thêm câu hỏi này vào QA.md"*.
3. Entry mới đặt **ở đầu mục** tương ứng; mục lục cập nhật số câu hỏi tự động.

**Mẫu một entry:**

```markdown
### Q: [Câu hỏi ngắn gọn]?

**Ngày:** YYYY-MM-DD  
**Chủ đề:** #diffusion / #swiftedit / #aram / …

**Trả lời (tóm tắt):**
- …

**Ghi chú thêm / link:**
- …
```

---

## Mục lục nhanh

| # | Chủ đề | Số câu hỏi |
|---|---|---|
| 1 | [Diffusion & Text-to-Image](#1-diffusion--text-to-image) | 1 |
| 2 | [SwiftEdit & Pipeline](#2-swiftedit--pipeline) | 2 |
| 3 | [Inversion & Noise](#3-inversion--noise) | 0 |
| 4 | [Mask & ARaM](#4-mask--aram) | 0 |
| 5 | [Đánh giá & PieBench](#5-đánh-giá--piebench) | 0 |
| 6 | [Triển khai Mac / Colab](#6-triển-khai-mac--colab) | 0 |
| 7 | [Chỉnh sửa & Style](#7-chỉnh-sửa--style) | 0 |
| 8 | [Chưa phân loại](#8-chưa-phân-loại) | 1 |

*(Cập nhật cột "Số câu hỏi" khi thêm entry.)*

---

## 1. Diffusion & Text-to-Image

*Khái niệm nền: diffusion model, one-step vs multi-step, SBv2, CLIP, VAE, …*

<!-- qa:insert -->
### Q: Mỗi step trong diffusion chỉnh sửa ảnh cũ làm gì?

**Ngày:** 2026-06-01  
**Chủ đề:** #diffusion #one-step #inversion #pipeline

**Trả lời (tóm tắt):**
- Ảnh tưởng tượng: bức tranh bị phủ sương mù dày; mỗi step = lau bớt một lớp sương nhỏ.
- Mỗi step: mạng UNet nhìn ảnh hiện tại + prompt + "đang ở bước mấy", dự đoán nên bớt nhiễu thế nào → ảnh rõ hơn một chút.
- Sinh ảnh mới: bắt đầu từ nhiễu ngẫu nhiên, lặp ~50 step → ảnh khớp prompt.
- Inversion (giai đoạn 1): có ảnh thật, chạy ngược — mỗi step tìm "thêm nhiễu kiểu gì" để quay lại trạng thái trước, cuối cùng ra noise khởi đầu.
- Editing (giai đoạn 2): từ noise đó, lại lau sương ~50 step nhưng prompt mới + chỉnh attention để đổi nội dung (vd mèo cam→đen) mà giữ bố cục.
- 1 step = 1 lần UNet chạy (tốn GPU); 50+50 step ≈ 100 lần chạy → chậm.




---

## 2. SwiftEdit & Pipeline

*SwiftEdit là gì, input/output, so sánh P2P / TurboEdit, tại sao one-step nhanh, …*

<!-- qa:insert -->
### Q: Có ảnh minh họa pipeline inversion + editing diffusion không?

**Ngày:** 2026-06-01  
**Chủ đề:** #pipeline #inversion #swiftedit #diffusion

**Trả lời (tóm tắt):**
- Có — mục 1.3.3 SwiftEdit_DeTai_CS2309.md và §2.2 SwiftEdit_Overview.md.
- Ảnh lưu local assets/pipeline/ (hiển thị trong Cursor MD preview).
- Toàn pipeline: sage-fig3-pipeline.png (SAGE Fig. 3).
- Inversion: nulltext-diagram.png.
- Mỗi step: sage-fig4-denoise-steps.png; tutorial HF DDIM Inversion.
- Editing: p2p-cross-attention.png.
- Kết quả: p2p-teaser.png, nulltext-editing-results.png.


### Q: Tại sao các phương pháp diffusion chỉnh sửa ảnh cũ lại tốn nhiều step?

**Ngày:** 2026-06-01  
**Chủ đề:** #swiftedit #pipeline #diffusion #one-step #inversion

**Trả lời (tóm tắt):**
- Pipeline kinh điển gồm 2 giai đoạn độc lập, mỗi giai đoạn đều multi-step: inversion (20–50+) rồi sampling+edit (20–50+).
- Mỗi step = 1 forward pass UNet lớn trên latent → chi phí tính toán nhân đôi so với số step.
- Inversion (DDIM/Null-text): chạy ngược quá trình denoise để tìm noise ban đầu khớp ảnh nguồn; Null-text thêm optimize từng ảnh.
- Editing (P2P, MasaCtrl, PnP): chạy lại denoise đa bước, can thiệp cross/self-attention mỗi step để đổi prompt nhưng giữ layout.
- Các method này xây trên SD multi-step (50 step huấn luyện) — không thể rút còn 1 step mà không distill/học riêng như SBv2.
- Few-step (TurboEdit, ICD) giảm còn 3–8 step nhưng vẫn 2 giai đoạn; SwiftEdit: 1 step inversion + 1 step edit.




---

## 3. Inversion & Noise

*DDIM inversion, Null-text, one-step inversion `F_theta`, inverted noise `eps_hat`, stage 1/2 training, …*

<!-- qa:insert -->

*(Chưa có câu hỏi.)*

---

## 4. Mask & ARaM

*Self-guided mask, editing mask, `s_y` / `s_edit` / `s_non-edit`, IP-Adapter, cross-attention, …*

<!-- qa:insert -->

*(Chưa có câu hỏi.)*

---

## 5. Đánh giá & PieBench

*PSNR, MSE, CLIP-Whole, CLIP-Edited, IoU/Dice, 10 loại editing, …*

<!-- qa:insert -->

*(Chưa có câu hỏi.)*

---

## 6. Triển khai Mac / Colab

*MPS vs CUDA, cài env, weights, OOM, runtime, …*

<!-- qa:insert -->

*(Chưa có câu hỏi.)*

---

## 7. Chỉnh sửa & Style

*Semantic editing vs style transfer, prompt tiếng Anh, failure cases, …*

<!-- qa:insert -->

*(Chưa có câu hỏi.)*

---

## 8. Chưa phân loại

*Câu hỏi chưa xếp vào mục — sắp xếp lại sau.*

<!-- qa:insert -->
### Q: Tại sao không dùng LaTeX trong file MD?

**Ngày:** 2026-06-01  
**Chủ đề:** #markdown #swiftedit

**Trả lời (tóm tắt):**
- Nhiều MD reviewer (GitHub, LMS) không render LaTeX.
- Dùng backtick, code block và Unicode thay thế.
- Skill write-markdown quy định rule này cho repo CS2309.




---

## Tags gợi ý

Dùng trong entry để dễ tìm:

`#diffusion` `#one-step` `#swiftedit` `#inversion` `#noise` `#aram` `#mask` `#attention` `#ip-adapter` `#piebench` `#metrics` `#mac` `#colab` `#style-editing` `#prompt`

---

## Gợi ý câu hỏi có thể hỏi sau

- Diffusion model hoạt động thế nào? One-step khác multi-step ở đâu?
- Inversion là gì? Tại sao SwiftEdit không dùng DDIM Inversion?
- Inverted noise (`eps_hat`) dùng để làm gì khi edit?
- Self-guided mask được tính như thế nào?
- ARaM khác gì so với chỉnh global scale `s_x`?
- PSNR và CLIP đo cái gì? Trade-off giữa chúng?
- Tại sao paper chạy A100 còn mình dùng Mac + Colab?
- SwiftEdit có làm tốt style editing (watercolor, anime) không?

---

*File Q&A — bổ sung dần trong quá trình học và làm đề tài.*
