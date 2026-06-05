# SwiftEdit — Overview

> Tài liệu tổng quan về phương pháp **SwiftEdit** (CVPR 2025)  
> Dùng cho đề tài CS2309.CH201 — *Chỉnh sửa và thay đổi phong cách ảnh*

| | |
|---|---|
| **Paper** | [SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_SwiftEdit_Lightning_Fast_Text-Guided_Image_Editing_via_One-Step_Diffusion_CVPR_2025_paper.pdf) |
| **Authors** | Trong-Tung Nguyen, Quang Nguyen, Khoi Nguyen, Anh Tran, Cuong Pham (Qualcomm AI Research) |
| **Code** | [github.com/Qualcomm-AI-research/SwiftEdit](https://github.com/Qualcomm-AI-research/SwiftEdit) |
| **Project page** | [swift-edit.github.io](https://swift-edit.github.io/) |
| **Venue** | CVPR 2025 |

Tài liệu liên quan trong repo: [README](./README.md) · [Đề tài chi tiết](./SwiftEdit_DeTai_CS2309.md) · [Nhật ký](./NHAT_KY.md)

---

## 1. SwiftEdit là gì?

**SwiftEdit** là công cụ **chỉnh sửa ảnh theo văn bản** (text-guided image editing): người dùng đưa vào một ảnh gốc và mô tả thay đổi mong muốn bằng prompt tiếng Anh, hệ thống trả về ảnh đã chỉnh sửa.

Ví dụ:

| Ảnh nguồn (mô tả) | Edit prompt | Kết quả mong đợi |
|---|---|---|
| Chú mèo **cam** trên hàng rào | *"A black cat sitting on top of a fence."* | Mèo **đen**, background giữ nguyên |
| Giỏ **táo** | *"A basket of puppies"* | Giỏ **cún con**, phần còn lại ổn định |
| Miệng **đóng** | *"mouth opened"* | Miệng mở, khuôn mặt còn lại giữ |

Điểm nổi bật: SwiftEdit thực hiện toàn bộ quy trình trong **một bước diffusion** (one-step), đạt **~0.23 giây/ảnh** trên NVIDIA A100 — nhanh hơn ít nhất **50×** so với các phương pháp multi-step phổ biến, trong khi vẫn giữ chất lượng chỉnh sửa cạnh tranh.

---

## 2. Bài toán và động lực

### 2.1. Bài toán

Cho:

- Ảnh nguồn `x_source`
- Source prompt `y_source` *(khuyến nghị)*
- Edit prompt `y_edit`

Sinh ảnh `x_edit` sao cho:

1. **Đúng semantics** — nội dung khớp edit prompt `y_edit`
2. **Bảo toàn background** — vùng không liên quan giữ nguyên
3. **Nhanh** — đủ tốc độ cho ứng dụng tương tác / on-device

### 2.2. Hạn chế của phương pháp cũ

Các pipeline diffusion chỉnh sửa ảnh kinh điển (Prompt-to-Prompt, Null-text Inversion, MasaCtrl, Plug-and-Play, …) thường gồm **hai giai đoạn nặng**:

```
Inversion đa bước (20–50+ steps)  →  Sampling + attention edit đa bước
         ↑                                      ↑
   DDIM / Null-text …                    P2P, MasaCtrl, PnP …
```

→ Thời gian **12–130+ giây**/ảnh, khó dùng real-time.

Một số hướng **few-step** (TurboEdit, ICD, ReNoise) rút xuống 3–8 bước nhưng vẫn chậm hơn nhiều so với one-step.

**SwiftEdit** đặt mục tiêu: *instant editing* — inversion **1 bước** + editing **1 bước**.

**Ảnh minh họa pipeline cũ (multi-step)** — chi tiết + nguồn: [Đề tài §1.3.3](./SwiftEdit_DeTai_CS2309.md#133-ảnh-minh-họa-pipeline-cũ-multi-step--tham-khảo). Pipeline SwiftEdit (one-step): [§1.2](./SwiftEdit_DeTai_CS2309.md#12-kiến-trúc-và-pipeline-suy-luận).

![Pipeline inversion → sampling (SAGE Fig. 3)](./assets/pipeline/sage-fig3-pipeline.png)

*Sơ đồ tham khảo: ảnh gốc → DDIM inversion → noise/latent → DDIM sampling + edit prompt → ảnh sửa (SAGE, 2025).*

Giải thích dễ hiểu *mỗi step làm gì* và ảnh từng giai đoạn: [Đề tài §1.3.1–1.3.3](./SwiftEdit_DeTai_CS2309.md#131-mỗi-step-làm-gì-giải-thích-dễ-hiểu).

---

## 3. Ý tưởng cốt lõi

SwiftEdit dựa trên mô hình sinh ảnh one-step **SwiftBrushv2 (SBv2)** và có **hai đóng góp chính**:

### 3.1. One-step Inversion Framework

Thay vì DDIM Inversion (không phù hợp one-step), SwiftEdit học mạng **encoder-based** `F_theta` ánh xạ:

```
eps_hat = F_theta(z, c_y)
z_hat   = G_IP(eps_hat, c_y, c_x)
```

- `z`: latent ảnh nguồn (VAE encode)
- `c_y`: embedding text prompt (CLIP)
- `c_x`: điều kiện ảnh (IP-Adapter branch)
- `eps_hat`: **inverted noise** — điểm khởi đầu để chỉnh sửa

Huấn luyện **2 giai đoạn**:

| Stage | Dữ liệu | Mục tiêu |
|---|---|---|
| **Stage 1** | Ảnh synthetic từ SBv2 + cặp `(eps, z)` | Regression noise + reconstruction |
| **Stage 2** | Ảnh thực (CommonCanvas) | Perceptual loss (DISTS) + regularization giữ phân phối noise |

→ Sau training, invert **bất kỳ ảnh nào** trong **1 forward pass**, không cần optimization từng ảnh như Null-text Inversion.

### 3.2. Attention Rescaling for Mask-aware Editing (ARaM)

Để chỉnh sửa **cục bộ** và giữ background:

1. **Self-guided mask** `M`: so sánh inverted noise khi conditioning source prompt vs edit prompt

   ```
   M = normalize(abs(eps_hat_source - eps_hat_edit))
   ```

2. **ARaM**: tách cross-attention theo vùng mask với 3 hệ số:
   - `s_y` — cường độ alignment edit prompt trong vùng sửa
   - `s_edit` — ảnh hưởng image condition trong vùng edit
   - `s_non-edit` — giữ vùng nền

Người dùng có thể **tự cung cấp mask** hoặc dùng mask tự sinh.

---

## 4. Pipeline suy luận

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT                                                          │
│  • Source image x_source                                        │
│  • Source prompt y_source  (optional, khuyến nghị)              │
│  • Edit prompt y_edit                                           │
│  • Editing mask M (optional — tự sinh nếu bỏ qua)               │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
              z = VAE_encode(x_source)
                             ▼
              eps_hat = F_theta(z, c_source)   <- One-step inversion
                             ▼
              M = abs(eps_hat_source - eps_hat_edit)   <- Self-guided mask
                             ▼
              z_edit = G_IP(eps_hat, c_edit, c_x) + ARaM(M, s_y, s_edit, s_non-edit)
                             ▼
              x_edit = VAE_decode(z_edit)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: Edited image x_edit                                    │
└─────────────────────────────────────────────────────────────────┘
```

**Thời gian:** ~0.23s trên A100 (paper) · ~3–15s trên Mac M4 (MPS) · vài giây trên Colab T4

---

## 5. Kiến trúc code (repository)

```
SwiftEdit/
├── infer.py                          # Pipeline inference chính
├── models.py                         # SBv2 generator + inversion network
├── src/
│   ├── attention_processor.py        # IP-Adapter base
│   ├── mask_attention_processor.py   # Cross-attention có mask
│   └── mask_ip_controller.py         # ARaM controller
├── assets/imgs_demo/                 # Ảnh demo
└── swiftedit_weights/                # Checkpoint (tải riêng, ~GB)
```

**Backbone:** SwiftBrushv2 — one-step text-to-image, distilled từ SD-Turbo với cải tiến CLIP loss và model merging.

**IP-Adapter branch** `G_IP`: tích hợp image condition qua decoupled cross-attention, giảm gánh nặng lên inverted noise.

---

## 6. So sánh với các hướng tiếp cận

| Nhóm | Ví dụ | Steps (inv + edit) | Runtime ~ | Đặc điểm |
|---|---|---|---|---|
| **Multi-step** | DDIM + P2P | 50 + 50 | 26–134s | Chất lượng ổn, rất chậm |
| **Multi-step + opt** | Null-text + P2P | 50 + optimize | ~134s | PSNR cao, cực chậm |
| **Few-step** | TurboEdit | 4 + 4 | ~1.3s | Nhanh hơn, vẫn multi-step |
| **Few-step** | ICD (SD 1.5) | 3–4 | ~1.6s | PSNR tốt, CLIP thấp hơn |
| **One-step** | **SwiftEdit** | **1 + 1** | **~0.23s** | **Nhanh nhất, cạnh tranh CLIP/PSNR** |

### Kết quả định lượng (PieBench, Table 1 paper)

| Method | PSNR↑ | MSE×10⁴↓ | CLIP-Whole↑ | CLIP-Edited↑ | Time (s)↓ |
|---|---|---|---|---|---|
| DDIM + P2P | 17.87 | 219.88 | 25.01 | 22.44 | 25.98 |
| NT-Inv + P2P | 27.03 | 35.86 | 24.75 | 21.86 | 134.06 |
| TurboEdit | 22.43 | 9.48 | 25.49 | 21.82 | 1.32 |
| ICD (SD 1.5) | 26.93 | 3.32 | 22.42 | 19.07 | 1.62 |
| **SwiftEdit** | **23.33** | **6.60** | **25.16** | **21.25** | **0.23** |

→ SwiftEdit **nhanh hơn rất nhiều**, PSNR/CLIP **cạnh tranh** với few-step và nhiều multi-step method.

---

## 7. Input / Output

### Input

| Thành phần | Bắt buộc | Mô tả |
|---|---|---|
| Source image | ✅ | Ảnh RGB cần sửa |
| Edit prompt | ✅ | Mô tả nội dung sau khi sửa |
| Source prompt | Khuyến nghị | Mô tả ảnh gốc; hỗ trợ mask & reconstruction |
| Editing mask | Không | Binary/soft mask; tự sinh nếu bỏ qua |
| Hyperparameters ARaM (`s_y`, `s_edit`, `s_non-edit`) | Không | Điều khiển cường độ chỉnh sửa |

### Output

| Thành phần | Mô tả |
|---|---|
| Edited image | Ảnh đã chỉnh sửa theo edit prompt |
| Editing mask | (Tùy chọn) Mask vùng sửa |
| Metrics | PSNR, MSE, CLIP — khi đánh giá trên PieBench |

---

## 8. Loại chỉnh sửa hỗ trợ

Theo benchmark **PieBench** (700 mẫu, 10 loại):

1. Random editing  
2. **Change object** — đổi vật thể  
3. Add / Delete object  
4. **Change attribute** — màu, chất liệu, …  
5. Change background  
6. Change texture  
7. **Change style** — liên quan chủ đề *thay đổi phong cách*  
8. Change action / pose  
9. Change counting  

SwiftEdit mạnh ở **semantic editing cục bộ** (đổi đối tượng, thuộc tính). **Style transfer** phức tạp (watercolor, anime, …) có thể kém hơn — phù hợp khảo sát thêm trong đề tài.

---

## 9. Đánh giá (metrics)

| Metric | Ý nghĩa / cách đo | Hướng tốt |
|---|---|---|
| **PSNR-unedit** | PSNR giữa ảnh nguồn và ảnh edited trên background `(1 - mask)` | Cao ↑ |
| **MSE-unedit** | Sai số pixel trên background `(1 - mask)` | Thấp ↓ |
| **CLIP-Whole** | CLIPScore giữa toàn ảnh edited và `edit_prompt` | Cao ↑ |
| **CLIP-Edited** | CLIPScore giữa vùng edit mask và `edit_prompt` | Cao ↑ |
| **Runtime** | Thời gian một lần gọi `edit_image()` sau khi model đã load | Thấp ↓ |
| **IoU / Dice** | Mask tự sinh vs GT mask của PIE-Bench | Cao ↑ |

Dataset chuẩn: [PIE-Bench / PnP Inversion](https://github.com/cure-lab/PnPInversion) — 700 ảnh, 10 loại edit, có source/edit prompt và GT mask. SwiftEdit CVPR 2025 dùng các metric này trong Table 1; CLIP-Whole/Edited dựa trên CLIP/CLIPScore để đo độ khớp ảnh-văn bản.

---

## 10. Yêu cầu triển khai

| Thành phần | Paper / repo gốc | Đề tài CS2309 |
|---|---|---|
| GPU | NVIDIA A100 40GB | Mac M4 (MPS) + Colab T4 |
| VRAM | ≥ 24GB khuyến nghị | 24GB unified memory (Mac) |
| Python | 3.12 | 3.12 |
| PyTorch | CUDA 11.8 | Mac: MPS · Colab: CUDA |
| Checkpoint | GitHub Releases v1.0 | Lưu Drive, không commit Git |

**Lưu ý Mac:** Repo gốc cài PyTorch CUDA — cần cài bản Apple Silicon và `device = "mps"`.

---

## 11. Ưu điểm và hạn chế

### Ưu điểm

- **Tốc độ vượt trội** — one-step inversion + one-step editing
- **Không cần vẽ mask** — self-guided mask tự động
- **Code + weights công khai** — tái hiện được cho đề tài môn học
- **Benchmark rõ ràng** — PieBench, so sánh nhiều baseline
- **Điều khiển linh hoạt** — hyperparameter ARaM, mask tùy chọn

### Hạn chế

- Phụ thuộc chất lượng **SwiftBrushv2** và dữ liệu huấn luyện
- Prompt **tiếng Anh**; chưa tối ưu đa ngôn ngữ
- Style editing phức tạp có thể không ổn định
- Cần GPU đủ RAM; Mac MPS chậm hơn paper
- Training nặng (100k + 180k iter) — đề tài sinh viên nên dùng **checkpoint pretrained**

---

## 12. Liên hệ với đề tài CS2309

| Khía cạnh môn học | SwiftEdit liên quan |
|---|---|
| Diffusion models | One-step SBv2, inversion |
| Image editing | Text-guided, localized edit |
| Attention mechanism | ARaM, cross-attention rescaling |
| Segmentation / mask | Self-guided editing mask; SAM 3 chỉ là hướng phân tích mask optional |
| Evaluation | PSNR, CLIP, PieBench |
| Ứng dụng | Chỉnh sửa ảnh nhanh, gần on-device (Mac M4); tối ưu inference realtime |

**Hướng nghiên cứu đề tài (không train nặng):** tái hiện inference, ablation hyperparameter, đánh giá PieBench subset, thử style/weather global edit & ảnh bối cảnh Việt Nam.

**Hướng mở rộng được chọn:** **SwiftEdit-RT: Realtime-Oriented Inference Acceleration**. Ý tưởng là giữ đúng điểm mạnh realtime của SwiftEdit bằng cách profile từng module, bỏ overhead không cần thiết, cache latent/embedding cho demo tương tác và thử các tối ưu inference như `fp16`, `channels_last`, `torch.compile` hoặc TinyVAE/TAESD. Hướng SAM 3 mask replacement được chuyển thành optional/không chọn làm hướng chính vì thêm segmentation model làm tăng latency end-to-end.

**Hướng ứng dụng đang khảo sát:** **Global scene/style editing** như ngày↔đêm, mùa, mưa↔nắng. Hướng này khả thi mức trung bình: đáng làm để phân tích giới hạn của SwiftEdit, nhưng không dùng mask IoU/PSNR nền; thay bằng CLIP target/style, DINO/CLIP image similarity, LPIPS/SSIM phụ, IQA/human rating và runtime.

**Hướng ứng dụng mới:** **Object removal / inpainting**. Hướng này phù hợp hơn global style vì vẫn là local edit có mask: dùng prompt `"without [object]"`, so SwiftEdit self-guided mask với user/GT mask, đo detector confidence drop, background preservation ngoài mask và realism vùng inpaint. Khả thi mức trung bình-cao với object nhỏ/vừa; cần so với baseline inpainting chuyên dụng như LaMa khi object lớn hoặc nền phức tạp.

---

## 13. Tài liệu tham khảo

```bibtex
@InProceedings{Nguyen_2025_CVPR,
    author    = {Nguyen, Trong-Tung and Nguyen, Quang and Nguyen, Khoi and Tran, Anh and Pham, Cuong},
    title     = {SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion},
    booktitle = {CVPR},
    year      = {2025},
    pages     = {21492--21501}
}
```

**Đọc thêm:**

- [SwiftBrush v2 (ECCV 2025)](https://github.com/) — backbone one-step T2I
- [PIE-Bench / PnP Inversion (ICLR 2024)](https://github.com/cure-lab/PnPInversion)
- [Diffusers AutoencoderTiny / TAESD](https://huggingface.co/docs/diffusers/en/api/models/autoencoder_tiny)
- [Diffusers inference optimization](https://huggingface.co/docs/diffusers/main/optimization/fp16)
- [PyTorch `torch.compile` + Diffusers](https://docs.pytorch.org/devlogs/inductor/2026-05-11-torch-compile-and-diffusers/)
- [SAM 3: Segment Anything with Concepts](https://github.com/facebookresearch/sam3) — optional mask analysis
- [LaMa: Resolution-robust Large Mask Inpainting](https://openaccess.thecvf.com/content/WACV2022/html/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.html)
- [CLIP (ICML 2021)](https://proceedings.mlr.press/v139/radford21a) · [CLIPScore (EMNLP 2021)](https://arxiv.org/abs/2104.08718)
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
- [TurboEdit](https://github.com/GaMaLielD/TurboEdit) — baseline few-step

---

*Tài liệu overview — cập nhật cho đề tài CS2309.CH201. Chi tiết kế hoạch thực nghiệm xem [`SwiftEdit_DeTai_CS2309.md`](./SwiftEdit_DeTai_CS2309.md).*
