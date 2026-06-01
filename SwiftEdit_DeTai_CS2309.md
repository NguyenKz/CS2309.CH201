# Đề tài: SwiftEdit — Chỉnh sửa ảnh theo văn bản một bước (One-step Text-guided Image Editing)

**Môn học:** CS2309.CH201 — Chuyên đề nghiên cứu và ứng dụng về Thị giác máy tính  
**Chủ đề:** Chỉnh sửa và thay đổi phong cách ảnh  
**Đề tài:** SwiftEdit  
**Tác giả gốc:** Trong-Tung Nguyen, Quang Nguyen, Khoi Nguyen, Anh Tran, Cuong Pham (Qualcomm AI Research)  
**Paper:** [SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_SwiftEdit_Lightning_Fast_Text-Guided_Image_Editing_via_One-Step_Diffusion_CVPR_2025_paper.pdf)  
**Mã nguồn:** [https://github.com/Qualcomm-AI-research/SwiftEdit](https://github.com/Qualcomm-AI-research/SwiftEdit)  
**Trang dự án:** [https://swift-edit.github.io/](https://swift-edit.github.io/)

**Môi trường thực nghiệm:** **MacBook Air M4, 24GB** (local) + **Google Colab** (GPU CUDA — T4 / A100)

---

## 1. Overview

### 1.1. Giới thiệu tổng quan

**SwiftEdit** là phương pháp chỉnh sửa ảnh theo hướng dẫn văn bản (text-guided image editing) được công bố tại **CVPR 2025**. Phương pháp này giải quyết bài toán chỉnh sửa ảnh tự nhiên bằng prompt tiếng Anh, ví dụ: đổi "basket of apples" thành "basket of puppies", hoặc "mouth closed" thành "mouth opened".

Điểm khác biệt cốt lõi của SwiftEdit so với các phương pháp diffusion truyền thống (Prompt-to-Prompt, Null-text Inversion, MasaCtrl, Plug-and-Play, …) là **cả quá trình inversion và editing đều chỉ cần một bước (one-step)**, đạt thời gian suy luận khoảng **0.23 giây** trên GPU A100 — nhanh hơn ít nhất **50 lần** so với các phương pháp multi-step.

SwiftEdit xây dựng trên mô hình sinh ảnh một bước **SwiftBrushv2 (SBv2)** và gồm hai thành phần chính:

1. **One-step Inversion Framework:** Mạng inversion \(F_\theta\) ánh xạ ảnh nguồn về latent noise có thể chỉnh sửa trong một bước, được huấn luyện theo chiến lược **hai giai đoạn** (synthetic data → real data).
2. **Attention Rescaling for Mask-aware Editing (ARaM):** Cơ chế điều chỉnh cross-attention theo vùng mask để chỉnh sửa cục bộ, bảo toàn background, đồng thời cho phép điều khiển cường độ chỉnh sửa qua các hệ số \(s_y\), \(s_{edit}\), \(s_{non\text{-}edit}\).

Một đặc điểm quan trọng là SwiftEdit có thể **tự động trích xuất editing mask** từ sự khác biệt giữa inverted noise của source prompt và edit prompt, không bắt buộc người dùng vẽ mask thủ công.

### 1.2. Kiến trúc và pipeline suy luận

```
Ảnh nguồn + Source prompt + Edit prompt
        │
        ▼
┌─────────────────────────┐
│  One-step Inversion Fθ  │  → Inverted noise ε̂
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ Self-guided mask M      │  → |ε̂_source − ε̂_edit|
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ G_IP + ARaM             │  → Edited latent → VAE decode
└─────────────────────────┘
        │
        ▼
    Ảnh đã chỉnh sửa
```

**Các module chính trong repository:**

| Thành phần | Vai trò |
|---|---|
| `models.py` | Mô hình sinh ảnh (SBv2) và mạng inversion |
| `infer.py` | Pipeline suy luận chính |
| `src/mask_attention_processor.py` | Cross-attention có mask |
| `src/mask_ip_controller.py` | Điều khiển attention rescaling |
| `swiftedit_weights/` | Checkpoint đã huấn luyện sẵn |

### 1.3. Bối cảnh và động lực

Các phương pháp chỉnh sửa ảnh diffusion hiện tại thường yêu cầu:

- **Inversion đa bước** (DDIM, Null-text Inversion) để tìm noise ban đầu tương ứng ảnh nguồn.
- **Sampling đa bước** kết hợp thao tác attention để áp dụng thay đổi theo prompt.

Quy trình này tốn **12–130+ giây** mỗi lần chỉnh sửa, không phù hợp ứng dụng real-time hoặc on-device. SwiftEdit hướng tới **chỉnh sửa tức thì** với chất lượng cạnh tranh so với multi-step và few-step methods.

---

## 2. Lý do chọn đề tài

| Tiêu chí | Lý do |
|---|---|
| **Tính thời sự** | Chỉnh sửa ảnh bằng ngôn ngữ tự nhiên là hướng nghiên cứu nóng trong CV và generative AI (2024–2025). |
| **Liên quan trực tiếp môn học** | Bài toán kết hợp nhiều khái niệm thị giác máy tính: diffusion model, inversion, attention mechanism, segmentation mask, đánh giá chất lượng ảnh. |
| **Có mã nguồn và checkpoint công khai** | Repository chính thức cung cấp inference code và pretrained weights, phù hợp điều kiện **không training nặng**. |
| **Đóng góp khoa học rõ ràng** | Paper CVPR 2025 có benchmark (PieBench), ablation study và so sánh với nhiều baseline, tạo nền tảng để tái hiện và mở rộng thự nghiệm. |
| **Phù hợp Mac Air M4 + Colab** | Mac cho dev/demo; Colab Free (T4) cho CUDA benchmark — **không cần máy GPU riêng**. |
| **Ứng dụng thực tế** | Demo chỉnh sửa ảnh nhanh trên Mac cho người sáng tạo nội dung, thiết kế. |
| **Phù hợp nguồn lực sinh viên** | Tập trung **inference, đánh giá, ablation** — không train lại toàn bộ mô hình. |

---

## 3. Input / Output của bài toán

### 3.1. Input

| Input | Mô tả | Bắt buộc |
|---|---|---|
| **Source image** \(x_{source}\) | Ảnh gốc cần chỉnh sửa (RGB) | Có |
| **Edit prompt** \(y_{edit}\) | Mô tả văn bản về nội dung mong muốn sau khi chỉnh sửa | Có |
| **Source prompt** \(y_{source}\) | Mô tả ảnh gốc; giúp reconstruction và trích xuất mask | Khuyến nghị |
| **Editing mask** \(M\) | Vùng cần chỉnh sửa (binary/soft mask) | Không (có thể tự sinh) |
| **Hyperparameters** | \(s_y\), \(s_{edit}\), \(s_{non\text{-}edit}\) — điều khiển cường độ chỉnh sửa | Tùy chọn |

### 3.2. Output

| Output | Mô tả |
|---|---|
| **Edited image** \(x_{edit}\) | Ảnh đã chỉnh sửa theo edit prompt |
| **Editing mask** (tùy chọn) | Mask tự sinh hoặc mask người dùng cung cấp |
| **Metrics** (trong thực nghiệm) | PSNR, MSE (background), CLIP-Whole, CLIP-Edited, thời gian suy luận |

### 3.3. Ví dụ minh họa

```
Input:
  - Ảnh: chú mèo cam ngồi trên hàng rào
  - Source prompt: "An orange cat sitting on top of a fence."
  - Edit prompt: "A black cat sitting on top of a fence."

Output:
  - Ảnh: chú mèo đen ngồi trên hàng rào, background giữ nguyên
  - Thời gian: ~0.23s (A100) / **~3–15s (Mac Air M4, MPS)** — vẫn nhanh hơn nhiều so với multi-step
```

---

## 4. Môi trường thực nghiệm: Mac M4 + Google Colab

> **Chiến lược chính:** Dùng **Mac** cho phát triển, demo và phân tích; dùng **Google Colab** cho workload cần CUDA (benchmark PieBench, so sánh baseline, tái hiện số liệu paper). Hai môi trường **bổ sung cho nhau**, không thay thế hoàn toàn.

### 4.1. MacBook Air M4 (24GB) — Môi trường local

| Thành phần | MacBook Air M4 (24GB) |
|---|---|
| CPU | Apple M4 (10-core) |
| GPU | GPU tích hợp (Metal / MPS) |
| Bộ nhớ | 24GB **unified memory** (CPU + GPU dùng chung) |
| Backend PyTorch | **MPS** (Metal Performance Shaders), **không có CUDA** |
| Hệ điều hành | macOS |

### Khả năng chạy SwiftEdit trên Mac M4

| Khía cạnh | Đánh giá | Ghi chú |
|---|---|---|
| **Inference (demo, ablation)** | ✅ Khả thi | 24GB unified memory đủ cho inference 512×512 |
| **Tốc độ** | ⚠️ Chậm hơn paper | ~3–15 giây/ảnh (MPS) vs 0.23s (A100) |
| **PieBench batch lớn** | ⚠️ Chậm | 15–25 mẫu OK; batch 50–100 → **chuyển Colab** |
| **Fine-tune** | ❌ Không | Chuyển sang Colab |
| **Viết báo cáo, demo Gradio** | ✅ Tốt | Làm việc hàng ngày trên Mac |

### Cài đặt trên macOS

```bash
# 1. Tạo môi trường
conda create -n SwiftEdit python=3.12 -y
conda activate SwiftEdit

# 2. PyTorch cho Mac (MPS) — KHÔNG dùng dòng cu118 trong requirements.txt
pip install torch torchvision torchaudio

# 3. Các package còn lại (bỏ qua dòng --extra-index-url cu118)
pip install transformers==4.37.2 accelerate ftfy tensorboard Jinja2 \
            diffusers==0.22.0 huggingface-hub==0.25.2 einops
pip install numpy==1.26.4

# 4. Clone và tải weights
git clone https://github.com/Qualcomm-AI-research/SwiftEdit.git
cd SwiftEdit
# Tải checkpoint từ GitHub Releases v1.0 → giải nén vào swiftedit_weights/
```

**Sửa device trong code** (`infer.py` / `models.py` — tùy repo gốc):

```python
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

**Mẹo giảm RAM trên Mac:**

- Dùng `torch.float16` / `model.half()` nếu code hỗ trợ (giảm ~40% memory).
- Giữ độ phân giải **512×512** như paper; tránh 1024×1024 trên 24GB.
- Đóng Chrome, IDE nặng khi chạy inference.
- Nếu gặp lỗi MPS: thử `PYTORCH_ENABLE_MPS_FALLBACK=1` hoặc fallback `device="cpu"` (chỉ debug).

---

### 4.2. Google Colab — Môi trường GPU CUDA

| Thành phần | Google Colab (Free / Pro) |
|---|---|
| GPU | **T4** (Free, ~15GB) hoặc **A100** (Pro, nếu có) |
| Backend | **CUDA** — tương thích `requirements.txt` gốc |
| Bộ nhớ | ~12–15GB VRAM (T4) — đủ inference SwiftEdit |
| Thời gian session | Free ~12h/session, có thể disconnect |
| Lưu trữ | Google Drive mount — lưu weights, kết quả |

### Khả năng trên Colab

| Khía cạnh | Đánh giá | Ghi chú |
|---|---|---|
| **Cài đặt repo gốc (CUDA)** | ✅ Trực tiếp | `pip install -r requirements.txt` như README |
| **PieBench 50–100 mẫu** | ✅ Khuyến nghị | ~5–20 phút trên T4; gần môi trường paper |
| **So sánh TurboEdit / baseline** | ✅ Khuyến nghị | Multi-step methods cần CUDA |
| **Tái hiện runtime ~0.23s** | ⚠️ Chỉ A100 | T4 chậm hơn A100 nhưng vẫn nhanh hơn Mac MPS |
| **Fine-tune nhẹ** | ⚠️ Tùy chọn | T4 có thể OOM; batch size = 1 |
| **Lưu kết quả lâu dài** | ✅ Drive | Mount Drive, sync về Mac |

### Cài đặt trên Google Colab

**Bước 1 — Tạo notebook mới:** [Google Colab](https://colab.research.google.com/) → Runtime → Change runtime type → **T4 GPU**.

**Bước 2 — Mount Google Drive** (lưu weights + kết quả, tránh tải lại mỗi session):

```python
from google.colab import drive
drive.mount('/content/drive')

WORK_DIR = '/content/drive/MyDrive/CS2309_SwiftEdit'  # thư mục riêng cho đề tài
!mkdir -p {WORK_DIR}
%cd {WORK_DIR}
```

**Bước 3 — Clone repo và cài dependencies (CUDA):**

```python
!git clone https://github.com/Qualcomm-AI-research/SwiftEdit.git
%cd SwiftEdit

# Cài theo requirements.txt gốc (CUDA)
!pip install -r requirements.txt
!pip install numpy==1.26.4

# PieBench metrics (nếu đánh giá)
!pip install lpips scikit-image open-clip-torch
```

**Bước 4 — Tải checkpoint (chỉ lần đầu, lưu trên Drive):**

```python
import os
WEIGHTS_DIR = f'{WORK_DIR}/swiftedit_weights'

if not os.path.exists(f'{WEIGHTS_DIR}/.downloaded'):
    %cd /content/SwiftEdit
    !wget -q https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-aa
    !wget -q https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ab
    !wget -q https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ac
    !wget -q https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ad
    !wget -q https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ae
    !cat swiftedit_weights.tar.gz.part-* > swiftedit_weights.tar.gz
    !tar zxf swiftedit_weights.tar.gz -C {WORK_DIR}
    !touch {WEIGHTS_DIR}/.downloaded
else:
    !ln -sf {WEIGHTS_DIR} /content/SwiftEdit/swiftedit_weights
    print("Weights đã có trên Drive, bỏ qua tải.")
```

**Bước 5 — Kiểm tra GPU:**

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
device = "cuda"  # Colab luôn dùng cuda
```

**Bước 6 — Tải PieBench (Colab):**

```python
!git clone https://github.com/Prompt-to-Prompt/PieBench.git {WORK_DIR}/PieBench
# Hoặc tải subset từ Hugging Face / Drive nếu repo gốc quá lớn
```

**Mẹo Colab:**

- **Lưu weights trên Drive** — file checkpoint vài GB, không tải lại mỗi session.
- **Lưu kết quả CSV/ảnh về Drive** — sync về Mac để viết báo cáo.
- Session disconnect → chạy lại cell mount + symlink weights (nhanh, không cần tải lại).
- Colab Free có giới hạn GPU — chạy batch lớn vào ban đêm hoặc chia nhiều session.
- Nếu OOM trên T4: dùng `torch.float16`, batch size = 1, giảm số mẫu/session.

---

### 4.3. Phân công công việc Mac ↔ Colab

| Công việc | Mac M4 | Google Colab | Ghi chú |
|---|---|---|---|
| Đọc paper, viết báo cáo | ✅ Chính | — | |
| Clone repo, hiểu code | ✅ | ✅ | Mac tiện hơn khi dev |
| Chạy demo lần đầu | ✅ | ✅ | Mac xác nhận pipeline; Colab xác nhận CUDA |
| Ablation hyperparameter (5–10 ảnh) | ✅ Chính | Tùy chọn | Mac đủ, lưu grid ảnh |
| PieBench metrics (50–100 mẫu) | 15–25 mẫu | ✅ **Chính** | Colab nhanh + CUDA |
| So sánh TurboEdit / baseline | — | ✅ **Chính** | Cần CUDA |
| Self-guided mask vs GT mask | ✅ | ✅ | Colab cho batch; Mac cho visualize |
| Bộ ảnh tự thu thập (VN) | ✅ Chính | Tùy chọn | Thu ảnh trên Mac, batch eval trên Colab |
| Demo Gradio / Streamlit | ✅ Chính | — | Chạy local trên Mac |
| Fine-tune nhẹ (tùy chọn) | ❌ | ✅ | Chỉ Colab |
| Đo runtime so sánh paper | M4 (MPS) | ✅ T4/A100 (CUDA) | Báo cáo **cả hai** + paper |

### 4.4. Luồng làm việc Mac + Colab (workflow)

```
  ┌──────────────────────────────────────────────────────────┐
  │  MAC AIR M4                                              │
  │  ① Clone repo, đọc infer.py / models.py                  │
  │  ② Cài MPS, chạy demo assets/imgs_demo                   │
  │  ③ Ablation s_y, s_edit, s_non-edit (5–10 ảnh)           │
  │  ④ Thu thập ảnh VN + viết báo cáo                        │
  │  ⑤ (Tùy chọn) Demo Gradio local                          │
  └────────────────────────┬─────────────────────────────────┘
                           │ Upload ảnh VN / subset PieBench
                           │ Download kết quả CSV + ảnh từ Drive
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │  GOOGLE COLAB (T4 GPU)                                   │
  │  ⑥ Setup notebook + mount Drive + symlink weights        │
  │  ⑦ Chạy PieBench 50–100 mẫu → PSNR, CLIP, MSE, runtime  │
  │  ⑧ So sánh SwiftEdit vs TurboEdit (20 mẫu chung)         │
  │  ⑨ Lưu metrics.csv + edited_images/ → Google Drive       │
  └────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │  MAC — TỔNG HỢP BÁO CÁO                                  │
  │  ⑩ Gộp bảng metrics Colab + ảnh ablation Mac             │
  │  ⑪ So sánh runtime: Mac MPS vs Colab T4 vs Paper A100    │
  │  ⑫ Viết Kết luận                                         │
  └──────────────────────────────────────────────────────────┘
```

> **Ghi chú cho báo cáo:** Ghi rõ nguồn từng số liệu:
> - *Mac M4 (MPS)* — demo, ablation, runtime local
> - *Colab T4/A100 (CUDA)* — PieBench batch, baseline comparison
> - *Paper (A100)* — reference từ Table 1, không tự đo

---

## 5. Những vấn đề cần nghiên cứu

> Phần này xác định các câu hỏi nghiên cứu phù hợp phạm vi đề tài môn học, ưu tiên **tái hiện, phân tích và mở rộng thự nghiệm** thay vì huấn luyện mô hình từ đầu.

### 5.1. Vấn đề cốt lõi (theo bài gốc)

1. **Làm sao invert ảnh thực về editable noise trong một bước?**
   - Inversion đa bước (DDIM, Null-text) không phù hợp mô hình one-step.
   - SwiftEdit học mạng encoder-based \(F_\theta\), huấn luyện 2 stage với synthetic + real data.

2. **Làm sao chỉnh sửa cục bộ mà không phá background?**
   - Dùng self-guided mask từ sự khác biệt inverted noise.
   - ARaM tách riêng attention cho vùng edit và non-edit.

3. **Cân bằng giữa tốc độ, chất lượng chỉnh sửa và bảo toàn background?**
   - So sánh PSNR/MSE vs CLIP score vs runtime trên PieBench.

### 5.2. Vấn đề mở rộng (phù hợp đề tài sinh viên)

4. **SwiftEdit hoạt động thế nào trên dữ liệu Việt Nam / ảnh tự thu thập?**
   - PieBench dùng prompt tiếng Anh; cần kiểm tra độ ổn định trên ảnh thực tế (chân dung, phong cảnh, sản phẩm, …).

5. **Ảnh hưởng của hyperparameter ARaM đến chất lượng chỉnh sửa?**
   - \(s_y\): cường độ alignment với edit prompt trong vùng mask.
   - \(s_{edit}\), \(s_{non\text{-}edit}\): trade-off giữa sửa vùng mục tiêu và giữ vùng nền.

6. **Self-guided mask vs ground-truth mask — mức chênh lệch bao nhiêu?**
   - Paper báo cáo kết quả gần tương đương GT mask; cần xác minh trên subset PieBench hoặc bộ ảnh riêng.

7. **SwiftEdit so với baseline multi-step / few-step trên cùng phần cứng?**
   - So sánh trực quan và định lượng với TurboEdit, ReNoise, hoặc phương pháp multi-step đơn giản hơn (nếu tài nguyên cho phép).

8. **Giới hạn khi áp dụng cho "thay đổi phong cách ảnh" (style transfer)?**
   - SwiftEdit tập trung semantic editing (đổi đối tượng, thuộc tính); cần khảo sát khả năng chuyển phong cách (ví dụ: "oil painting style", "anime style") qua prompt.

10. **Phân công Mac vs Colab có hiệu quả không?**
    - Task nào nên chạy local, task nào cần CUDA? Runtime Colab T4 so với Mac MPS?

11. **SwiftEdit chạy thế nào trên Mac Apple Silicon (M4, unified memory)?**
    - Repo gốc target CUDA; cần port sang MPS cho local dev.

12. **Yêu cầu tài nguyên và khả năng triển khai on-device?**
    - Mac M4 là case study gần on-device; Colab T4 mô phỏng server-side GPU.

### 5.3. Câu hỏi nghiên cứu đề xuất (Research Questions)

- **RQ1:** SwiftEdit có tái hiện được kết quả định lượng trên PieBench (PSNR, CLIP) khi chỉ dùng checkpoint pretrained không?
- **RQ2:** Tham số ARaM ảnh hưởng thế nào đến trade-off giữa editing semantics và background preservation?
- **RQ3:** Self-guided mask đạt độ chính xác (IoU/Dice so với GT mask) ở mức nào trên các loại chỉnh sửa khác nhau?
- **RQ4:** So với phương pháp few-step (TurboEdit) hoặc multi-step (P2P + DDIM), SwiftEdit đánh đổi bao nhiêu chất lượng để đạt tốc độ one-step?
- **RQ5:** Trên Mac M4 (MPS) và Colab T4 (CUDA), runtime và memory peak của SwiftEdit khác nhau thế nào so với paper (A100)?
- **RQ6:** Colab có giúp tái hiện metrics PieBench (PSNR, CLIP) gần với Table 1 paper hơn Mac MPS không?
- **RQ7:** (Tùy chọn — Colab) Fine-tune nhẹ Stage 2 trên Colab có cải thiện reconstruction trên domain-specific không?

---

## 6. Những đóng góp có thể thực hiện

> Định hướng phù hợp sinh viên: **inference-first**, train nhẹ (nếu có), tập trung đánh giá và phân tích.

### 6.1. Đóng góp bắt buộc / cốt lõi

| # | Đóng góp | Mức độ | Training |
|---|---|---|---|
| C1 | **Cài đặt và chạy SwiftEdit** từ repository chính thức, tải checkpoint, chạy demo trên `assets/imgs_demo` | Bắt buộc | Không |
| C2 | **Phân tích pipeline** inversion → mask extraction → ARaM editing; mô tả luồng xử lý và vai trò từng module | Bắt buộc | Không |
| C3 | **Thực nghiệm ablation hyperparameter** trên tập ảnh mẫu: thay đổi \(s_y\), \(s_{edit}\), \(s_{non\text{-}edit}\), so sánh trực quan | Bắt buộc | Không |
| C4 | **Đánh giá định lượng PieBench** | **Colab:** 50–100 mẫu (PSNR, CLIP, runtime CUDA). **Mac:** 10–15 mẫu xác nhận chéo | Không |

### 6.2. Đóng góp mở rộng (chọn 1–2)

| # | Đóng góp | Mô tả | Training |
|---|---|---|---|
| C5 | **So sánh với baseline trên Colab** | TurboEdit hoặc DDIM+P2P trên 20 mẫu chung; bảng metrics + ảnh | Colab |
| C6 | **Bộ ảnh tự xây dựng** | Thu thập trên Mac; chạy inference Mac + Colab | Không |
| C7 | **Phân tích self-guided mask** | IoU/Dice trên PieBench — **Colab batch**, visualize trên Mac | Colab |
| C8 | **Thử nghiệm style editing** | Khảo sát prompt chuyển phong cách | Không |
| C9 | **Demo ứng dụng** | Gradio trên Mac local | Không |
| C10 | **Benchmark đa nền tảng** | Runtime: Mac MPS vs Colab T4 vs Paper A100 | Không |
| C11 | **Colab notebook tái sử dụng** | Notebook setup + eval script lưu trên Drive, dùng lại nhiều session | Không |
| C12 | **Fine-tune nhẹ trên Colab (tùy chọn)** | Stage 2 vài nghìn iter; **không train trên Mac** | Colab |

### 6.3. Phạm vi không thực hiện (để giới hạn đề tài)

- Huấn luyện lại toàn bộ inversion network từ đầu (Stage 1: 100k iter + Stage 2: 180k iter).
- Xây dựng mô hình one-step generation mới thay SBv2.
- Huấn luyện diffusion model đa bước làm baseline từ scratch.

---

## 7. Kế hoạch nghiên cứu / thực nghiệm

### 7.1. Giai đoạn 1 — Tìm hiểu lý thuyết (Tuần 1–2)

- [ ] Đọc paper SwiftEdit (Abstract, Sec. 3–5).
- [ ] Tìm hiểu nền tảng: diffusion one-step (SwiftBrushv2), DDIM inversion, IP-Adapter, PieBench benchmark.
- [ ] Tổng hợp sơ đồ pipeline và bảng so sánh với related work (P2P, NT-Inv, TurboEdit, ICD, ReNoise).

**Deliverable:** Phần Overview + Related Work trong báo cáo.

### 7.2. Giai đoạn 2a — Cài đặt trên Mac M4 (Tuần 2)

- [ ] Clone repository trên Mac.
- [ ] Cài PyTorch Mac (MPS) — xem Mục 4.1.
- [ ] Tải checkpoint (hoặc tải 1 lần trên Colab → lưu Drive → sync về Mac).
- [ ] Chạy demo `assets/imgs_demo`; ghi thời gian + RAM peak.
- [ ] Đọc và ghi chú `infer.py`, `models.py`.

**Deliverable:** Demo Mac chạy OK, ảnh kết quả đầu tiên.

### 7.3. Giai đoạn 2b — Setup Google Colab (Tuần 2–3)

- [ ] Tạo notebook `CS2309_SwiftEdit.ipynb` trên Colab (xem Phụ lục C).
- [ ] Tạo thư mục `MyDrive/CS2309_SwiftEdit/` trên Google Drive.
- [ ] Mount Drive, clone repo, cài `requirements.txt` (CUDA).
- [ ] Tải checkpoint lên Drive (chỉ lần đầu — xem Mục 4.2).
- [ ] Chạy demo cùng ảnh với Mac → **so sánh kết quả** (phải giống nhau nếu setup đúng).
- [ ] Ghi GPU name (T4/A100), thời gian/ảnh Colab.

**Deliverable:** Notebook Colab tái sử dụng được, weights trên Drive, demo CUDA OK.

### 7.4. Giai đoạn 3 — Thực nghiệm cơ bản (Tuần 3–5)

#### 3a. Ablation hyperparameter ARaM — **Mac**

- [ ] Chọn 5–10 ảnh đại diện.
- [ ] Thay đổi \(s_y\), \(s_{edit}\), \(s_{non\text{-}edit}\) (0.5, 1.0, 1.5).
- [ ] Lưu grid ảnh trên Mac, đưa vào báo cáo.

#### 3b. Self-guided mask vs GT mask — **Colab**

- [ ] Tải subset PieBench có GT mask lên Colab.
- [ ] Chạy SwiftEdit với/không mask; tính IoU/Dice (batch trên Colab).
- [ ] Download ảnh mask visualization về Mac.

#### 3c. Đánh giá PieBench — **Colab (chính) + Mac (xác nhận)**

- [ ] **Colab:** 50–100 mẫu PieBench → PSNR, MSE, CLIP-Whole, CLIP-Edited, runtime.
- [ ] **Mac:** 10–15 mẫu trùng subset → xác nhận metrics gần khớp Colab.
- [ ] Lưu `metrics.csv` + `edited_images/` lên Google Drive.
- [ ] So sánh với Table 1 paper (reference A100).

**Deliverable:** `metrics.csv` (Colab), grid ablation (Mac), biểu đồ trade-off.

### 7.5. Giai đoạn 4 — So sánh và mở rộng (Tuần 5–7)

#### 4a. So sánh với phương pháp khác (chọn ít nhất 1)

| Phương pháp | Loại | Runtime (paper) | Ghi chú |
|---|---|---|---|
| DDIM + P2P | Multi-step (50 steps) | ~26s | Baseline kinh điển |
| NT-Inv + P2P | Multi-step + optimization | ~134s | PSNR cao, rất chậm |
| TurboEdit | Few-step (4 steps) | ~1.32s | Đối thủ gần nhất về tốc độ |
| ICD (SD 1.5) | Few-step | ~1.62s | PSNR tốt, CLIP thấp hơn |
| **SwiftEdit** | **One-step** | **~0.23s** | **Đề tài chính** |

- [ ] **Colab:** Chạy TurboEdit (hoặc 1 multi-step method) trên **20 mẫu chung** với SwiftEdit.
- [ ] **Mac:** So sánh định tính 5 mẫu đã chạy trên Colab (xem ảnh side-by-side).
- [ ] Lập bảng: SwiftEdit vs baseline vs paper (Table 1).
- [ ] Phân tích runtime: Mac MPS / Colab T4 / Paper A100 (3 cột riêng).

#### 4b. Bộ ảnh tự thu thập — **Mac thu + Colab eval**

- [ ] Xây dựng 20–30 cặp (ảnh, prompt) phù hợp bối cảnh Việt Nam.
- [ ] Phân loại kết quả: thành công / một phần / thất bại.
- [ ] Ghi nhận failure cases (prompt mơ hồ, chỉnh sửa toàn cục, domain lạ).

- [ ] Thu 20–30 cặp (ảnh, prompt) trên Mac.
- [ ] Upload lên Colab Drive folder `custom_vn/`; chạy batch inference.
- [ ] Download kết quả; phân loại success / partial / fail trên Mac.

#### 4c. Style editing (tùy chọn) — **Mac hoặc Colab**

- [ ] Thử prompt dạng: `"same scene in watercolor style"`, `"anime style portrait"`, …
- [ ] Đánh giá định tính khả năng "thay đổi phong cách" so với semantic editing.

**Deliverable:** Bảng so sánh model, phân tích failure cases, (nếu có) demo web.

### 7.6. Giai đoạn 5 — (Tùy chọn) Fine-tune nhẹ trên Colab

> **Không thực hiện trên MacBook Air M4** — training diffusion quá chậm và dễ OOM. Chỉ dùng Google Colab (GPU T4/A100) nếu thực sự cần.

- [ ] Chuẩn bị ~200–500 ảnh + caption trên Colab.
- [ ] Tiếp tục Stage 2 vài nghìn iterations (không train lại Stage 1).
- [ ] So sánh reconstruction (PSNR, LPIPS) trước/sau trên domain target.

**Deliverable:** Bảng ablation fine-tune (Colab), 5–10 ảnh so sánh. *Bỏ qua giai đoạn này vẫn đủ cho đề tài môn học.*

### 7.7. Giai đoạn 6 — Tổng hợp báo cáo (Tuần 7–8)

- [ ] Gộp kết quả Mac (ablation, demo) + Colab (`metrics.csv`, baseline).
- [ ] Viết bảng runtime 3 cột: Mac MPS | Colab T4 | Paper A100.
- [ ] Viết Kết luận và hướng phát triển.
- [ ] Chuẩn bị slide trình bày: pipeline, kết quả định lượng, demo trực quan.

---

## 8. Quá trình nghiên cứu / thực nghiệm

> **Phần này sẽ được cập nhật trong quá trình thực hiện.** Ghi chép theo từng giai đoạn trong Kế hoạch (Mục 7).

### 8.1. Nhật ký thực nghiệm

| Ngày | Giai đoạn | Công việc | Kết quả / Ghi chú |
|---|---|---|---|
| 2026-06-01 | 0. Khởi tạo project | Tạo README, đề tài chi tiết, skill Cursor hỗ trợ nhật ký | Repo CS2309.CH201 sẵn sàng; skill sync README + NHAT_KY + §8.1 (Mac M4) |

### 8.2. Kết quả trung gian

#### 8.2.1. Demo — Mac vs Colab

| Môi trường | Backend | GPU/RAM | Thời gian/ảnh |
|---|---|---|---|
| Mac Air M4 | MPS | 24GB unified | … giây |
| Google Colab | CUDA | T4 / A100 | … giây |
| Paper (ref) | CUDA | A100 40GB | 0.23s |

```
(Chèn ảnh cùng input — output giống nhau trên Mac và Colab)
```

#### 8.2.2. Bảng metrics PieBench (Colab)

| Metric | Colab (tái hiện) | Mac (subset) | Paper (A100) |
|---|---|---|---|
| PSNR ↑ | | | 23.33 |
| MSE ×10⁴ ↓ | | | 6.60 |
| CLIP-Whole ↑ | | | 25.16 |
| CLIP-Edited ↑ | | | 21.25 |
| Runtime (s) ↓ | | | 0.23 |

#### 8.2.3. Ablation hyperparameter

```
(Chèn grid ảnh: thay đổi s_y, s_edit, s_non-edit)
```

#### 8.2.4. So sánh với baseline

```
(Chèn bảng + ảnh so sánh SwiftEdit vs TurboEdit / P2P)
```

### 8.3. Khó khăn gặp phải và cách xử lý

| Vấn đề | Cách xử lý |
|---|---|
| Colab disconnect giữa chừng | Lưu weights + checkpoint progress trên Drive; chia batch nhỏ (20 mẫu/session) |
| OOM trên Colab T4 | `float16`, batch=1, giảm resolution |
| Mac MPS lỗi operator | `PYTORCH_ENABLE_MPS_FALLBACK=1`; chạy mẫu đó trên Colab |
| Metrics Mac ≠ Colab | Kiểm tra cùng seed, cùng hyperparameter, cùng dtype |

---

## 9. So sánh dataset và phương pháp

### 9.1. Dataset sử dụng

| Dataset | Mô tả | Mục đích trong đề tài |
|---|---|---|
| **PieBench** | 700 mẫu, 10 loại editing, có GT mask và metrics chuẩn | Đánh giá định lượng chính |
| **assets/imgs_demo** | Ảnh demo trong repo SwiftEdit | Kiểm tra cài đặt, demo nhanh |
| **Bộ ảnh tự thu thập** | 20–50 ảnh + prompt tự đặt | Đánh giá định tính, bối cảnh VN |
| **CommonCanvas** (tùy chọn) | 5k ảnh thực + caption (dùng trong Stage 2 training gốc) | Fine-tune nhẹ nếu có |

### 9.2. Phân loại 10 loại editing trong PieBench

1. Random editing  
2. Change object  
3. Add object  
4. Delete object  
5. Change attribute (color, material, …)  
6. Change background  
7. Change texture  
8. Change style  
9. Change action/pose  
10. Change counting  

→ **Gợi ý phân tích:** Nhóm loại 2, 5, 8 liên quan trực tiếp chủ đề "chỉnh sửa và thay đổi phong cách ảnh".

### 9.3. Metrics đánh giá

| Metric | Ý nghĩa | Hướng tốt |
|---|---|---|
| **PSNR** | Bảo toàn vùng background | Cao ↑ |
| **MSE** | Sai số pixel vùng background | Thấp ↓ |
| **CLIP-Whole** | Alignment toàn ảnh với edit prompt | Cao ↑ |
| **CLIP-Edited** | Alignment vùng edit với edit prompt | Cao ↑ |
| **Runtime** | Thời gian suy luận | Thấp ↓ |
| **IoU / Dice** (mask) | Độ trùng mask tự sinh vs GT | Cao ↑ |
| **LPIPS / SSIM** (tùy chọn) | Chất lượng cảm nhận / cấu trúc | LPIPS thấp, SSIM cao |

### 9.4. Bảng so sánh phương pháp (tham khảo từ paper)

| Nhóm | Phương pháp | Steps | Runtime ~ | PSNR | CLIP-Whole | Phù hợp so sánh |
|---|---|---|---|---|---|---|
| Multi-step | DDIM + P2P | 50+50 | 26s | 17.87 | 25.01 | Baseline chậm |
| Multi-step | NT-Inv + P2P | 50+opt | 134s | 27.03 | 24.75 | Chất lượng cao, rất chậm |
| Few-step | TurboEdit | 4+4 | 1.32s | 22.43 | 25.49 | Đối thủ tốc độ |
| Few-step | ICD (SD 1.5) | 3–4 | 1.62s | 26.93 | 22.42 | PSNR cao |
| **One-step** | **SwiftEdit** | **1+1** | **0.23s** | **23.33** | **25.16** | **Đề tài chính** |

---

## 10. Kết luận

> **Phần này sẽ được viết sau khi hoàn thành các thực nghiệm.**

### 10.1. Tóm tắt kết quả đạt được

- [ ] Đã tái hiện trên **Mac M4** (demo, ablation) và **Colab** (PieBench metrics)
- [ ] Metrics trên PieBench subset: …
- [ ] Phát hiện chính về hyperparameter / mask / so sánh baseline: …

### 10.2. Đánh giá mức độ đáp ứng mục tiêu

| Mục tiêu | Đạt / Chưa | Ghi chú |
|---|---|---|
| Hiểu và mô tả pipeline SwiftEdit | | |
| Chạy inference với checkpoint pretrained | | |
| Ablation hyperparameter ARaM | | |
| Đánh giá định lượng trên PieBench | | |
| So sánh với ít nhất 1 baseline | | |

### 10.3. Hạn chế

- Phụ thuộc chất lượng SBv2 và checkpoint pretrained.
- Prompt tiếng Anh; chưa tối ưu cho tiếng Việt.
- Repo gốc target CUDA; Mac cần port MPS — **Colab là môi trường chính cho benchmark**.
- Colab Free: session giới hạn, GPU không ổn định 100% thời gian.
- Style transfer phức tạp có thể kém hơn semantic editing.

### 10.4. Hướng phát triển

- Thử nghiệm trên one-step generator mới hơn (thay SBv2).
- Tích hợp prompt tiếng Việt (dịch hoặc multilingual CLIP).
- Tối ưu inference cho Apple Silicon (Core ML, quantization) — gần với hướng on-device của SwiftEdit.
- Kết hợp ControlNet / reference image cho style editing mạnh hơn.

---

## 11. Tài liệu tham khảo

1. Nguyen, T.-T., Nguyen, Q., Nguyen, K., Tran, A., & Pham, C. (2025). **SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion.** CVPR 2025. [Paper PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_SwiftEdit_Lightning_Fast_Text-Guided_Image_Editing_via_One-Step_Diffusion_CVPR_2025_paper.pdf)
2. Repository: [Qualcomm-AI-research/SwiftEdit](https://github.com/Qualcomm-AI-research/SwiftEdit)
3. Project page: [swift-edit.github.io](https://swift-edit.github.io/)
4. Dao, T., et al. (2025). **SwiftBrush v2.** ECCV 2025. (One-step T2I backbone)
5. Ju, X., et al. (2024). **PnP Inversion / PieBench.** ICLR 2024. (Benchmark)
6. Mokady, R., et al. (2023). **Null-text Inversion.** CVPR 2023.
7. Deutch, G., et al. (2024). **TurboEdit.** SIGGRAPH Asia 2024.
8. Ye, H., et al. (2023). **IP-Adapter.** (Image prompt adapter)

---

## Phụ lục A — Cấu trúc báo cáo đề xuất

```
1. Giới thiệu / Lý do chọn đề tài
2. Tổng quan SwiftEdit (Overview)
3. Cơ sở lý thuyết
4. Input / Output bài toán
5. Vấn đề nghiên cứu
6. Phương pháp và pipeline
7. Kế hoạch thực nghiệm
8. Quá trình nghiên cứu / Kết quả thực nghiệm
9. So sánh dataset / model
10. Kết luận và hướng phát triển
11. Tài liệu tham khảo
12. Phụ lục (ảnh kết quả, code snippet, log)
```

## Phụ lục B — Checklist nhanh trước khi nộp

- [ ] Mac: chạy được `infer.py` (MPS), có ≥5 ảnh ablation
- [ ] Colab: notebook tái sử dụng, weights trên Drive
- [ ] Colab: `metrics.csv` PieBench ≥50 mẫu
- [ ] So sánh runtime Mac / Colab / Paper (3 cột)
- [ ] Có so sánh baseline (Colab) HOẶC phân tích failure cases chi tiết
- [ ] Kết luận phản ánh đúng kết quả thực nghiệm

## Phụ lục C — Cấu trúc notebook Colab đề xuất

File: `CS2309_SwiftEdit.ipynb` — lưu trên Google Drive + Colab

| Cell | Nội dung |
|---|---|
| 1 | Mount Drive, set `WORK_DIR` |
| 2 | Clone SwiftEdit (skip nếu đã có trên Drive) |
| 3 | `pip install -r requirements.txt` |
| 4 | Symlink / tải weights từ Drive |
| 5 | Check CUDA + GPU name |
| 6 | **Demo:** 1 ảnh từ `assets/imgs_demo` |
| 7 | **Eval loop:** PieBench subset → save `metrics.csv` |
| 8 | **Baseline:** TurboEdit trên 20 mẫu (tùy chọn) |
| 9 | Download kết quả: zip `edited_images/` → Drive |
| 10 | (Tùy chọn) Fine-tune Stage 2 |

> Mỗi session Colab mới: chỉ cần chạy lại Cell 1, 4, 5 — bỏ qua tải weights.
