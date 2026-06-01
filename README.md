# CS2309.CH201 — SwiftEdit

**Chỉnh sửa ảnh theo văn bản một bước (One-step Text-guided Image Editing)**

| | |
|---|---|
| **Môn học** | CS2309.CH201 — Chuyên đề nghiên cứu và ứng dụng về Thị giác máy tính |
| **Chủ đề** | Chỉnh sửa và thay đổi phong cách ảnh |
| **Paper** | [SwiftEdit (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_SwiftEdit_Lightning_Fast_Text-Guided_Image_Editing_via_One-Step_Diffusion_CVPR_2025_paper.pdf) |
| **Repo gốc** | [Qualcomm-AI-research/SwiftEdit](https://github.com/Qualcomm-AI-research/SwiftEdit) |
| **Môi trường** | MacBook Air M4 24GB + Google Colab (T4) |

Tài liệu chi tiết: [`SwiftEdit_DeTai_CS2309.md`](./SwiftEdit_DeTai_CS2309.md)

---

## Tổng quan

**SwiftEdit** chỉnh sửa ảnh bằng prompt tiếng Anh trong **một bước diffusion** (~0.23s trên A100), nhanh hơn ít nhất **50×** so với multi-step (P2P, Null-text Inversion, …).

**Pipeline:**

```
Ảnh nguồn + Source/Edit prompt
    → One-step Inversion (Fθ)
    → Self-guided mask
    → ARaM editing
    → Ảnh đã chỉnh sửa
```

**Chiến lược thực nghiệm:**

| Mac M4 (local) | Google Colab (CUDA) |
|---|---|
| Demo, ablation, viết báo cáo | PieBench batch, so sánh baseline |
| Gradio demo | Lưu weights + metrics trên Drive |

---

## Cấu trúc repo

```
CS2309.CH201/
├── README.md                  ← File này (hướng dẫn + checklist)
├── NHAT_KY.md                 ← Nhật ký làm việc (tự cập nhật bởi skill)
├── SwiftEdit_DeTai_CS2309.md  ← Báo cáo / kế hoạch chi tiết
├── .cursor/skills/
│   └── update-readme-progress/  ← Skill: cập nhật README + NHAT_KY
└── (sau này)
    ├── SwiftEdit/             ← Clone từ GitHub
    ├── notebooks/
    │   └── CS2309_SwiftEdit.ipynb
    ├── results/
    │   ├── ablation/
    │   ├── piebench/
    │   └── custom_vn/
    └── report/                ← Slide, ảnh báo cáo
```

---

## Cài đặt nhanh

### Mac (MPS)

```bash
conda create -n SwiftEdit python=3.12 -y
conda activate SwiftEdit

pip install torch torchvision torchaudio
pip install transformers==4.37.2 accelerate ftfy tensorboard Jinja2 \
            diffusers==0.22.0 huggingface-hub==0.25.2 einops
pip install numpy==1.26.4

git clone https://github.com/Qualcomm-AI-research/SwiftEdit.git
cd SwiftEdit
# Tải weights → swiftedit_weights/ (GitHub Releases v1.0)
```

Sửa `device = "mps"` trong `infer.py` / `models.py`.

### Google Colab

1. Runtime → **T4 GPU**
2. Mount Drive → `MyDrive/CS2309_SwiftEdit/`
3. `pip install -r requirements.txt` (CUDA — dùng file gốc)
4. Tải weights 1 lần, lưu Drive, session sau symlink

Chi tiết: [Mục 4 trong tài liệu đề tài](./SwiftEdit_DeTai_CS2309.md#4-môi-trường-thực-nghiệm-mac-m4--google-colab)

### Tải checkpoint

```bash
# Trong thư mục SwiftEdit (Mac hoặc Colab)
wget https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-aa
wget https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ab
wget https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ac
wget https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ad
wget https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ae
cat swiftedit_weights.tar.gz.part-* > swiftedit_weights.tar.gz
tar zxf swiftedit_weights.tar.gz
```

---

## Checklist tổng thể

> Đánh dấu `[x]` khi hoàn thành. Cập nhật file này trong quá trình làm đề tài.
> Cập nhật tiến độ lần cuối: 2026-06-01 — 0/64 task bắt buộc

### Trạng thái nhanh

| Giai đoạn | Tiến độ |
|---|---|
| 1. Lý thuyết | ⬜ Chưa bắt đầu |
| 2. Setup Mac + Colab | ⬜ Chưa bắt đầu |
| 3. Thực nghiệm cơ bản | ⬜ Chưa bắt đầu |
| 4. So sánh & mở rộng | ⬜ Chưa bắt đầu |
| 5. Báo cáo & nộp | ⬜ Chưa bắt đầu |




---

## Giai đoạn 1 — Lý thuyết (Tuần 1–2)

- [ ] Đọc paper SwiftEdit (Abstract, Sec. 3–5)
- [ ] Tìm hiểu SwiftBrushv2, DDIM inversion, IP-Adapter
- [ ] Tìm hiểu PieBench benchmark
- [ ] Vẽ/tóm tắt pipeline inversion → mask → ARaM
- [ ] So sánh SwiftEdit vs P2P, NT-Inv, TurboEdit, ICD (bảng related work)
- [ ] Viết phần Overview + Related Work trong báo cáo

---

## Giai đoạn 2 — Setup môi trường (Tuần 2–3)

### 2a. MacBook Air M4

- [ ] Tạo conda env `SwiftEdit` (Python 3.12)
- [ ] Cài PyTorch Mac (MPS) — **không** dùng bản CUDA
- [ ] Cài dependencies còn lại
- [ ] Clone repo SwiftEdit
- [ ] Tải checkpoint → `swiftedit_weights/`
- [ ] Sửa `device = "mps"` trong code
- [ ] Chạy demo `assets/imgs_demo` thành công
- [ ] Ghi thời gian/ảnh và RAM peak (Activity Monitor)
- [ ] Đọc và ghi chú `infer.py`, `models.py`

### 2b. Google Colab

- [ ] Tạo notebook `CS2309_SwiftEdit.ipynb`
- [ ] Tạo folder `MyDrive/CS2309_SwiftEdit/` trên Drive
- [ ] Mount Drive + clone repo
- [ ] Cài `requirements.txt` (CUDA)
- [ ] Tải weights lên Drive (chỉ lần đầu)
- [ ] Chạy demo CUDA — kết quả khớp Mac
- [ ] Ghi GPU name (T4/A100) và runtime Colab
- [ ] Test symlink weights khi mở session mới

---

## Giai đoạn 3 — Thực nghiệm cơ bản (Tuần 3–5)

### 3a. Ablation hyperparameter — Mac

- [ ] Chọn 5–10 ảnh đại diện (đổi màu, đối tượng, chi tiết)
- [ ] Thử `s_y` ∈ {0.5, 1.0, 1.5}
- [ ] Thử `s_edit` ∈ {0.5, 1.0, 1.5}
- [ ] Thử `s_non-edit` ∈ {0.5, 1.0, 1.5}
- [ ] Lưu grid ảnh → `results/ablation/`
- [ ] Phân tích trade-off editing vs background preservation

### 3b. Self-guided mask — Colab

- [ ] Tải subset PieBench có GT mask
- [ ] Chạy SwiftEdit với mask tự sinh
- [ ] Chạy SwiftEdit với GT mask
- [ ] Tính IoU / Dice (mask tự sinh vs GT)
- [ ] Download ảnh visualization mask → Mac

### 3c. Đánh giá PieBench — Colab (+ Mac xác nhận)

- [ ] Colab: chạy **50–100 mẫu** PieBench
- [ ] Tính PSNR, MSE (background)
- [ ] Tính CLIP-Whole, CLIP-Edited
- [ ] Tính runtime trung bình / ảnh
- [ ] Lưu `results/piebench/metrics.csv`
- [ ] Lưu ảnh edited → `results/piebench/edited_images/`
- [ ] Mac: chạy **10–15 mẫu** trùng subset — xác nhận metrics gần khớp Colab
- [ ] So sánh với Table 1 paper (reference A100)

---

## Giai đoạn 4 — So sánh & mở rộng (Tuần 5–7)

### 4a. So sánh baseline — Colab

- [ ] Chọn 20 mẫu chung (PieBench)
- [ ] Chạy SwiftEdit trên 20 mẫu
- [ ] Chạy TurboEdit (hoặc 1 multi-step method) trên 20 mẫu
- [ ] Lập bảng metrics + runtime
- [ ] Mac: side-by-side 5 mẫu để minh họa báo cáo
- [ ] Bảng runtime 3 cột: **Mac MPS | Colab T4 | Paper A100**

### 4b. Bộ ảnh tự thu thập (tùy chọn, khuyến nghị)

- [ ] Thu 20–30 cặp (ảnh + source/edit prompt) bối cảnh VN
- [ ] Upload lên Colab Drive `custom_vn/`
- [ ] Chạy batch inference
- [ ] Phân loại: thành công / một phần / thất bại
- [ ] Ghi nhận 3–5 failure cases

### 4c. Style editing (tùy chọn)

- [ ] Thử prompt: watercolor, anime, oil painting, …
- [ ] Đánh giá định tính khả năng chuyển phong cách
- [ ] So sánh với semantic editing (đổi đối tượng/màu)

### 4d. Demo Gradio (tùy chọn)

- [ ] Gradio: upload ảnh + nhập prompt
- [ ] Hiển thị ảnh output + thời gian suy luận
- [ ] Chạy local trên Mac

### 4e. Fine-tune nhẹ Colab (tùy chọn — bỏ qua vẫn đủ)

- [ ] Chuẩn bị 200–500 ảnh + caption
- [ ] Stage 2 vài nghìn iterations
- [ ] So sánh reconstruction trước/sau

---

## Giai đoạn 5 — Báo cáo & nộp (Tuần 7–8)

### Viết báo cáo

- [ ] Lý do chọn đề tài
- [ ] Overview + pipeline SwiftEdit
- [ ] Input / Output bài toán
- [ ] Vấn đề nghiên cứu + Research Questions
- [ ] Kế hoạch thực nghiệm (đã thực hiện)
- [ ] **Quá trình nghiên cứu / kết quả thực nghiệm** (điền số liệu thật)
- [ ] So sánh dataset / model
- [ ] **Kết luận** (viết sau khi có kết quả)
- [ ] Tài liệu tham khảo

### Checklist nộp

- [ ] Mac: ≥5 ví dụ chỉnh sửa (input → output)
- [ ] Colab: `metrics.csv` PieBench ≥50 mẫu
- [ ] Có ablation ≥1 hyperparameter (grid ảnh)
- [ ] Có so sánh baseline (Colab) **hoặc** phân tích failure cases chi tiết
- [ ] Bảng runtime: Mac / Colab / Paper
- [ ] Slide trình bày (pipeline + kết quả + demo)
- [ ] Kết luận phản ánh đúng kết quả thực tế

---

## Checklist theo môi trường

### Chỉ Mac

- [ ] Env MPS hoạt động
- [ ] Demo `infer.py` OK
- [ ] Ablation hyperparameter xong
- [ ] Runtime + RAM đã ghi
- [ ] (Tùy chọn) Gradio demo

### Chỉ Colab

- [ ] Notebook tái sử dụng được (mount + symlink)
- [ ] Weights trên Drive, không tải lại mỗi session
- [ ] PieBench eval xong
- [ ] Baseline comparison xong (nếu có)
- [ ] `metrics.csv` + ảnh đã sync về Mac/Drive

---

## Xử lý sự cố thường gặp

| Vấn đề | Cách xử lý |
|---|---|
| Mac: lỗi MPS operator | `export PYTORCH_ENABLE_MPS_FALLBACK=1` hoặc chạy mẫu đó trên Colab |
| Mac: OOM / chậm | `float16`, giữ 512×512, đóng app nặng |
| Colab: disconnect | Chia batch 20 mẫu/session; lưu progress trên Drive |
| Colab: OOM T4 | `float16`, batch=1 |
| Metrics Mac ≠ Colab | Kiểm tra cùng seed, dtype, hyperparameter |
| Không tải được weights | Dùng `wget` từng part; kiểm tra dung lượng Drive |

---

## Tài liệu tham khảo

1. [SwiftEdit Paper (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_SwiftEdit_Lightning_Fast_Text-Guided_Image_Editing_via_One-Step_Diffusion_CVPR_2025_paper.pdf)
2. [GitHub — Qualcomm-AI-research/SwiftEdit](https://github.com/Qualcomm-AI-research/SwiftEdit)
3. [Project page](https://swift-edit.github.io/)
4. [PieBench](https://github.com/Prompt-to-Prompt/PieBench)
5. [TurboEdit](https://github.com/GaMaLielD/TurboEdit) *(baseline, nếu so sánh)*

---

## Ghi chú

- **Không training nặng** — dùng checkpoint pretrained; fine-tune chỉ trên Colab nếu cần.
- Ghi rõ trong báo cáo metric nào đo trên **Mac**, **Colab**, hay lấy từ **paper**.
- Cập nhật checklist và bảng "Trạng thái nhanh" khi hoàn thành từng giai đoạn.
- Skill Cursor: `.cursor/skills/update-readme-progress/` — cập nhật **README checklist** + **NHAT_KY.md** (nhật ký làm việc).
- Chạy thủ công:
  ```bash
  # Sync tiến độ README
  python .cursor/skills/update-readme-progress/scripts/sync_progress.py

  # Đánh task + ghi nhật ký + sync báo cáo §8.1
  python .cursor/skills/update-readme-progress/scripts/sync_progress.py \
    --mark "Clone repo SwiftEdit" \
    --journal-phase "2a. Mac" \
    --journal-work "Mô tả công việc" \
    --journal-result "Kết quả cụ thể" \
    --journal-env "Mac M4 (MPS)"
  ```
