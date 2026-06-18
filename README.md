# CS2309.CH201 — SwiftEdit

**Chỉnh sửa ảnh theo văn bản một bước (One-step Text-guided Image Editing)**

| | |
|---|---|
| **Môn học** | CS2309.CH201 — Chuyên đề nghiên cứu và ứng dụng về Thị giác máy tính |
| **Chủ đề** | Chỉnh sửa và thay đổi phong cách ảnh |
| **Paper** | [SwiftEdit (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_SwiftEdit_Lightning_Fast_Text-Guided_Image_Editing_via_One-Step_Diffusion_CVPR_2025_paper.pdf) |
| **Repo gốc** | [Qualcomm-AI-research/SwiftEdit](https://github.com/Qualcomm-AI-research/SwiftEdit) |
| **Môi trường** | MacBook Air M4 24GB + Google Colab (T4) |

Tài liệu: [`SwiftEdit_Overview.md`](./SwiftEdit_Overview.md) · [`SwiftEdit_DeTai_CS2309.md`](./SwiftEdit_DeTai_CS2309.md) · [`QA.md`](./QA.md) · [`NHAT_KY.md`](./NHAT_KY.md) · [`HUONG_DAN_PUBLICATION_THAC_SI.md`](./HUONG_DAN_PUBLICATION_THAC_SI.md) *(cân nhắc paper / tốt nghiệp thạc sĩ UIT)*

### Mở notebook trên Google Colab

| Notebook | Mở |
|----------|-----|
| Benchmark tốc độ & chất lượng (fp32/fp16/fp8/fp4) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NguyenKz/CS2309.CH201/blob/main/notebooks/CS2309_SwiftEdit_quality_speed_bench.ipynb) |
| Thực nghiệm giai đoạn 3 (ablation + PIE-Bench) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NguyenKz/CS2309.CH201/blob/main/notebooks/CS2309_SwiftEdit_phase3.ipynb) |
| Test nhanh pipeline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NguyenKz/CS2309.CH201/blob/main/notebooks/CS2309_SwiftEdit_test.ipynb) |

> Mở badge → Colab → Runtime **T4 GPU** → chạy tuần tự. Repo private cần thêm `GITHUB_TOKEN` trong Colab Secrets (xem cell setup).

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

**Đánh giá chính:** dùng PIE-Bench/PnP Inversion với PSNR/MSE trên vùng background `(1 - mask)`, CLIP-Whole/CLIP-Edited theo `edit_prompt`, và runtime mỗi lần gọi `edit_image()`. Chi tiết xem [Đề tài §9.3](./SwiftEdit_DeTai_CS2309.md#93-metrics-đánh-giá) và [Q&A §5](./QA.md#5-đánh-giá--piebench).

**Hướng mở rộng được chọn:** **SwiftEdit-RT: Realtime-Oriented Inference Acceleration** — giữ tinh thần realtime của SwiftEdit bằng cách profile bottleneck, bỏ overhead không cần thiết, cache embedding/latent cho demo tương tác và thử các tối ưu inference như `fp16`, `channels_last`, `torch.compile` hoặc TinyVAE/TAESD. SAM 3 được giữ như hướng không chọn/optional vì thêm segmentation model làm tăng latency end-to-end. Chi tiết xem [Đề tài §5.4](./SwiftEdit_DeTai_CS2309.md#54-hướng-chọn-đào-sâu-swiftedit-rt) và [§7.5.4d](./SwiftEdit_DeTai_CS2309.md#4d-hướng-đào-sâu--swiftedit-rt-realtime-inference-acceleration).

---

## Cấu trúc repo

```
CS2309.CH201/
├── README.md                  ← Hướng dẫn + checklist
├── .python-version            ← pyenv: 3.12.10
├── requirements-mac.txt       ← pip cho Mac MPS
├── .venv/                     ← virtualenv (tạo local, không commit)
├── scripts/
│   ├── download_swiftedit_weights.sh
│   ├── download_hf_models.sh
│   └── run_swiftedit.sh
├── SwiftEdit/                 ← Repo gốc + patch MPS (infer.py, models.py)
│   ├── infer.py
│   └── swiftedit_weights/     ← Checkpoint Qualcomm (~9.6 GB)
├── assets/pipeline/
├── NHAT_KY.md · QA.md · SwiftEdit_DeTai_CS2309.md
├── notebooks/
│   ├── CS2309_SwiftEdit_test.ipynb
│   ├── CS2309_SwiftEdit_phase3.ipynb
│   └── CS2309_SwiftEdit_quality_speed_bench.ipynb
├── experimental_data/         ← Bằng chứng thực nghiệm (metrics, ảnh mẫu)
└── (sau này) results/, report/
```

---

## Cài đặt và chạy cơ bản (Mac MPS)

Yêu cầu: **macOS**, **pyenv** + Python **3.12**, **~25 GB** disk (weights + cache HF), RAM khuyến nghị **24 GB**.

### 1. Clone repo đề tài (nếu chưa có)

```bash
git clone <url-repo-CS2309.CH201> CS2309.CH201
cd CS2309.CH201
```

Repo đã có sẵn thư mục `SwiftEdit/` (fork/clone từ [Qualcomm SwiftEdit](https://github.com/Qualcomm-AI-research/SwiftEdit)) kèm patch cho Mac.

### 2. Môi trường Python (pyenv + venv)

```bash
# Cài Python 3.12 qua pyenv (một lần)
pyenv install 3.12.10   # bỏ qua nếu đã có
pyenv local 3.12.10     # đọc từ .python-version

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-mac.txt
```

Kiểm tra MPS:

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

*(Có thể dùng conda thay venv — xem [Mục 4.1 đề tài](./SwiftEdit_DeTai_CS2309.md#41-macbook-air-m4-24gb--môi-trường-local).)*

### 3. Tải checkpoint SwiftEdit (~9.6 GB)

```bash
bash scripts/download_swiftedit_weights.sh
```

Script tải 5 file `part-aa` … `part-ae` từ GitHub Releases, ghép và giải nén vào `SwiftEdit/swiftedit_weights/`.

Tải tay (nếu script lỗi): xem [releases v1.0](https://github.com/Qualcomm-AI-research/SwiftEdit/releases/tag/v1.0).

### 4. Tải model Hugging Face (lần đầu)

SwiftEdit cần thêm **sd-turbo**, **SD 2.1** (text encoder/VAE cho edit), **IP-Adapter image_encoder**. Lần chạy `infer.py` cũng có thể tự tải, nhưng nên prefetch:

```bash
source .venv/bin/activate
bash scripts/download_hf_models.sh
```

**Lưu ý:** Repo `stabilityai/stable-diffusion-2-1-base` trên HF có thể trả **401**; code trong `SwiftEdit/models.py` đã trỏ mirror `Manojb/stable-diffusion-2-1-base` (cùng weights).

Nếu timeout mạng:

```bash
export HF_HUB_DOWNLOAD_TIMEOUT=900
bash scripts/download_hf_models.sh
```

### 5. Chạy demo

**CLI:**

```bash
source .venv/bin/activate
bash scripts/run_swiftedit.sh
```

Hoặc `cd SwiftEdit && python infer.py`.

**Notebook (test tương tác):**

```bash
source .venv/bin/activate
pip install jupyter ipywidgets matplotlib   # hoặc đã có trong requirements-mac.txt
jupyter notebook notebooks/CS2309_SwiftEdit_test.ipynb
```

Chọn kernel `.venv`, chạy lần lượt các cell. Kết quả lưu `results/notebook/`. Trên Colab: đặt `IN_COLAB = True` ở cell 2.

Ảnh mặc định: `SwiftEdit/assets/imgs_demo/woman_face.jpg` — prompt `woman` → `Taylor Swift`. Kết quả: `SwiftEdit/result_woman->Taylor Swift.png` (trên Mac M4 thường **~90 giây/ảnh** lần đầu, gồm load model).

### 6. Đổi ảnh / prompt

Sửa cuối file `SwiftEdit/infer.py`:

```python
img_path = "./assets/imgs_demo/woman_face.jpg"
src_p = "woman"          # mô tả ảnh gốc (có thể để ngắn hoặc "")
edit_p = "Taylor Swift"  # bắt buộc — mô tả chỉnh sửa
```

`src_p` giúp mask/inversion; có thể bỏ trống nhưng kém ổn định (xem `QA.md`).

### Google Colab (tóm tắt)

**Cách nhanh:** mở [`notebooks/CS2309_SwiftEdit_test.ipynb`](./notebooks/CS2309_SwiftEdit_test.ipynb) trên Colab → Runtime **T4 GPU** → chạy all cells.

- **Chỉ clone repo đề tài** `CS2309.CH201` (đã có `SwiftEdit/` + patch) — không clone Qualcomm riêng
- Tải weights + HF vào **`/content`** (ổ Colab, **không** bắt buộc Google Drive; mất khi reset runtime)
- Đổi `REPO_URL` trong notebook nếu dùng fork

**Lưu weights lâu dài (tùy chọn):** copy `swiftedit_weights/` lên Drive thủ công; notebook mặc định không mount Drive.

Chi tiết: [Mục 4.2 đề tài](./SwiftEdit_DeTai_CS2309.md#42-google-colab--môi-trường-gpu-cuda).

---

## Checklist tổng thể

> Đánh dấu `[x]` khi hoàn thành. Cập nhật file này trong quá trình làm đề tài.
> Cập nhật tiến độ lần cuối: 2026-06-18 — 33/80 task bắt buộc

### Trạng thái nhanh

| Giai đoạn | Tiến độ |
|---|---|
| 1. Lý thuyết | 🔄 Đang làm (1/6) |
| 2. Setup Mac + Colab | 🔄 Đang làm (11/17) |
| 3. Thực nghiệm cơ bản | 🔄 Đang làm (6/19) |
| 4. So sánh & mở rộng | 🔄 Đang làm (10/19) |
| 5. Báo cáo & nộp | 🔄 Đang làm (5/19) |


**Tổng:** 33/80 task bắt buộc (~41%) — phần thực nghiệm SwiftEdit-RT đã có số liệu lớn; báo cáo và ablation còn thiếu.

### Tiến trình hiện tại (tổng hợp 2026-06-18)

**Điểm mạnh (đã có bằng chứng):**

| Hạng mục | Chi tiết |
|---|---|
| Hạ tầng | Mac M4 MPS + Colab T4 chạy end-to-end; 3 notebook + scripts eval |
| **SwiftEdit-RT** (hướng đào sâu) | fp16 + cache + channels_last; benchmark **2400 edit** Colab (fp32/fp16/fp8/fp4) |
| Benchmark Colab T4 | fp16+cache **khuyến nghị**: 1.70×, VRAM −42%, PSNR 48.6 dB — [`quality_speed_bench_2026-06-17/`](./experimental_data/quality_speed_bench_2026-06-17/) |
| PIE-Bench Mac | 20 mẫu (2/loại × 10 loại); CLIP-Whole 23.02 — [`piebench_subset20_2026-06-14/`](./experimental_data/piebench_subset20_2026-06-14/) |
| Demo | Gradio fp16+cache + tab xóa vật thể — [`app_gradio.py`](./scripts/app_gradio.py) |
| Runtime 3 cột | Mac MPS ~30s / Colab T4 ~1.3–2.9s / Paper A100 ~0.23s |

**Khoảng trống (cần cho nộp báo cáo):**

| Hạng mục | Trạng thái | Mức ưu tiên |
|---|---|---|
| Viết báo cáo GĐ5 | Chưa bắt đầu | **Cao** |
| Ablation hyperparameter (`s_y`, `s_edit`, `s_non-edit`) | Chưa | **Cao** (checklist nộp) |
| PIE-Bench 50–100 mẫu metrics Colab | Chỉ 20 mẫu Mac | Trung bình |
| So sánh baseline (TurboEdit) | Chưa | Trung bình |
| Slide trình bày | Chưa | **Cao** |
| Lý thuyết (đọc paper, Related Work) | 1/6 | Trung bình |

**Checklist nộp — đã đạt / chưa:**

- [x] Mac ≥5 ví dụ · Colab ≥50 mẫu (2400 edit benchmark) · SwiftEdit-RT · object removal · bảng runtime
- [ ] Ablation ≥1 hyperparameter · baseline **hoặc** failure cases chi tiết · slide · kết luận · báo cáo đầy đủ

### Việc tiếp theo (gợi ý ưu tiên)

| Ưu tiên | Việc | Thời gian ước tính |
|---|---|---|
| **1** | Bắt đầu viết báo cáo GĐ5 (dùng số liệu SwiftEdit-RT + object removal) | 2–4 ngày |
| **2** | Ablation hyperparameter Mac (5–10 ảnh, grid `s_edit`/`s_non-edit`) | 0.5–1 ngày |
| **3** | Slide: pipeline + before/after + bảng fp16 vs fp32 + demo Gradio | 0.5–1 ngày |
| **4** | PIE-Bench 50 mẫu Colab (`run_piebench_eval.py` + fp16+cache) | 1 ngày |
| **5** | Object removal Hướng A — test 2–3 ảnh phù hợp ([`KE_HOACH_KIEM_TRA.md`](./experimental_data/object_removal_2026-06-14/KE_HOACH_KIEM_TRA.md)) | 0.5 ngày |
| **6** | TurboEdit baseline 20 mẫu **hoặc** `torch.compile` Colab | Tùy thời gian |

> **Khuyến nghị:** Nếu deadline gần → ưu tiên **báo cáo + ablation + slide** (đã có xương sống thực nghiệm). Nếu còn 1–2 tuần → thêm PIE-Bench 50 mẫu Colab.

---

## Giai đoạn 1 — Lý thuyết (Tuần 1–2)

> **Mục tiêu:** Hiểu SwiftEdit đủ sâu để viết báo cáo §Overview + Related Work + pipeline.  
> **Tài liệu sẵn có:** [`SwiftEdit_Overview.md`](./SwiftEdit_Overview.md) · [`SwiftEdit_DeTai_CS2309.md` §1](./SwiftEdit_DeTai_CS2309.md#1-overview) · [`QA.md`](./QA.md) · ảnh pipeline cũ [`assets/pipeline/`](./assets/pipeline/)

### Checklist tổng (6 mục)

- [ ] Đọc paper SwiftEdit (Abstract, Sec. 3–5)
- [ ] Tìm hiểu SwiftBrushv2, DDIM inversion, IP-Adapter
- [x] Tìm hiểu PieBench benchmark
- [ ] Vẽ/tóm tắt pipeline inversion → mask → ARaM
- [ ] So sánh SwiftEdit vs P2P, NT-Inv, TurboEdit, ICD (bảng related work)
- [ ] Viết phần Overview + Related Work trong báo cáo

### Chi tiết từng task (làm tuần tự)

#### Task 1 — Đọc paper SwiftEdit

| # | Việc | Đọc | Deliverable |
|---|---|---|---|
| 1.1 | Abstract + Introduction | Motivation, 2 đóng góp chính, so với multi-step | 5–10 bullet ghi chú |
| 1.2 | **Sec. 3** — One-step Inversion | `F_theta`, inverted noise, Stage 1 (synthetic) + Stage 2 (real/DISTS) | Tóm tắt 1 đoạn: *tại sao DDIM không dùng được* |
| 1.3 | **Sec. 4** — ARaM + mask | Self-guided mask, `s_y` / `s_edit` / `s_non-edit`, IP-Adapter branch | Liên hệ `infer.py` + `mask_ip_controller.py` |
| 1.4 | **Sec. 5** — Experiments | Table 1 (PieBench), ablation trong paper, Fig. qualitative | Chép/summarize Table 1 vào ghi chú |

**Paper:** [CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_SwiftEdit_Lightning_Fast_Text-Guided_Image_Editing_via_One-Step_Diffusion_CVPR_2025_paper.pdf)

#### Task 2 — Nền tảng kỹ thuật (SwiftBrushv2, DDIM, IP-Adapter)

| # | Chủ đề | Cần nắm | Nguồn gợi ý |
|---|---|---|---|
| 2.1 | **Diffusion one-step / SBv2** | SBv2 = backbone one-step T2I; distilled từ SD-Turbo; SwiftEdit build trên SBv2 | Paper SwiftEdit + [Overview §3](./SwiftEdit_Overview.md#3-ý-tưởng-cốt-lõi) |
| 2.2 | **DDIM inversion** | Multi-step đảo ngược ảnh → noise; mỗi step = 1 lần UNet | [Đề tài §1.3.1](./SwiftEdit_DeTai_CS2309.md#131-mỗi-step-làm-gì-giải-thích-dễ-hiểu) · [QA §1](./QA.md#1-diffusion--text-to-image) |
| 2.3 | **Null-text Inversion** | Optimization từng timestep; chậm nhưng PSNR cao — bối cảnh so sánh | `assets/pipeline/nulltext-diagram.png` |
| 2.4 | **IP-Adapter** | Image condition qua decoupled cross-attention; vai trò `G_IP` trong SwiftEdit | Paper IP-Adapter · code `attention_processor.py` |

**Deliverable:** Thêm 2–4 câu Q&A vào [`QA.md` §3 Inversion & Noise](./QA.md#3-inversion--noise) (hiện còn trống).

#### Task 3 — PieBench *(đã [x], ôn lại nhanh)*

| # | Việc | Deliverable |
|---|---|---|
| 3.1 | 10 loại editing + GT mask | 1 bảng tóm tắt (copy từ [Đề tài §9.2](./SwiftEdit_DeTai_CS2309.md#92-phân-loại-10-loại-editing-trong-piebench)) |
| 3.2 | Bộ metric chuẩn | PSNR/MSE nền, CLIP-Whole/Edited, runtime — [QA §5](./QA.md#5-đánh-giá--piebench) |
| 3.3 | Liên hệ thực nghiệm đề tài | Ghi 2–3 câu: subset 20 mẫu Mac + benchmark 2400 edit Colab |

#### Task 4 — Pipeline SwiftEdit (sơ đồ + map code)

| # | Việc | Deliverable |
|---|---|---|
| 4.1 | Vẽ sơ đồ **one-step** (khác pipeline cũ multi-step) | ASCII/Mermaid hoặc figure cho slide + báo cáo |
| 4.2 | So sánh trực quan pipeline **cũ vs SwiftEdit** | Dùng ảnh `assets/pipeline/` (cũ) + sơ đồ §1.2 đề tài (mới) |
| 4.3 | Map từng bước → file code | Bảng: VAE encode → `F_theta` → mask → ARaM → VAE decode → `infer.py` / `models.py` |
| 4.4 | Giải thích hyperparameter ARaM | `scale_edit`, `scale_non_edit`, `mask_threshold` — khi nào tăng/giảm |

**Tham khảo sẵn:** [Overview §4](./SwiftEdit_Overview.md#4-pipeline-suy-luận) · [Đề tài §1.2](./SwiftEdit_DeTai_CS2309.md#12-kiến-trúc-và-pipeline-suy-luận)

#### Task 5 — Bảng Related Work

| # | Việc | Deliverable |
|---|---|---|
| 5.1 | Nhóm **multi-step** | P2P, Null-text+P2P, MasaCtrl, Plug-and-Play — steps, runtime, ưu/nhược |
| 5.2 | Nhóm **few-step** | TurboEdit, ICD, ReNoise — steps, runtime |
| 5.3 | Nhóm **one-step** | SwiftEdit — điểm khác biệt |
| 5.4 | Bảng tổng hợp + số liệu Table 1 | Copy/adapt từ [Overview §6](./SwiftEdit_Overview.md#6-so-sánh-với-các-hướng-tiếp-cận) |

**Cột gợi ý:** Method · Inversion steps · Edit steps · Runtime · PSNR · CLIP-Whole · Đặc điểm chính

#### Task 6 — Viết báo cáo (draft lý thuyết)

| # | Mục báo cáo | Nội dung lấy từ | Ghi chú |
|---|---|---|---|
| 6.1 | Lý do chọn đề tài | Motivation §1.3 + realtime + đã có SwiftEdit-RT | 0.5–1 trang |
| 6.2 | Bài toán Input/Output | [Overview §2, §7](./SwiftEdit_Overview.md#2-bài-toán-và-động-lực) | Định nghĩa `x_source`, `y_source`, `y_edit` |
| 6.3 | Overview + pipeline SwiftEdit | Task 4 | Figure + mô tả 4 bước |
| 6.4 | Related Work | Task 5 | Bảng so sánh + 1–2 đoạn nhận xét |
| 6.5 | Research Questions | [Đề tài §5.7](./SwiftEdit_DeTai_CS2309.md#57-câu-hỏi-nghiên-cứu-đề-xuất-research-questions) | Chọn **RQ1–RQ6** (bắt buộc) + **RQ8–RQ11** (SwiftEdit-RT, đã có số liệu) |
| 6.6 | Tài liệu tham khảo | Paper SwiftEdit, PIE-Bench, CLIP, IP-Adapter, TurboEdit | Format chuẩn UIT |

**Nơi viết:** draft trong `SwiftEdit_DeTai_CS2309.md` §1–2 hoặc file báo cáo riêng (nếu có template UIT).

### Thứ tự làm đề xuất

```
1.1–1.4 (đọc paper) → 2.1–2.4 (nền tảng) → 4.1–4.4 (pipeline)
→ 5.1–5.4 (related work) → 3.1–3.3 (ôn PieBench) → 6.1–6.6 (viết báo cáo)
```

Sau mỗi task xong, báo agent *"xong task X.Y"* để đánh `[x]` README + ghi `NHAT_KY.md`.

---

## Giai đoạn 2 — Setup môi trường (Tuần 2–3)

### 2a. MacBook Air M4

- [ ] Tạo conda env `SwiftEdit` (Python 3.12)
- [x] Cài PyTorch Mac (MPS) — **không** dùng bản CUDA
- [x] Cài dependencies còn lại
- [x] Clone repo SwiftEdit
- [x] Tải checkpoint → `swiftedit_weights/`
- [x] Sửa `device = "mps"` trong code
- [x] Chạy demo `assets/imgs_demo` thành công
- [x] Ghi thời gian/ảnh và RAM peak (Activity Monitor)
- [ ] Đọc và ghi chú `infer.py`, `models.py`

### 2b. Google Colab

- [x] Tạo notebook `CS2309_SwiftEdit.ipynb`
- [ ] Tạo folder `MyDrive/CS2309_SwiftEdit/` trên Drive
- [ ] Mount Drive + clone repo
- [x] Cài `requirements.txt` (CUDA)
- [ ] Tải weights lên Drive (chỉ lần đầu)
- [x] Chạy demo CUDA — kết quả khớp Mac
- [x] Ghi GPU name (T4/A100) và runtime Colab
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
- [x] Tính PSNR, MSE (background)
- [x] Tính CLIP-Whole, CLIP-Edited với `edit_prompt`
- [x] Tính runtime trung bình / ảnh
- [x] Lưu `results/piebench/metrics.csv`
- [x] Lưu ảnh edited → `experimental_data/piebench_subset20_2026-06-14/edited_images/`
- [x] Mac: chạy **20 mẫu** subset (2/loại × 10 loại) — TB 69.0s/ảnh; CLIP-Whole 23.02
- [ ] So sánh với Table 1 paper (reference A100)

> **Dữ liệu thực nghiệm 2026-06-14:** [`experimental_data/piebench_subset20_2026-06-14/`](./experimental_data/piebench_subset20_2026-06-14/) — 20 ảnh kết quả, `metrics.csv` (runtime + timing + chất lượng), báo cáo timing từng công đoạn (`timing_report.md`).
>
> **Cache embedding 2026-06-14:** [`experimental_data/cache_benchmark_2026-06-14/`](./experimental_data/cache_benchmark_2026-06-14/) — benchmark cache latent + CLIP image embed + source prompt embed: tiết kiệm ~9.93s/edit khi cùng ảnh + source prompt (`bench_cache.py`).
>
> **fp16 / channels_last 2026-06-14:** [`experimental_data/fp16_benchmark_2026-06-14/`](./experimental_data/fp16_benchmark_2026-06-14/) — fp16 tăng tốc ~3.3×–7× trên Mac M4/MPS, PSNR ~45dB vs fp32, không NaN/đen (`bench_dtype.py`).
>
> **Demo Gradio 2026-06-14:** [`experimental_data/gradio_demo_2026-06-14/`](./experimental_data/gradio_demo_2026-06-14/) — UI web chỉnh sửa ảnh bằng prompt, tích hợp fp16 + channels_last + cache (`scripts/app_gradio.py`).
>
> **Xóa vật thể (khoanh vùng) 2026-06-14:** [`experimental_data/object_removal_2026-06-14/`](./experimental_data/object_removal_2026-06-14/) — vẽ mask để xóa vật thể (`user_mask` + tab "Xóa vật thể"); xóa OK vật nhỏ/vừa, vật rất lớn còn sót. **Kế hoạch kiểm tra tiếp:** [`KE_HOACH_KIEM_TRA.md`](./experimental_data/object_removal_2026-06-14/KE_HOACH_KIEM_TRA.md).
>
> **Tốc độ & chất lượng quy mô lớn 2026-06-17 (Colab T4):** [`experimental_data/quality_speed_bench_2026-06-17/`](./experimental_data/quality_speed_bench_2026-06-17/) — **200 ảnh × 3 prompt × 4 config** (fp32 / fp16 / fp8 / fp4): **fp16+cache khuyến nghị** — nhanh **1.70×–1.82×**, VRAM **−42.1%**, PSNR **48.6dB**; **fp8 nhanh 1.92× nhưng PSNR 6.0dB (hỏng)**; **fp4 VRAM −48.5% nhưng PSNR 21.7dB** (notebook `CS2309_SwiftEdit_quality_speed_bench.ipynb`). Bản cũ fp32 vs fp16: `quality_speed_bench_2026-06-14` (nếu còn trong repo).

---

## Giai đoạn 4 — So sánh & mở rộng (Tuần 5–7)

### 4a. So sánh baseline — Colab

- [x] Chọn 20 mẫu chung (PieBench)
- [x] Chạy SwiftEdit trên 20 mẫu
- [ ] Chạy TurboEdit (hoặc 1 multi-step method) trên 20 mẫu
- [ ] Lập bảng metrics + runtime
- [ ] Mac: side-by-side 5 mẫu để minh họa báo cáo
- [x] Bảng runtime 3 cột: **Mac MPS | Colab T4 | Paper A100**

### 4b. Bộ ảnh tự thu thập (tùy chọn, khuyến nghị)

- [ ] Thu 20–30 cặp (ảnh + source/edit prompt) bối cảnh VN
- [ ] Upload lên Colab Drive `custom_vn/`
- [ ] Chạy batch inference
- [ ] Phân loại: thành công / một phần / thất bại
- [ ] Ghi nhận 3–5 failure cases

### 4c. Style editing (tùy chọn)

- [ ] Tập trung global edit: ngày↔đêm, mùa xuân/hạ/thu/đông, mưa↔nắng, overcast/golden hour
- [ ] Chọn 20–40 ảnh outdoor/street/landscape
- [ ] Chạy prompt nhẹ và prompt mạnh trên cùng ảnh
- [ ] Nếu có full-image mask: so SwiftEdit-SG vs SwiftEdit-FullMask
- [ ] Đánh giá bằng CLIP target, zero-shot CLIP label, DINO/CLIP image similarity, LPIPS/SSIM phụ, IQA/human rating và runtime
- [ ] Ghi failure cases: under-edit, đổi không đều, mất layout, artifact ánh sáng/mưa/tuyết

### 4d. Object removal / inpainting (tùy chọn, khuyến nghị)

- [x] Cho phép người dùng **khoanh vùng** vật thể cần xóa: `user_mask` trong `edit_image` ghi đè self-guided mask; UI tab "Xóa vật thể" (`scripts/app_gradio.py`)
- [x] Viết prompt `"a scene with [object]"` → `"the same scene without [object]"`
- [x] Chạy SwiftEdit-UserMask: xóa OK vật nhỏ/vừa (headphones), kiểm chứng mask khoanh đúng vùng (test tường gạch nửa trái)
- [ ] Chọn 20–40 ảnh có object cần xóa: người, xe, chai/lon, biển báo, rác, vật trên bàn
- [ ] Nếu có object mask GT: chạy SwiftEdit-GTMask và SwiftEdit-SG+Dilate
- [ ] Nếu đủ thời gian: chạy LaMa baseline với cùng mask
- [ ] Đánh giá detector confidence drop / CLIP margin, PSNR/SSIM/LPIPS ngoài mask, realism human rating/IQA và runtime
- [x] Ghi failure cases: vật rất lớn (xe đạp ~39% khung) còn sót — SwiftEdit không phải inpainting chuyên dụng

> **Dữ liệu thực nghiệm:** `experimental_data/object_removal_2026-06-14/` (report + ảnh source/mask/result cho ca thành công và ca giới hạn).
> **Chạy:** tab "Xóa vật thể (khoanh vùng)" trong `python scripts/app_gradio.py`, hoặc self-test `python scripts/app_gradio.py --selftest-removal <ảnh>`.

### 4e. SwiftEdit-RT inference acceleration (hướng đào sâu)

- [x] Chốt hướng đầu tiên: tắt decode `noise_image` không dùng trong `gen_img()` khi chỉ cần ảnh edited
- [ ] Đo speedup của `SwiftEdit-no-noise-decode` so với baseline
- [x] Patch self-guided mask threshold chạy vectorized trên GPU, tránh CPU round-trip
- [x] Profile baseline theo module: VAE encode/decode, text encoder, inverse UNet, mask, CLIP image encoder, generation UNet
- [x] Thêm chế độ cache latent ảnh nguồn, image embedding IP-Adapter và source prompt embedding cho demo cùng ảnh nhiều prompt — `EditCache` (infer.py); tiết kiệm ~9.93s/edit khi cùng ảnh+source prompt
- [x] Thử `fp16` + `channels_last` (Mac M4/MPS): ~3.3× (nguội) → ~7× (chạy liên tục); PSNR ~45dB vs fp32, không NaN/đen; VAE giữ fp32
- [x] **Benchmark quy mô lớn trên Colab T4** (200 ảnh × 3 prompt × 4 config fp32/fp16/fp8/fp4): fp16+cache **1.70×–1.82×**, VRAM **−42.1%**, PSNR **48.6dB**; fp8 **1.92×** nhưng PSNR **6.0dB** (không dùng được); fp4 VRAM **−48.5%**, PSNR **21.7dB** — `experimental_data/quality_speed_bench_2026-06-17/`
- [ ] Thử `torch.compile` trên Colab CUDA
- [ ] Thử TinyVAE/TAESD như ablation tốc độ/chất lượng cho VAE encode/decode
- [x] Lập bảng latency breakdown, speedup, peak memory (VRAM) và metric PSNR/SSIM/LPIPS/MSE so với baseline — notebook `CS2309_SwiftEdit_quality_speed_bench.ipynb`

### 4f. Demo Gradio (tùy chọn)

- [x] Gradio: upload ảnh + nhập prompt — `scripts/app_gradio.py`
- [x] Hiển thị ảnh output + thời gian suy luận (kèm dtype + trạng thái cache)
- [x] Chạy local trên Mac (fp16 + channels_last + EditCache); self-test OK ~7.8s edit đầu

> **Chạy demo:** `python scripts/app_gradio.py` (mặc định fp16 trên MPS), mở `http://127.0.0.1:7860`. Dùng cùng 1 ảnh + source prompt rồi đổi edit prompt để thấy cache tăng tốc.

### 4g. SAM 3 mask quality analysis (optional, không phải hướng chính)

- [ ] Ghi rõ lý do không chọn làm hướng chính: thêm segmentation model làm chậm pipeline realtime
- [ ] Nếu còn thời gian: dùng SAM 3 offline để phân tích chất lượng mask, không tính là pipeline realtime
- [ ] So sánh định tính 5–10 mẫu self-guided / GT / SAM 3 mask

### 4h. Fine-tune nhẹ Colab (tùy chọn — bỏ qua vẫn đủ)

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

- [x] Mac: ≥5 ví dụ chỉnh sửa (input → output) — `experimental_data/piebench_subset20_2026-06-14/`, `quality_speed_bench_2026-06-17/`
- [x] Colab: metrics ≥50 mẫu — **2400 edit** (600×4 config) trên Tesla T4 (`experimental_data/quality_speed_bench_2026-06-17/quality_raw.csv`)
- [ ] Có ablation ≥1 hyperparameter (grid ảnh)
- [x] Có phân tích SwiftEdit-RT: latency breakdown + ít nhất 2 tối ưu inference (cache + fp16/channels_last; tốc độ + VRAM + chất lượng quy mô lớn)
- [ ] Có phân tích global style/weather edit: metric không dùng mask + failure cases
- [x] Có phân tích object removal: removal success (vật nhỏ/vừa) + failure case (vật lớn) — `experimental_data/object_removal_2026-06-14/`
- [ ] Có so sánh baseline (Colab) **hoặc** phân tích failure cases chi tiết
- [x] Bảng runtime: Mac (`fp16_benchmark_2026-06-14`) + Colab T4 (`quality_speed_bench_2026-06-17`: fp32 2.91s, fp16+cache 1.71s, fp8 1.52s, fp4 1.74s/edit)
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
| HF **401** `stable-diffusion-2-1-base` | Dùng `models.py` đã patch (mirror `Manojb/...`); chạy `bash scripts/download_hf_models.sh` |
| HF **timeout** / `cas-bridge.xethub` | `export HF_HUB_DOWNLOAD_TIMEOUT=900`; chạy lại `download_hf_models.sh` |
| `deserialize object on a CUDA device` (Mac) | `models.py` đã dùng `map_location="cpu"` cho `ip_adapter.bin` |
| Mac: lỗi MPS operator | `export PYTORCH_ENABLE_MPS_FALLBACK=1` hoặc chạy mẫu đó trên Colab |
| Mac: OOM / chậm (~90s/ảnh MPS) | Bình thường vs A100 paper; giữ 512×512; benchmark lớn → Colab T4 |
| Colab: Drive ~15 GB trống | Chật — weights ~9.6 GB + HF tối thiểu; khuyến nghị ≥20–25 GB trên Drive |
| Colab T4 VRAM ~15 GB | Đủ inference 512×512; PieBench batch=1 |
| Colab: disconnect | Chia batch 20 mẫu/session; lưu progress trên Drive |
| Colab: OOM T4 | `float16`, batch=1 |
| Không tải được weights Qualcomm | `bash scripts/download_swiftedit_weights.sh` hoặc `curl` từng `part-aa`…`ae` |
| Metrics Mac ≠ Colab | Cùng seed, dtype, hyperparameter |

---

## Tài liệu tham khảo

1. [SwiftEdit Paper (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_SwiftEdit_Lightning_Fast_Text-Guided_Image_Editing_via_One-Step_Diffusion_CVPR_2025_paper.pdf)
2. [GitHub — Qualcomm-AI-research/SwiftEdit](https://github.com/Qualcomm-AI-research/SwiftEdit)
3. [Project page](https://swift-edit.github.io/)
4. [PIE-Bench / PnP Inversion](https://github.com/cure-lab/PnPInversion)
5. [CLIP](https://proceedings.mlr.press/v139/radford21a) · [CLIPScore](https://arxiv.org/abs/2104.08718)
6. [Diffusers — AutoencoderTiny / TAESD](https://huggingface.co/docs/diffusers/en/api/models/autoencoder_tiny)
7. [Diffusers — Optimize inference](https://huggingface.co/docs/diffusers/main/optimization/fp16)
8. [PyTorch — `torch.compile` and Diffusers](https://docs.pytorch.org/devlogs/inductor/2026-05-11-torch-compile-and-diffusers/)
9. [NVIDIA TensorRT — Stable Diffusion acceleration](https://developer.nvidia.com/blog/tensorrt-accelerates-stable-diffusion-nearly-2x-faster-with-8-bit-post-training-quantization/)
10. [SAM 3 — Segment Anything with Concepts](https://github.com/facebookresearch/sam3) *(optional, không chọn làm hướng chính)*
11. [TurboEdit](https://github.com/GaMaLielD/TurboEdit) *(baseline, nếu so sánh)*
12. [DINO — self-supervised ViT features](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper)
13. [LPIPS — perceptual similarity](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0299.pdf)
14. [MUSIQ — no-reference image quality](https://mlanthology.org/iccv/2021/ke2021iccv-musiq/)
15. [FID / TTUR](https://papers.nips.cc/paper/7240-gans-trained-by-a-two-time-scale-update-rule-converge-to-a-local-nash-equilibrium)
16. [CycleGAN — unpaired image-to-image translation](https://junyanz.github.io/CycleGAN/)
17. [LaMa — large mask inpainting](https://openaccess.thecvf.com/content/WACV2022/html/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.html)
18. [ReMOVE — reference-free object erasure metric](https://arxiv.org/abs/2409.00707)

---

## Ghi chú

- **Không training nặng** — dùng checkpoint pretrained; fine-tune chỉ trên Colab nếu cần.
- Ghi rõ trong báo cáo metric nào đo trên **Mac**, **Colab**, hay lấy từ **paper**.
- `CLIP-Whole` và `CLIP-Edited` đều đo ảnh edited với **edit prompt**; không dùng source prompt.
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
