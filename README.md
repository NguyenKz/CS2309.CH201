# CS2309.CH201 — SwiftEdit

**Chỉnh sửa ảnh theo văn bản một bước (One-step Text-guided Image Editing)**

| | |
|---|---|
| **Môn học** | CS2309.CH201 — Chuyên đề nghiên cứu và ứng dụng về Thị giác máy tính |
| **Chủ đề** | Chỉnh sửa và thay đổi phong cách ảnh |
| **Paper** | [SwiftEdit (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_SwiftEdit_Lightning_Fast_Text-Guided_Image_Editing_via_One-Step_Diffusion_CVPR_2025_paper.pdf) |
| **Repo gốc** | [Qualcomm-AI-research/SwiftEdit](https://github.com/Qualcomm-AI-research/SwiftEdit) |
| **Môi trường** | MacBook Air M4 24GB + Google Colab (T4) |

Tài liệu: [`SwiftEdit_Overview.md`](./SwiftEdit_Overview.md) · [`SwiftEdit_DeTai_CS2309.md`](./SwiftEdit_DeTai_CS2309.md) · [`QA.md`](./QA.md) · [`NHAT_KY.md`](./NHAT_KY.md)

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
│   └── CS2309_SwiftEdit_test.ipynb
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
> Cập nhật tiến độ lần cuối: 2026-06-14 — 22/78 task bắt buộc

### Trạng thái nhanh

| Giai đoạn | Tiến độ |
|---|---|
| 1. Lý thuyết | 🔄 Đang làm (1/6) |
| 2. Setup Mac + Colab | 🔄 Đang làm (11/17) |
| 3. Thực nghiệm cơ bản | 🔄 Đang làm (6/19) |
| 4. So sánh & mở rộng | 🔄 Đang làm (4/17) |
| 5. Báo cáo & nộp | ⬜ Chưa bắt đầu |












---

## Giai đoạn 1 — Lý thuyết (Tuần 1–2)

- [ ] Đọc paper SwiftEdit (Abstract, Sec. 3–5)
- [ ] Tìm hiểu SwiftBrushv2, DDIM inversion, IP-Adapter
- [x] Tìm hiểu PieBench benchmark
- [ ] Vẽ/tóm tắt pipeline inversion → mask → ARaM
- [ ] So sánh SwiftEdit vs P2P, NT-Inv, TurboEdit, ICD (bảng related work)
- [ ] Viết phần Overview + Related Work trong báo cáo

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

- [ ] Chọn 20–40 ảnh có object cần xóa: người, xe, chai/lon, biển báo, rác, vật trên bàn
- [ ] Viết prompt `"a scene with [object]"` → `"the same scene without [object]"`
- [ ] Chạy SwiftEdit-SG và visualize self-guided mask
- [ ] Nếu có object mask: chạy SwiftEdit-UserMask/GTMask và SwiftEdit-SG+Dilate
- [ ] Nếu đủ thời gian: chạy LaMa baseline với cùng mask
- [ ] Đánh giá detector confidence drop / CLIP margin, PSNR/SSIM/LPIPS ngoài mask, realism human rating/IQA và runtime
- [ ] Ghi failure cases: ghost object, viền artifact, nền méo, xóa nhầm

### 4e. SwiftEdit-RT inference acceleration (hướng đào sâu)

- [x] Chốt hướng đầu tiên: tắt decode `noise_image` không dùng trong `gen_img()` khi chỉ cần ảnh edited
- [ ] Đo speedup của `SwiftEdit-no-noise-decode` so với baseline
- [ ] Patch self-guided mask threshold chạy vectorized trên GPU, tránh CPU round-trip
- [ ] Profile baseline theo module: VAE encode/decode, text encoder, inverse UNet, mask, CLIP image encoder, generation UNet
- [ ] Thêm chế độ cache latent ảnh nguồn, image embedding IP-Adapter và source prompt embedding cho demo cùng ảnh nhiều prompt
- [ ] Thử `fp16`, `channels_last`, `torch.compile` trên Colab CUDA
- [ ] Thử TinyVAE/TAESD như ablation tốc độ/chất lượng cho VAE encode/decode
- [ ] Lập bảng latency breakdown, speedup, peak memory và metric PSNR/MSE/CLIP so với baseline

### 4f. Demo Gradio (tùy chọn)

- [ ] Gradio: upload ảnh + nhập prompt
- [ ] Hiển thị ảnh output + thời gian suy luận
- [ ] Chạy local trên Mac

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

- [ ] Mac: ≥5 ví dụ chỉnh sửa (input → output)
- [ ] Colab: `metrics.csv` PieBench ≥50 mẫu
- [ ] Có ablation ≥1 hyperparameter (grid ảnh)
- [ ] Có phân tích SwiftEdit-RT: latency breakdown + ít nhất 2 tối ưu inference
- [ ] Có phân tích global style/weather edit: metric không dùng mask + failure cases
- [ ] Có phân tích object removal: removal success + background preservation + failure cases
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
