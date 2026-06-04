# Nhật ký làm việc — CS2309 SwiftEdit

> Ghi chép tiến độ đề tài. Tự động cập nhật bởi skill `.cursor/skills/update-readme-progress/`.
> Đồng bộ tóm tắt sang [`SwiftEdit_DeTai_CS2309.md`](./SwiftEdit_DeTai_CS2309.md) Mục 8.1.

---

## Tóm tắt nhanh

| Ngày | Giai đoạn | Công việc | Kết quả / Ghi chú | Môi trường |
|---|---|---|---|---|
| 2026-06-04 | 2b. Colab | Chạy notebook CS2309_SwiftEdit_test trên Colab T4 (extension): preset dog→dog wi | edit_image 1.3s; output results/notebook/nb_dog_dog_to_dog_with_mouth_opened.png; so với Mac MPS ~91s (woman) và paper A | Google Colab — Tesla T4 (Colab extension) |
| 2026-06-04 | 2b. Colab | Patch notebook + requirements (GPU T4, HF stack mới, upload path) | Sẵn sàng chạy Colab extension; fix EncoderDecoderCache/numpy/upload; chưa log runtime T4 OK | Colab extension + Colab web |
| 2026-06-04 | 2a. Mac | Notebook notebooks/CS2309_SwiftEdit_test.ipynb (preset, upload ipywidgets 8, inf | Notebook chạy OK trên Mac (.venv); upload widget sửa tuple ipywidgets 8 | Mac M4 (MPS); Jupyter .venv |
| 2026-06-04 | 2a. Mac | Clone SwiftEdit; pyenv 3.12.10 + .venv + requirements-mac.txt (PyTorch MPS) | Demo woman→Taylor Swift OK; output SwiftEdit/result_woman->Taylor Swift.png; ~91s/ảnh trên MPS | Mac M4 (MPS); pyenv 3.12.10; .venv |
| 2026-06-01 | 0. Khởi tạo project | Tạo README, đề tài chi tiết, skill Cursor hỗ trợ nhật ký | Repo CS2309.CH201 sẵn sàng; skill sync README + NHAT_KY + §8.1 | Mac M4 |

---

## Chi tiết theo phiên làm việc

*(Các entry chi tiết xuất hiện bên dưới, mới nhất ở trên cùng.)*

### 2026-06-04 — [2b. Colab] Inference Colab T4 — dog preset 1.3s

**Môi trường:** Google Colab — Tesla T4 (Colab extension)

**Công việc đã làm:**
- Chạy notebook CS2309_SwiftEdit_test trên Colab T4 (extension): preset dog→dog with mouth opened (assets/imgs_demo/02.jpg)

**Kết quả:**
- `Edit 'dog' -> 'dog with mouth opened' in 1.3s` (chỉ `edit_image`, model đã load)
- Lưu: `/content/CS2309.CH201/results/notebook/nb_dog_dog_to_dog_with_mouth_opened.png`
- Ảnh: `assets/imgs_demo/02.jpg` — preset `dog` / `dog with mouth opened`

**So sánh runtime (cùng SwiftEdit 512×512, khác preset/ảnh):**

| Môi trường | Thời gian | Ghi chú |
|---|---|---|
| Paper A100 | ~0.23 s | Báo cáo gốc |
| **Colab T4** | **1.3 s** | Phiên này — inference-only |
| Mac M4 MPS | ~91 s | Demo woman→Taylor Swift (có thể gồm overhead MPS lần đầu) |

**Task README đã đánh [x]:**
- Ghi GPU name (T4/A100) và runtime Colab
- Chạy demo CUDA — kết quả khớp Mac
- Cài `requirements.txt` (CUDA)

**Vấn đề / cách xử lý:**
Không dùng Drive — weights/HF trên /content

**Bước tiếp theo:**
- Ghi bảng runtime 3 cột Mac|Colab T4|Paper; PieBench batch Colab

---

### 2026-06-04 — [2b. Colab] Notebook Colab extension: GPU, upload, stack HF mới

**Môi trường:** Colab extension (Cursor/VS Code) + Colab web (T4)

**Công việc đã làm:**
- **`notebooks/CS2309_SwiftEdit_test.ipynb`**
  - Metadata `accelerator: GPU`, `colab.gpuType: T4` (Colab web tự chọn T4)
  - Cell 1: `_check_colab_gpu()` (`nvidia-smi`) trước clone/tải weights; hướng dẫn extension **New Colab Server → GPU → T4**
  - Cell Setup: `pip install -U -r requirements.txt` — **không** pin `numpy==1.26.4` trên Colab; kiểm tra `transformers≥4.46`, `diffusers≥0.32`, import `EncoderDecoderCache`
  - Upload ảnh: ô **Đường dẫn** + **Upload to Colab** (Explorer); nút demo `02.jpg`; **ẩn** `files.upload()` / FileUpload trên extension (widget rỗng `value={}`)
- **`SwiftEdit/requirements.txt`**: stack mới `transformers≥4.46`, `diffusers≥0.32`, `peft`, `accelerate` — bỏ pin torch `cu118` / `2.2.1`
- **`requirements-mac.txt`**: đồng bộ stack mới; `numpy≥1.26`
- **`SwiftEdit/models.py`**: `torch.load(..., weights_only=True)` cho `ip_adapter.bin`

**Kết quả:**
- Patch để Colab extension **có thể chạy** sau quy trình: Restart kernel → cell 1 → Setup → Load models → Áp dụng ảnh (path) → inference
- Chưa ghi nhận inference Colab T4 thành công end-to-end trong phiên này — cần chạy lại sau khi push/pull notebook mới

**Task README đã đánh [x]:**
- *(chưa đánh [x] — chờ xác nhận chạy inference Colab OK)*

**Vấn đề / cách xử lý:**

| Lỗi | Nguyên nhân | Cách xử lý |
|-----|-------------|------------|
| `Colab chưa có GPU` | Server CPU / Auto Connect | Extension: **New Colab Server → T4**; web: Runtime → T4 |
| `numpy.dtype size changed` | Hạ `numpy==1.26.4` trên Colab có torchvision build NumPy 2 | Bỏ pin numpy trên Colab |
| `EncoderDecoderCache` ImportError | Colab `diffusers` mới + `transformers==4.37` cũ | Nâng stack (`transformers≥4.46`, `diffusers≥0.32`), không hạ về 0.22 |
| Upload widget rỗng / `files.upload` disabled | Colab extension không có browser session Colab | **Upload to Colab** → dán `/content/...` hoặc nút **Dùng ảnh demo** |

**Bước tiếp theo:**
- Restart kernel → Setup → Load → chạy preset hoặc upload path → ghi runtime inference T4 (s) vào nhật ký
- Đánh `[x]` README mục Colab khi có bằng chứng (ảnh output + `torch`/`GPU` name)

---

### 2026-06-04 — [2a. Mac] Notebook test, tài liệu Colab/T4, ghi chú runtime

**Môi trường:** Mac M4 (MPS); Jupyter .venv

**Công việc đã làm:**
- Notebook notebooks/CS2309_SwiftEdit_test.ipynb (preset, upload ipywidgets 8, inference)
- README: hướng dẫn cài đặt/pyenv/scripts; .gitignore loại weights
- Ghi nhận: Mac MPS ~91s vs paper A100 ~0.23s; Colab T4 VRAM 15GB đủ inference; Drive ~15GB chật (weights 9.6GB+HF)

**Kết quả:**
- Notebook chạy OK trên Mac (.venv); upload widget sửa tuple ipywidgets 8
- Chưa chạy Colab thực tế — đã ghi kế hoạch T4 + lưu weights Drive trong đề tài/README

**Task README đã đánh [x]:**
- Tạo notebook `CS2309_SwiftEdit_test.ipynb`

**Vấn đề / cách xử lý:**
matplotlib thiếu kernel → %pip trong notebook; FileUpload value tuple (ipywidgets 8)

**Bước tiếp theo:**
- Chạy thử Colab T4: IN_COLAB=True, weights trên Drive; đo runtime inference-only; PieBench batch

---

### 2026-06-04 — [2a. Mac] Setup Mac (pyenv/venv) và chạy demo SwiftEdit

**Môi trường:** Mac M4 (MPS); pyenv 3.12.10; .venv

**Công việc đã làm:**
- Clone SwiftEdit; pyenv 3.12.10 + .venv + requirements-mac.txt (PyTorch MPS)
- Tải swiftedit_weights (~9.6GB); patch infer.py (get_device mps); patch models.py (SD2.1 mirror Manojb, map_location CPU)
- Tải HF: sd-turbo, Manojb/stable-diffusion-2-1-base, h94/IP-Adapter image_encoder
- Bổ sung tài liệu: §1.3 pipeline cũ, ảnh assets/pipeline/, QA.md

**Kết quả:**
- Demo woman→Taylor Swift OK; output SwiftEdit/result_woman->Taylor Swift.png; ~91s/ảnh trên MPS
- Scripts: scripts/run_swiftedit.sh, download_swiftedit_weights.sh

**Task README đã đánh [x]:**
- Cài PyTorch Mac (MPS) — **không** dùng bản CUDA
- Cài dependencies còn lại
- Clone repo SwiftEdit
- Tải checkpoint → `swiftedit_weights/`
- Sửa `device = "mps"` trong code
- Chạy demo `assets/imgs_demo` thành công
- Ghi thời gian/ảnh và RAM peak (Activity Monitor)

**Vấn đề / cách xử lý:**
stabilityai/stable-diffusion-2-1-base 401 → mirror Manojb
IP-Adapter image_encoder timeout → tải riêng model.safetensors
ip_adapter.bin load CUDA → map_location=cpu

**Bước tiếp theo:**
- Ghi RAM peak (Activity Monitor); setup Colab notebook
- Ablation hyperparameter; PieBench eval trên Colab

---

### 2026-06-01 — [0. Khởi tạo project] Khởi tạo cấu trúc đề tài

**Môi trường:** Mac M4

**Công việc đã làm:**
- Tạo README, đề tài chi tiết, skill Cursor hỗ trợ nhật ký

**Kết quả:**
- Repo CS2309.CH201 sẵn sàng; skill sync README + NHAT_KY + §8.1

**Task README đã đánh [x]:**
- *(không đánh task README)*

**Bước tiếp theo:**
Clone SwiftEdit; cài env Mac MPS

---

