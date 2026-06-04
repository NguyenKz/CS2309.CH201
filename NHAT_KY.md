# Nhật ký làm việc — CS2309 SwiftEdit

> Ghi chép tiến độ đề tài. Tự động cập nhật bởi skill `.cursor/skills/update-readme-progress/`.
> Đồng bộ tóm tắt sang [`SwiftEdit_DeTai_CS2309.md`](./SwiftEdit_DeTai_CS2309.md) Mục 8.1.

---

## Tóm tắt nhanh

| Ngày | Giai đoạn | Công việc | Kết quả / Ghi chú | Môi trường |
|---|---|---|---|---|
| 2026-06-04 | 2a. Mac | Clone SwiftEdit; pyenv 3.12.10 + .venv + requirements-mac.txt (PyTorch MPS) | Demo woman→Taylor Swift OK; output SwiftEdit/result_woman->Taylor Swift.png; ~91s/ảnh trên MPS | Mac M4 (MPS); pyenv 3.12.10; .venv |
| 2026-06-01 | 0. Khởi tạo project | Tạo README, đề tài chi tiết, skill Cursor hỗ trợ nhật ký | Repo CS2309.CH201 sẵn sàng; skill sync README + NHAT_KY + §8.1 | Mac M4 |

---

## Chi tiết theo phiên làm việc

*(Các entry chi tiết xuất hiện bên dưới, mới nhất ở trên cùng.)*

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

