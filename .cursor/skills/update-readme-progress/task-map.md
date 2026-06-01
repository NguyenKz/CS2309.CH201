# Task map — README checkbox ↔ tín hiệu hoàn thành

Agent dùng bảng này để biết checkbox nào đánh `[x]` khi thấy bằng chứng trong repo hoặc user báo xong.

| Checkbox (substring khớp) | Bằng chứng |
|---|---|
| `Clone repo SwiftEdit` | Thư mục `SwiftEdit/` có `infer.py` |
| `Tạo conda env` | User chạy conda / có `environment.yml` |
| `Cài PyTorch Mac` | `import torch; torch.backends.mps.is_available()` OK |
| `Tải checkpoint` | `SwiftEdit/swiftedit_weights/` không rỗng |
| `Chạy demo assets/imgs_demo` | Ảnh output trong `results/` hoặc user xác nhận |
| `Tạo notebook CS2309_SwiftEdit` | `notebooks/CS2309_SwiftEdit.ipynb` tồn tại |
| `Mount Drive` | Notebook có cell mount Drive đã chạy |
| `PieBench` + `50` | `results/piebench/metrics.csv` ≥50 dòng |
| `metrics.csv` | File `results/piebench/metrics.csv` |
| `ablation` | Ảnh trong `results/ablation/` |
| `Gradio` | File demo Gradio hoặc user chạy demo |
| `Slide trình bày` | File trong `report/*.pptx` hoặc `*.pdf` |
| `Đọc paper SwiftEdit` | User xác nhận / ghi chú trong báo cáo |
| `TurboEdit` | Kết quả baseline trong `results/` |

## Gợi ý nội dung nhật ký theo task

| Task xong | `--journal-work` | `--journal-result` gợi ý |
|---|---|---|
| Clone + demo Mac | Setup SwiftEdit trên Mac M4 | Demo OK; Xs/ảnh; RAM Y GB |
| Setup Colab | Notebook + Drive + weights | T4 OK; symlink weights session 2 |
| PieBench | Eval N mẫu PieBench | metrics.csv; PSNR/CLIP; runtime |
| Ablation | Thử sy/s_edit/s_non-edit | Grid ảnh trong results/ablation/ |
| Đọc paper | Đọc SwiftEdit Sec. X | Tóm tắt pipeline / insight |

## Giai đoạn ↔ section README

| ID | Section header |
|---|---|
| 1 | `## Giai đoạn 1 — Lý thuyết` |
| 2 | `## Giai đoạn 2 — Setup môi trường` |
| 3 | `## Giai đoạn 3 — Thực nghiệm cơ bản` |
| 4 | `## Giai đoạn 4 — So sánh & mở rộng` |
| 5 | `## Giai đoạn 5 — Báo cáo & nộp` |

Task **tùy chọn** (4b–4e, Gradio, fine-tune) không tính vào % hoàn thành giai đoạn trong script sync.
