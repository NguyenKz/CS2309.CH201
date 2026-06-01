---
name: update-readme-progress
description: >-
  Cập nhật checklist README.md, bảng tiến độ và nhật ký làm việc (NHAT_KY.md) cho đề tài
  CS2309 SwiftEdit. Đồng bộ tóm tắt sang SwiftEdit_DeTai_CS2309.md Mục 8.1. Dùng khi hoàn
  thành task đề tài, user báo tiến độ, "cập nhật readme", "ghi nhật ký", "nhật ký làm việc",
  "sync progress", hoặc sau khi cài đặt/chạy thử SwiftEdit, Colab, PieBench trên CS2309.CH201.
---

# Cập nhật tiến độ + Nhật ký làm việc

Skill đồng bộ 3 file:

| File | Nội dung |
|---|---|
| `README.md` | Checklist `[x]` + bảng **Trạng thái nhanh** |
| `NHAT_KY.md` | Nhật ký chi tiết từng phiên làm việc |
| `SwiftEdit_DeTai_CS2309.md` §8.1 | Bảng tóm tắt nhật ký (copy từ `NHAT_KY.md`) |

## Khi nào chạy (bắt buộc)

Cuối turn nếu vừa: cài env, chạy demo, PieBench, ablation, Colab, viết báo cáo — hoặc user nói *cập nhật readme / ghi nhật ký*.

## Quy trình đầy đủ

### 1. Xác định task + bằng chứng

- Tra [task-map.md](task-map.md).
- Chỉ đánh `[x]` khi có bằng chứng (file, lệnh OK, user xác nhận).

### 2. Ghi nhật ký

Mỗi session làm việc → **1 entry**. Viết theo [journal-template.md](journal-template.md).

Chạy một lệnh (ưu tiên):

```bash
python .cursor/skills/update-readme-progress/scripts/sync_progress.py \
  --mark "Clone repo SwiftEdit" "Chạy demo assets/imgs_demo" \
  --journal-phase "2a. Mac" \
  --journal-work "Clone SwiftEdit, cài conda env Python 3.12 + PyTorch MPS" \
  --journal-result "Demo assets/imgs_demo OK; ~8s/ảnh; RAM peak ~14GB" \
  --journal-env "Mac M4 (MPS)" \
  --journal-title "Setup Mac và chạy demo đầu tiên" \
  --journal-issues "Ban đầu lỗi cu118 — đã cài torch bản Mac" \
  --journal-next "Setup Colab notebook; tải weights lên Drive"
```

Chỉ ghi nhật ký (không đánh task):

```bash
python .cursor/skills/update-readme-progress/scripts/sync_progress.py \
  --journal-phase "1" \
  --journal-work "Đọc paper SwiftEdit Sec. 3-5" \
  --journal-result "Hiểu pipeline inversion + ARaM; ghi chú related work" \
  --journal-env "Mac (đọc tài liệu)"
```

Chỉ sync README (không nhật ký):

```bash
python .cursor/skills/update-readme-progress/scripts/sync_progress.py --mark "Đọc paper SwiftEdit"
```

Dry-run: thêm `--dry-run`.

### 3. Nội dung agent tự viết cho nhật ký

Khi không chạy script (hoặc bổ sung tay), thêm vào `NHAT_KY.md`:

**Bảng tóm tắt** — 1 dòng: ngày, giai đoạn (`1`, `2a`, `2b`, `3a`…), công việc, kết quả có **số liệu**, môi trường.

**Block chi tiết** (prepend, mới nhất trên) gồm:
- Công việc đã làm (bullet)
- Kết quả cụ thể
- Task README đã `[x]`
- Vấn đề / cách xử lý (nếu có)
- Bước tiếp theo

Sau đó chạy:

```bash
python .cursor/skills/update-readme-progress/scripts/sync_progress.py --journal-sync-detai-only
```

### 4. Sync README

Script tự chạy sau `--journal-*` hoặc `--mark`. Cập nhật:
- Bảng **Trạng thái nhanh** (`⬜` / `🔄 (n/m)` / `✅`)
- Dòng `> Cập nhật tiến độ lần cuối: …`

### 5. Báo cáo user

Tóm tắt: task `[x]`, entry nhật ký mới, trạng thái 5 giai đoạn, bước tiếp theo.

## Label giai đoạn nhật ký

| Label | Ý nghĩa |
|---|---|
| `1` | Lý thuyết |
| `2a. Mac` | Setup Mac |
| `2b. Colab` | Setup Colab |
| `3a` | Ablation hyperparameter |
| `3b` | Self-guided mask |
| `3c` | PieBench eval |
| `4a` | So sánh baseline |
| `4b`–`4e` | Tùy chọn (ảnh VN, style, Gradio, fine-tune) |
| `5` | Báo cáo & nộp |

## Ví dụ

**User:** "Hôm nay chạy PieBench 50 mẫu trên Colab xong."

```bash
python .cursor/skills/update-readme-progress/scripts/sync_progress.py \
  --mark "Colab: chạy" "metrics.csv" "PieBench" \
  --journal-phase "3c" \
  --journal-work "Eval PieBench 50 mẫu trên Colab T4" \
  --journal-result "metrics.csv lưu Drive; PSNR≈22.8, CLIP-Whole≈24.9; ~6s/ảnh" \
  --journal-env "Google Colab (T4)" \
  --journal-next "So sánh TurboEdit 20 mẫu"
```

## Không làm

- Không xóa entry nhật ký cũ.
- Không đánh `[x]` / ghi nhật ký cho việc chưa làm.
- Không sửa nội dung báo cáo user đã viết tay (§8.2 trở đi) trừ khi user yêu cầu.
- Không commit trừ khi user yêu cầu.

## File liên quan

| File | Vai trò |
|---|---|
| `README.md` | Checklist + trạng thái |
| `NHAT_KY.md` | Nhật ký chính (source of truth) |
| `SwiftEdit_DeTai_CS2309.md` §8.1 | Bảng tóm tắt sync từ nhật ký |
| [journal-template.md](journal-template.md) | Mẫu entry |
| [task-map.md](task-map.md) | Map bằng chứng → checkbox |
| [scripts/sync_progress.py](scripts/sync_progress.py) | Script sync tất cả |
