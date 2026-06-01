---
name: update-qa
description: >-
  Tự động thêm câu hỏi–trả lời khái niệm vào QA.md cho đề tài CS2309 SwiftEdit.
  Dùng khi user hỏi "là gì", "giải thích", "khác nhau thế nào", "thêm vào QA",
  "ghi QA.md", "lưu câu hỏi", hoặc sau khi agent giải thích khái niệm CV/diffusion/SwiftEdit.
---

# Cập nhật QA.md

Ghi câu hỏi và **tóm tắt câu trả lời** vào [`QA.md`](../../../QA.md) — không LaTeX (theo skill `write-markdown`).

## Khi nào chạy (bắt buộc)

**Cuối turn** nếu:

- User hỏi khái niệm / thuật ngữ / so sánh phương pháp
- User nói *"thêm vào QA"*, *"ghi QA.md"*, *"lưu câu hỏi"*
- Agent vừa giải thích khái niệm liên quan SwiftEdit, diffusion, PieBench, Mac/Colab

**Không chạy** khi: chỉ sửa code, checklist, nhật ký tiến độ (dùng `update-readme-progress`).

## Quy trình

### 1. Rút Q&A từ turn vừa rồi

- **Câu hỏi:** giữ nguyên ý user (ngắn, có dấu `?`)
- **Trả lời:** tóm tắt 3–7 bullet — **không copy cả essay**, không LaTeX
- **Ghi chú:** link paper, file repo (`SwiftEdit_Overview.md`), nếu có

### 2. Chọn section (1–8)

Tra [topic-map.md](topic-map.md). Mặc định **8** nếu không chắc.

### 3. Chạy script

```bash
python .cursor/skills/update-qa/scripts/add_qa.py \
  --question "Diffusion inversion là gì?" \
  --answer "Quá trình tìm noise/latent ban đầu để tái tạo ảnh nguồn.\nSwiftEdit dùng mạng F_theta one-step thay DDIM multi-step.\nInverted noise eps_hat là điểm xuất phát cho editing." \
  --section 3 \
  --tags "#inversion #noise #swiftedit"
```

Nhiều dòng `--answer`: dùng `\n` giữa các bullet trong shell, hoặc sửa file trực tiếp nếu câu trả lời dài.

Dry-run:

```bash
python .cursor/skills/update-qa/scripts/add_qa.py --dry-run -q "..." -a "..."
```

Script tự:

- Thêm entry **mới nhất lên trên** trong mục
- Xóa dòng `*(Chưa có câu hỏi.)*`
- Cập nhật bảng **Mục lục nhanh** (số câu / mục)
- Bỏ qua nếu câu hỏi trùng (dùng `--force` để ghi đè)

### 4. Hoặc sửa QA.md trực tiếp

Nếu script không chạy được, chèn entry sau dòng `<!-- qa:insert -->` trong mục tương ứng:

```markdown
### Q: [Câu hỏi]?

**Ngày:** YYYY-MM-DD  
**Chủ đề:** #tag1 #tag2

**Trả lời (tóm tắt):**
- …

**Ghi chú thêm / link:**
- …
```

Sau đó cập nhật tay cột **Số câu hỏi** trong mục lục.

### 5. Báo user

Một dòng: *"Đã thêm vào QA.md → mục X: [câu hỏi rút gọn]"*

## Phối hợp skill khác

| Tình huống | Skill |
|---|---|
| Hỏi khái niệm + ghi QA | **update-qa** (skill này) |
| Xong task thực nghiệm | `update-readme-progress` |
| Viết file MD mới | `write-markdown` |

Có thể chạy **cả hai** trong cùng turn nếu vừa giải thích khái niệm vừa hoàn thành task.

## Không làm

- Không LaTeX trong entry
- Không trả lời dài trong QA — chỉ tóm tắt bullet
- Không thêm câu hỏi trùng
- Không xóa entry cũ

## File liên quan

| File | Vai trò |
|---|---|
| `QA.md` | Target |
| [topic-map.md](topic-map.md) | Chọn section + tags |
| [scripts/add_qa.py](scripts/add_qa.py) | Append + đếm |
