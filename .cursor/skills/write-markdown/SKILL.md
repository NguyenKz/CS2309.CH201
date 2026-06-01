---
name: write-markdown
description: >-
  Viết và chỉnh sửa file Markdown (.md) trong project CS2309.CH201: cấu trúc rõ,
  tiếng Việt, bảng/checklist chuẩn. KHÔNG dùng LaTeX vì nhiều MD reviewer không
  render được. Dùng khi tạo/sửa .md, viết báo cáo, overview, README, nhật ký,
  hoặc user yêu cầu "viết md", "viết markdown", "tài liệu md".
---

# Viết Markdown (không LaTeX)

Skill này áp dụng cho mọi file `*.md` trong repo **CS2309.CH201**, đặc biệt:
`README.md`, `SwiftEdit_Overview.md`, `SwiftEdit_DeTai_CS2309.md`, `NHAT_KY.md`.

## Quy tắc bắt buộc: KHÔNG LaTeX

**Một số MD reviewer (GitHub, GitLab, LMS, Word export) không render LaTeX.**

### Cấm dùng

| Cấm | Ví dụ |
|---|---|
| Inline math `\( ... \)` | `\(x_{source}\)` |
| Block math `\[ ... \]` | `\[ \hat{\epsilon} = F_\theta(z) \]` |
| Dollar math `$...$`, `$$...$$` | `$s_y$` |
| LaTeX commands | `\text{}`, `\quad`, `\hat{}`, `\epsilon` |

### Thay bằng (ưu tiên theo thứ tự)

1. **Chữ thường + Unicode** — ε, θ, σ, ×, ≥, →, ảnh nguồn x_source
2. **Backtick code** — `F_theta`, `s_y`, `s_edit`, `s_non-edit`, `x_source`
3. **Công thức dạng text** (một dòng, không block LaTeX):

```
eps_hat = F_theta(z, c_y)
z_hat = G_IP(eps_hat, c_y, c_x)
M = normalize(|eps_hat_source - eps_hat_edit|)
```

4. **Bảng** thay cho công thức phức tạp

Xem [examples.md](examples.md) để đối chiếu ❌/✅.

## Cấu trúc tài liệu

### Header metadata (file đề tài / overview)

```markdown
# Tiêu đề

> Mô tả ngắn 1–2 câu

| | |
|---|---|
| **Paper** | [link](url) |
| **Code** | [link](url) |
```

### Heading

- Một `#` cho title file
- `##` cho phần chính, `###` cho mục con
- Không nhảy cấp (## → ####)

### Bảng

- Hàng header + `|---|---|`
- Căn nội dung ngắn gọn; tránh công thức trong ô bảng

### Checklist

```markdown
- [ ] Việc chưa xong
- [x] Việc đã xong
```

### Code / pipeline

- Fence ``` trên dòng riêng
- ASCII diagram cho pipeline (không LaTeX)

### Link nội bộ repo

```markdown
[Xem overview](./SwiftEdit_Overview.md)
```

## Giọng văn (CS2309)

- **Tiếng Việt** cho nội dung đề tài; thuật ngữ CV giữ tiếng Anh khi quen dùng: *diffusion*, *inference*, *prompt*, *background preservation*
- Câu hoàn chỉnh, không bullet telegraphic
- Thuật ngữ nhất quán trong cùng file: dùng một tên (vd. "inverted noise", không đổi qua lại "noise đảo")

## Workflow khi viết / sửa MD

1. Đọc file hiện có — giữ tone và cấu trúc
2. Viết nội dung mới **không LaTeX**
3. Nếu gặp LaTeX cũ trong file đang sửa → **chuyển sang plain text** (cùng lần edit)
4. Kiểm tra nhanh: không còn `\(` `\[` `$` `\hat` `\epsilon` `\theta` `\text`
5. File đề tài: sau khi ghi nhật ký / tiến độ → gọi skill `update-readme-progress` nếu có thay đổi checklist

## Checklist trước khi xong

- [ ] Không có LaTeX (inline, block, `$`)
- [ ] Heading có thứ bậc hợp lý
- [ ] Bảng render được (cột khớp)
- [ ] Link không gãy
- [ ] Code fence đóng đúng
- [ ] Phù hợp mục đích file (overview vs checklist vs nhật ký)

## Không làm

- Không thêm LaTeX "cho đẹp" hoặc "giống paper"
- Không dùng HTML entity thay ký tự (`&amp;`, `&lt;`)
- Không tạo file MD user không yêu cầu (trừ khi là phần của task rõ ràng)
- Không duplicate nội dung dài giữa Overview và DeTai — link chéo thay vì copy

## File tham khảo

| File | Vai trò |
|---|---|
| [examples.md](examples.md) | Ví dụ ❌ LaTeX → ✅ plain text |
| `SwiftEdit_Overview.md` | Mẫu overview (cần bỏ LaTeX nếu còn) |
| `README.md` | Mẫu checklist + hướng dẫn |
