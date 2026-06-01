# Mẫu entry nhật ký

Agent dùng cấu trúc này khi thêm entry vào `NHAT_KY.md`.

## Dòng tóm tắt (bảng)

| Ngày | Giai đoạn | Công việc | Kết quả / Ghi chú | Môi trường |
| YYYY-MM-DD | 2a. Mac / 3c. PieBench / … | 1 câu ngắn | Kết quả đo được | Mac / Colab / Mac+Colab |

## Block chi tiết (prepend — mới nhất trên cùng)

```markdown
### YYYY-MM-DD — [Giai đoạn] Tiêu đề ngắn

**Môi trường:** Mac M4 (MPS) | Google Colab (T4) | …

**Công việc đã làm:**
- …

**Kết quả:**
- …

**Task README đã đánh [x]:**
- …

**Vấn đề / cách xử lý:** *(nếu có)*
- …

**Bước tiếp theo:**
- …

---
```

## Quy tắc viết

- Tiếng Việt, câu ngắn, có số liệu cụ thể (runtime, số mẫu, tên file).
- Mỗi session làm việc = 1 entry (kể cả session ngắn).
- Không xóa entry cũ; chỉ thêm mới.
- Giai đoạn dùng label README: `1`, `2a`, `2b`, `3a`–`3c`, `4a`–`4e`, `5`.
