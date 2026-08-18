# Kế hoạch kiểm tra — Xóa vật thể / Object removal

> Trạng thái: **chờ kiểm tra thủ công** (user sẽ test khi có thời gian).  
> Cập nhật: 2026-06-18 — ưu tiên Hướng A nằm ở **#5** trong [`README.md`](../../README.md) mục *Việc tiếp theo*.

Tính năng đã implement: `user_mask` trong `edit_image` + tab **"Xóa vật thể (khoanh vùng)"**
trong `scripts/app_gradio.py`. Báo cáo kỹ thuật: [`report.md`](./report.md).

---

## 1. Tóm tắt đã làm & giới hạn model

| Ca | Kết quả | Ghi chú |
|----|---------|---------|
| Xóa headphones (mèo, mask ~18%) | ✅ Tốt | Ảnh mẫu trong `sample_images/headphones_*` |
| Xóa xe đạp (mask ~39%) | ⚠️ Còn sót | Vật quá lớn — giới hạn SwiftEdit |
| Xóa cây (mask ~19%, không đè người) | ⚠️ Khá | Nền hơi mờ; prompt đúng giúp cải thiện |
| Mask đè lên **đầu người** | ❌ Ghost xanh | SwiftEdit không xóa được phần cơ thể người |
| Xóa **chữ trên banner** (tab prompt) | ❌ Đốm xanh | Diffusion không vẽ/xóa typography; cần LaMa |

**Kết luận ngắn:** SwiftEdit phù hợp vật rời nhỏ/vừa (headphones, chai, biển báo…).  
Không phù hợp: xóa chữ, vật chiếm phần lớn khung, mask đè lên người.

---

## 2. Hai hướng tiếp theo (chọn khi có thời gian)

### Hướng A — Chứng minh tính năng chạy đúng (ưu tiên nếu chỉ test demo)

**Mục tiêu:** Xác nhận UI + `user_mask` hoạt động trên ảnh **phù hợp**, bổ sung ví dụ thành công vào báo cáo.

**Ảnh gợi ý:**

- PIE-Bench delete: `data/PIE-Bench-subset20/annotation_images/3_delete_object_80/311000000001.jpg` (headphones — đã có mẫu)
- Tự chụp/tải: chai nước, lon, biển báo, túi xách, người phụ xa trong ảnh phong cảnh

**Checklist kiểm tra:**

- [ ] Mở `python scripts/app_gradio.py` → tab **"Xóa vật thể (khoanh vùng)"** (không dùng tab prompt cho xóa có kiểm soát)
- [ ] Tô **chỉ vật cần xóa**, không đè lên người/vật quan trọng khác
- [ ] Source prompt mô tả **toàn bộ ảnh gốc** (không chỉ vật cần xóa)
- [ ] Edit prompt mô tả **nền thay thế cụ thể** (vd `empty asphalt road`, `blue sky and green foliage`)
- [ ] Tham số mặc định tab xóa: `scale_edit=0`, `scale_non_edit=1.2`
- [ ] Chụp/lưu 2–3 cặp source/mask/result thành công → `experimental_data/object_removal_2026-06-14/sample_images/`
- [ ] Ghi 1–2 failure case (nếu có) vào báo cáo

**Lệnh self-test nhanh (không cần UI):**

```bash
cd /Users/nguyenkz/Documents/code/CS2309.CH201
source .venv/bin/activate
python scripts/app_gradio.py --selftest-removal \
  data/PIE-Bench-subset20/annotation_images/0_random_140/000000000000.jpg
```

---

### Hướng B — LaMa baseline (nếu cần xóa chữ / vật sạch hơn)

**Mục tiêu:** So sánh học thuật SwiftEdit vs inpainting chuyên dụng (RQ14 trong đề tài).

**Phạm vi implement (khi làm):**

1. Cài LaMa (hoặc wrapper nhẹ) — chạy offline, không gắn realtime pipeline
2. Cùng mask người dùng vẽ → chạy LaMa vs SwiftEdit trên 5–10 ảnh
3. Bảng so sánh: chất lượng vùng lấp (human rating), runtime, failure cases
4. Kết luận báo cáo: *SwiftEdit nhanh + text-guided; LaMa tốt hơn khi vùng xóa lớn hoặc có chữ*

**Ảnh test ưu tiên cho hướng B:**

- Banner có chữ (ca user đã thử — kỳ vọng LaMa sạch hơn)
- Vật lớn (xe đạp) — so sánh sót vật / nền méo

**Checklist (chưa làm — đánh dấu khi bắt đầu):**

- [ ] Thêm dependency / script `scripts/run_lama_baseline.py` (hoặc notebook)
- [ ] Chạy cùng mask với SwiftEdit trên ≥5 ảnh
- [ ] Lưu `experimental_data/object_removal_lama_YYYY-MM-DD/`
- [ ] Cập nhật `report.md`, README §4d, NHAT_KY

---

## 3. Mẹo dùng UI (tránh kết quả tệ)

| Sai | Đúng |
|-----|------|
| Tab "Chỉnh sửa bằng prompt" để xóa vùng cụ thể | Tab **"Xóa vật thể (khoanh vùng)"** + tô mask |
| Source prompt = "A tree" | Mô tả cả ảnh: người + bối cảnh + vật thể |
| Edit prompt = "empty background" | Mô tả nền cụ thể: `blue sky and green foliage` |
| Mask đè lên đầu/tóc người | Chỉ tô vật tách rời hoặc nền phía sau |
| Kỳ vọng xóa sạch chữ "NGHIỆP" trên banner | Chấp nhận giới hạn hoặc chuyển sang LaMa (hướng B) |

**Thời gian:** lần chạy đầu trên MPS ~20–35s (warmup); lần sau ~5–7s (fp16 + cache).

---

## 4. Ca user đã thử (2026-06-14) — ghi nhận

| # | Mô tả | Tab | Kết quả | Nguyên nhân chính |
|---|--------|-----|---------|-------------------|
| 1 | Xóa cây, mask ~19% | Xóa vật thể | Khá | Prompt ban đầu sai; sửa prompt cải thiện |
| 2 | Mask đè đầu người | Xóa vật thể | Ghost xanh | Mask lẫn cơ thể người |
| 3 | Banner áo dài, mask ~6% | Xóa vật thể | Cartoon hóa | Edit prompt mơ hồ + source prompt sai màu áo |
| 4 | Banner, tab prompt | Chỉnh sửa prompt | Đốm xanh | Xóa chữ — ngoài khả năng SwiftEdit |

---

## 5. Cải tiến UI tùy chọn (chưa implement)

- [ ] Slider **mask dilation** (nới rìa 2–10 px) giảm viền mờ
- [ ] Placeholder/gợi ý prompt theo tab xóa vật thể
- [ ] Cảnh báo khi mask >30% khung hoặc overlap vùng trung tâm ảnh

---

## 6. Liên kết

- Báo cáo thực nghiệm: [`report.md`](./report.md)
- Hướng phát triển §3.2: [`HUONG_PHAT_TRIEN.md`](../../HUONG_PHAT_TRIEN.md)
- Demo Gradio: [`scripts/app_gradio.py`](../../scripts/app_gradio.py)
- Đề tài RQ14 (SwiftEdit vs LaMa): [`SwiftEdit_DeTai_CS2309.md`](../../SwiftEdit_DeTai_CS2309.md) §5.6
