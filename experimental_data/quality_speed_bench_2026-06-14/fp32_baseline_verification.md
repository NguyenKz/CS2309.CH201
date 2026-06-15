# Kiểm chứng: baseline fp32 có đúng là SwiftEdit gốc không?

**Mục tiêu:** đảm bảo `baseline_fp32` (ground truth của benchmark) chạy đúng **thuật toán SwiftEdit gốc**, không bị "thêm thắt" gì làm sai lệch khi so với bản cải thiện fp16+cache.

**Cách làm:** clone bản gốc và diff từng file nguồn.

```bash
git clone --depth 1 https://github.com/Qualcomm-AI-research/SwiftEdit.git /tmp/swiftedit_upstream
# diff từng file: infer.py, models.py, src/*.py  (xem upstream_diff.md cho diff đầy đủ)
```

- **Upstream:** `github.com/Qualcomm-AI-research/SwiftEdit` (đúng repo ghi trong `SwiftEdit/README.md`).
- **Diff đầy đủ:** [`upstream_diff.md`](./upstream_diff.md).

## 1. Kết luận

> **baseline fp32 tương đương số học với SwiftEdit gốc.** Mọi thay đổi mình thêm vào hoặc **không hoạt động** ở luồng fp32 (cache/user_mask/return_noise_image đều tắt), hoặc là **no-op số học** khi dtype=fp32 (ép `.float()` trên tensor vốn đã fp32, so sánh `>` vectorized trùng hệt `to_binary`). Thay đổi thực chất duy nhất về **nguồn trọng số** là dùng **mirror SD2.1-base đã verify trùng trọng số** (vì repo gốc của Stability AI đã bị gỡ) — áp dụng cho **cả** baseline lẫn bản cải thiện nên không làm lệch phép so sánh.

## 2. Các file `src/` (IP-Adapter / attention)

| File | Khác biệt |
|------|-----------|
| `src/__init__.py` | **Giống hệt** |
| `src/attention_processor.py` | **Giống hệt** |
| `src/mask_attention_processor.py` | **Giống hệt** |
| `src/mask_ip_controller.py` | Chỉ thêm `mask.to(<dtype>)` (4 chỗ) để khớp dtype khi chạy fp16. Ở fp32 là **no-op** (cast fp32→fp32). |

## 3. `infer.py` — các thay đổi và ảnh hưởng tới fp32

| Thay đổi | Có hoạt động ở fp32? | Ảnh hưởng ảnh edited fp32 |
|----------|----------------------|---------------------------|
| Thêm import `numpy`, `F`, `StageTimer` | — | Không |
| `prepare_user_mask`, `EditCache`, `get_device` (hàm mới) | `user_mask=None`, `cache=None` ở baseline | Không — không được gọi |
| `"cuda"` → `device` động | Trên Colab `device="cuda"` | Không (cùng giá trị) |
| `vae.encode(... .to(weight_dtype))` → `.to(vae.dtype)` | fp32: cả hai = `float32` | Không |
| `unet_inverse(...).sample.to("cuda", weight_dtype)` → `.sample.float()` | fp32: `weight_dtype=float32` ⇒ `.float()` trùng | Không |
| Nhị phân hóa mask: `.cpu().apply_(to_binary)` → `(mask12 > threshold)` vectorized | Luôn chạy | **Không** — `to_binary` gốc cũng là `pix > threshold`; so sánh `>` là chính xác bit-for-bit, chỉ khác chỗ chạy CPU↔GPU |
| `gen_img(..., return_noise_image=False, timer, embed_cache=None)` | baseline truyền `cache=None` ⇒ embed_cache=None | Không (xem mục 4) |

`StageTimer` chỉ đo thời gian (context manager bao quanh), **không** đụng vào tensor.

## 4. `models.py` — các thay đổi và ảnh hưởng tới fp32

| Thay đổi | Có hoạt động ở fp32? | Ảnh hưởng ảnh edited fp32 |
|----------|----------------------|---------------------------|
| `resolve_dtype`, `module_dtype`, `_NullTimer` (helper mới) | `resolve_dtype("fp32")=float32` | Không |
| Tham số `channels_last` (mặc định `False`) | baseline `False` | Không — không đổi memory format |
| `weight_dtype` qua `resolve_dtype` thay if/elif | fp32 ⇒ `float32` (như cũ) | Không |
| AuxiliaryModel `.to(dtype=torch.float32)` → `.to(dtype=self.weight_dtype)` | mặc định `dtype="fp32"` ⇒ `float32` | Không |
| `get_image_embeds`: cast `float32` → `enc_dtype`/`proj_dtype` | fp32: hai dtype này = `float32` | Không |
| `torch.load(ip)` → `torch.load(ip, map_location="cpu", weights_only=True)` | Luôn | Không — cùng trọng số, chỉ nạp an toàn hơn |
| Khối ép dtype/channels_last trong IPSBV2Model | chỉ chạy khi `weight_dtype != float32` | Không — **bỏ qua ở fp32** |
| `gen_img` cache (`image_prompt_embeds`/`src_text_embed`) | `embed_cache=None` ⇒ dict tạm rỗng ⇒ luôn vào nhánh `else` (tính lại) | Không — đường đi trùng bản gốc |
| `noise.float().to(unet_dtype)`, `model_pred.float()` trước hậu xử lý | fp32: đã là `float32` | Không |
| `return_noise_image=False` (bỏ decode noise) | baseline `False` | Không — `edit_image` lấy `res_gen_img, _` (vứt `noise_image`); ảnh edited tính y hệt |

**Hậu xử lý số học** (`pred_original_sample = (noise - sigma_t*model_pred)/alpha_t`, thresholding/clip, VAE decode) **giữ nguyên** thứ tự và công thức như bản gốc.

## 5. Thay đổi thực chất duy nhất: nguồn trọng số SD2.1-base

- Bản gốc: `model_name="stabilityai/stable-diffusion-2-1-base"`.
- Bản mình: `model_name="Manojb/stable-diffusion-2-1-base"` (nạp scheduler + VAE + tokenizer + text_encoder; `image_encoder` vẫn từ `h94/IP-Adapter` gốc).

**Lý do:** repo `stabilityai/stable-diffusion-2-1-base` đã bị Stability AI set **private/deprecated** cuối 2025, không tải được nữa.

**Vì sao không làm sai lệch:**
- `Manojb/stable-diffusion-2-1-base` ghi rõ *"Cloned from stabilityai/stable-diffusion-2-1-base"* (verified, HF), **trọng số y hệt** — đã thành mirror chuẩn được nhiều paper khác dùng thay thế (TDSM ICCV'25, Stable-Virtual-Camera, Click2Mask AAAI'25).
- Mirror này được dùng cho **cả** `baseline_fp32` **lẫn** `improved_fp16_cache`, nên dù có chênh (không có) cũng triệt tiêu trong phép so sánh tương đối.
- Các checkpoint lõi của SwiftEdit (inverse UNet, SBv2 UNet, IP-Adapter) **không đổi** — vẫn nạp từ `swiftedit_weights/` của Qualcomm.

## 6. Tóm tắt

- `src/*`: giống gốc (trừ cast dtype no-op ở fp32).
- `infer.py` / `models.py`: mọi nhánh tối ưu **tắt hoặc no-op** ở fp32 ⇒ baseline = thuật toán gốc.
- Nguồn SD2.1-base: mirror trùng trọng số (bắt buộc do repo gốc bị gỡ).

⇒ **Ground truth fp32 đáng tin để làm mốc đánh giá fp16+cache.**
