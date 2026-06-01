# Map câu hỏi → mục QA.md (1–8)

Agent chọn **một** section dựa trên từ khóa. Nếu không chắc → **8** (Chưa phân loại).

| Section | Mục | Từ khóa gợi ý |
|---|---|---|
| **1** | Diffusion & Text-to-Image | diffusion, denoising, VAE, CLIP, SBv2, SwiftBrush, multi-step, few-step, one-step, text-to-image, latent |
| **2** | SwiftEdit & Pipeline | SwiftEdit, pipeline, overview, TurboEdit, P2P, Prompt-to-Prompt, MasaCtrl, Plug-and-Play, ICD, ReNoise, so sánh method |
| **3** | Inversion & Noise | inversion, DDIM, Null-text, inverted noise, eps_hat, F_theta, stage 1, stage 2, synthetic, CommonCanvas |
| **4** | Mask & ARaM | mask, ARaM, self-guided, s_y, s_edit, s_non-edit, IP-Adapter, cross-attention, attention rescaling |
| **5** | Đánh giá & PieBench | PieBench, PSNR, MSE, CLIP, metrics, IoU, Dice, benchmark, đánh giá |
| **6** | Triển khai Mac / Colab | Mac, M4, MPS, Colab, CUDA, cài đặt, conda, weights, OOM, runtime, GPU |
| **7** | Chỉnh sửa & Style | style editing, semantic edit, prompt, watercolor, anime, failure case, ảnh VN |
| **8** | Chưa phân loại | Không khớp mục trên / câu hỏi chung chung |

## Tags gợi ý theo section

| Section | Tags mặc định |
|---|---|
| 1 | `#diffusion` `#one-step` |
| 2 | `#swiftedit` `#pipeline` |
| 3 | `#inversion` `#noise` |
| 4 | `#aram` `#mask` `#attention` |
| 5 | `#piebench` `#metrics` |
| 6 | `#mac` `#colab` |
| 7 | `#style-editing` `#prompt` |
| 8 | `#swiftedit` |

## Khi nào KHÔNG thêm QA

- User hỏi thao tác code thuần (fix bug, chạy lệnh) — không phải khái niệm
- User yêu cầu sửa file / commit / setup — dùng skill khác
- Câu hỏi trùng đã có trong QA.md (trừ khi `--force`)

## Khi nào BẮT BUỘC thêm QA

- User hỏi *"... là gì?"*, *"giải thích ..."*, *"khác nhau thế nào?"* về CV/SwiftEdit
- User nói: *"thêm vào QA"*, *"ghi QA.md"*, *"lưu câu hỏi này"*
- Agent vừa trả lời giải thích khái niệm dài (>3 câu) trong session đề tài CS2309
