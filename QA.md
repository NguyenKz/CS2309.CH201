# Q&A — Câu hỏi & Giải đáp khái niệm

> Ghi lại các câu hỏi về khái niệm, thuật ngữ, kỹ thuật trong quá trình làm đề tài **SwiftEdit / CS2309.CH201**.  
> Skill Cursor `update-qa` **tự thêm entry** sau khi bạn hỏi khái niệm (hoặc nhắn *"thêm vào QA.md"*).

Tài liệu liên quan: [Overview](./SwiftEdit_Overview.md) · [Đề tài](./SwiftEdit_DeTai_CS2309.md) · [README](./README.md)

---

## Cách dùng

1. Hỏi trong chat Cursor (vd: *"Diffusion inversion là gì?"*).
2. Agent trả lời và **tự ghi vào QA.md** (skill `update-qa`), hoặc nhắn: *"Thêm câu hỏi này vào QA.md"*.
3. Entry mới đặt **ở đầu mục** tương ứng; mục lục cập nhật số câu hỏi tự động.

**Mẫu một entry:**

```markdown
### Q: [Câu hỏi ngắn gọn]?

**Ngày:** YYYY-MM-DD  
**Chủ đề:** #diffusion / #swiftedit / #aram / …

**Trả lời (tóm tắt):**
- …

**Ghi chú thêm / link:**
- …
```

---

## Mục lục nhanh

| # | Chủ đề | Số câu hỏi |
|---|---|---|
| 1 | [Diffusion & Text-to-Image](#1-diffusion--text-to-image) | 10 |
| 2 | [SwiftEdit & Pipeline](#2-swiftedit--pipeline) | 7 |
| 3 | [Inversion & Noise](#3-inversion--noise) | 15 |
| 4 | [Mask & ARaM](#4-mask--aram) | 4 |
| 5 | [Đánh giá & PieBench](#5-đánh-giá--piebench) | 10 |
| 6 | [Triển khai Mac / Colab](#6-triển-khai-mac--colab) | 13 |
| 7 | [Chỉnh sửa & Style](#7-chỉnh-sửa--style) | 2 |
| 8 | [Chưa phân loại](#8-chưa-phân-loại) | 2 |

*(Cập nhật cột "Số câu hỏi" khi thêm entry.)*

---

## 1. Diffusion & Text-to-Image

*Khái niệm nền: diffusion model, one-step vs multi-step, SBv2, CLIP, VAE, …*

<!-- qa:insert -->
### Q: Vì sao noise ε phải ~ N(0,I) chứ không ngẫu nhiên phân phối khác?

**Ngày:** 2026-07-31  
**Chủ đề:** #gaussian #noise #diffusion #sbv2 #stage1

**Trả lời (tóm tắt):**
- SBv2/diffusion train với giả định ε ~ N(0,I); F_theta phải dùng cùng loại nhiễu để khớp generator.
- Chọn Gaussian vì toán DDPM, CLT, và chuẩn SD/SBv2 — đổi phân phối phải train lại generator.
- Không phải ngẫu nhiên tùy ý. feedback.md F1.


### Q: Có trường hợp đường chéo covariance không bằng 1 không?

**Ngày:** 2026-07-31  
**Chủ đề:** #identity #covariance #variance #gaussian

**Trả lời (tóm tắt):**
- Có — nhưng lúc đó không còn là ma trận đơn vị I. I luôn đường chéo = 1 theo định nghĩa.
- N(0, σ²I): đường chéo = σ² (có thể ≠ 1), vẫn độc lập.
- N(0, Σ) tổng quát: diag tùy chọn (vd. 2, 0.5, …); ngoài đường chéo ≠ 0 thì có tương quan.
- Diffusion/SwiftEdit mặc định vẫn ε ~ N(0, I). feedback.md F1.


### Q: Ma trận đơn vị I là gì?

**Ngày:** 2026-07-31  
**Chủ đề:** #identity #matrix #linear-algebra #notation

**Trả lời (tóm tắt):**
- Ma trận vuông như số 1 trong phép nhân: I·x = x.
- Đường chéo toàn 1, ngoài đường chéo toàn 0. Ký hiệu I hoặc I_n.
- Ví dụ 2×2: [[1,0],[0,1]]. Trong N(0,I): dùng I làm covariance (mỗi chiều variance 1, độc lập).
- feedback.md F1.


### Q: N(0, I) ghi I thì phương sai = 1 lấy từ đâu?

**Ngày:** 2026-07-31  
**Chủ đề:** #identity #covariance #variance #notation

**Trả lời (tóm tắt):**
- I = ma trận đơn vị: đường chéo toàn 1, ngoài đường chéo = 0.
- Covariance: Σ_ii = phương sai chiều i → I_ii = 1 ⇒ variance = 1.
- Σ_ij (i≠j) = 0 ⇒ các chiều không tương quan.
- Số 1 nằm trong định nghĩa I, không viết cạnh chữ I trên hình. N(0,I) gọn hơn N(0, diag(1,1,…)).
- feedback.md F1.


### Q: Mean và covariance trong N(0, I) là gì?

**Ngày:** 2026-07-31  
**Chủ đề:** #mean #covariance #gaussian #notation #diffusion

**Trả lời (tóm tắt):**
- Mean (μ): trung bình / kỳ vọng — “tâm” phân phối. N(0,I): mean=0 → nhiễu không lệch ±.
- Covariance (Σ): ma trận độ lan + tương quan giữa các chiều. N(0,I): Σ=I → mỗi chiều độc lập, phương sai 1 (nhiễu trắng).
- 1D gần với N(0,1); latent nhiều chiều = mỗi phần tử ~ N(0,1) độc lập.
- Chi tiết: feedback.md F1.


### Q: Làm sao biết N trong ε ~ N(0, I) là phân phối Gaussian?

**Ngày:** 2026-07-31  
**Chủ đề:** #notation #gaussian #normal #probability #diffusion

**Trả lời (tóm tắt):**
- Ký hiệu toán xác suất chuẩn: X ~ N(μ, Σ) = X theo phân phối Normal (= Gaussian).
- N = Normal; cặp (0, I) = mean 0, covariance đơn vị — đúng khuôn mẫu Normal (Uniform thường viết U(a,b)).
- Diffusion (DDPM/SD/SBv2) mặc định noise ε ~ N(0, I); Fig. 2 SwiftEdit dùng lại convention đó.
- Không phải chữ tắt riêng của paper. Xem feedback.md F1.


### Q: SBv2 lấy eps (noise) ở đâu?

**Ngày:** 2026-06-19  
**Chủ đề:** #sbv2 #eps #noise #swiftedit #f_theta

**Trả lời (tóm tắt):**
- Tùy ngữ cảnh — SBv2 không tự sinh eps; eps/noise là đầu vào truyền vào gen_img(noise=...).
- Sinh ảnh mới (text-to-image): eps ~ torch.randn(...) trong latent space — random Gaussian, giống bắt đầu sample DDPM/DDIM.
- SwiftEdit editing (infer.py): eps đến từ F_theta — unet_inverse(latent, prompt) → inverted_noise_1 (= eps_hat). Trộn: input_sb = alpha_t*latent + sigma_t*inverted_noise_1 rồi truyền vào gen_img(noise=input_sb).
- Train Stage 1 (tạo data): random eps → SBv2 sinh ảnh synthetic → lưu eps làm nhãn cho F_theta.
- Tóm lại: sinh mới = random; edit = F_theta invert ảnh nguồn; không lấy từ DDIM multi-step.

**Ghi chú thêm / link:**
- Code: infer.py dòng input_sb + gen_img(noise=...); models.py IPSBV2Model.gen_img.


### Q: eps (epsilon) trong diffusion và SwiftEdit là gì?

**Ngày:** 2026-06-19  
**Chủ đề:** #diffusion #noise #epsilon #eps_hat #swiftedit

**Trả lời (tóm tắt):**
- eps (ε, epsilon) = tensor nhiễu Gaussian ngẫu nhiên, thường eps ~ N(0,1) — thành phần random trong công thức forward diffusion.
- Công thức DDPM (bước 01): x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps. eps càng lớn (t lớn) thì ảnh càng nhiễu; t=T gần như toàn noise.
- Train DDPM (bước 02): random eps, trộn vào ảnh → x_t; UNet học predict eps (hoặc x_0). Loss: predicted ≈ eps thật.
- Sample (bước 03–04): bắt đầu từ eps random thuần → denoise dần → ảnh sạch.
- SwiftEdit/SBv2: eps nằm trong latent space (vd 4×64×64), không phải pixel. SBv2 nhận eps + prompt → one-step → ảnh.
- eps_hat: inverted noise do F_theta dự đoán từ latent ảnh + prompt — noise mà SBv2 cần để tái tạo/sửa ảnh. Khác eps train DDPM: eps_hat là output invert, không phải random mỗi lần (trừ khi sinh ảnh mới từ đầu).

**Ghi chú thêm / link:**
- Học hands-on: learn/diffusion-from-scratch bước 01–02. SwiftEdit: infer.py inverted_noise, models.py gen_img(noise=...).


### Q: Công thức q_sample x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*epsilon từ đâu ra?

**Ngày:** 2026-06-19  
**Chủ đề:** #diffusion #forward-process #q_sample #ddpm

**Trả lời (tóm tắt):**
- DDPM định nghĩa từng bước: x_t = sqrt(alpha_t)*x_{t-1} + sqrt(1-alpha_t)*eps_t (alpha_t = 1 - beta_t).
- Khai triển x_2, x_3... xuất hiện tích alpha_1*alpha_2*... → đặt alpha_bar_t.
- Nhiều Gaussian độc lập có trọng số gộp thành một eps ~ N(0,1) → sqrt(1-alpha_bar_t)*eps.
- Kết quả đóng: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon.
- Train không cần loop 1000 bước — q_sample nhảy thẳng x_0 → x_t.

**Ghi chú thêm / link:**
- Giải thích đầy đủ: [learn/diffusion-from-scratch/README.md § Công thức forward q_sample](learn/diffusion-from-scratch/README.md#công-thức-forward-q_sample-từ-đâu-ra)


### Q: Mỗi step trong diffusion chỉnh sửa ảnh cũ làm gì?

**Ngày:** 2026-06-01  
**Chủ đề:** #diffusion #one-step #inversion #pipeline

**Trả lời (tóm tắt):**
- Ảnh tưởng tượng: bức tranh bị phủ sương mù dày; mỗi step = lau bớt một lớp sương nhỏ.
- Mỗi step: mạng UNet nhìn ảnh hiện tại + prompt + "đang ở bước mấy", dự đoán nên bớt nhiễu thế nào → ảnh rõ hơn một chút.
- Sinh ảnh mới: bắt đầu từ nhiễu ngẫu nhiên, lặp ~50 step → ảnh khớp prompt.
- Inversion (giai đoạn 1): có ảnh thật, chạy ngược — mỗi step tìm "thêm nhiễu kiểu gì" để quay lại trạng thái trước, cuối cùng ra noise khởi đầu.
- Editing (giai đoạn 2): từ noise đó, lại lau sương ~50 step nhưng prompt mới + chỉnh attention để đổi nội dung (vd mèo cam→đen) mà giữ bố cục.
- 1 step = 1 lần UNet chạy (tốn GPU); 50+50 step ≈ 100 lần chạy → chậm.




---

## 2. SwiftEdit & Pipeline

*SwiftEdit là gì, input/output, so sánh P2P / TurboEdit, tại sao one-step nhanh, …*

<!-- qa:insert -->
### Q: EditCache cache chỗ nào (thành phần nào)?

**Ngày:** 2026-07-24  
**Chủ đề:** #editcache #cache #latent #clip #swiftedit-rt

**Trả lời (tóm tắt):**
- Khóa cache: cùng đường dẫn ảnh + cùng source prompt; đổi ảnh hoặc source → invalidate.
- Cache được (tính 1 lần, tái dùng khi hit):
- 1) VAE latent của ảnh nguồn (encode ảnh → latents)
- 2) Source text embed cho InverseModel (CLIP text của source prompt)
- 3) CLIP image embed / image_prompt_embeds (IP-Adapter) + source text embed bên gen (gen_embed_cache)
- Không cache / tính lại mỗi lần: text embed của edit prompt; UNet inversion (phụ thuộc edit); generation one-step với edit; mask ARaM.
- Mục đích: cùng ảnh + nhiều edit prompt khác nhau thì bỏ VAE encode + CLIP image + encode source text — giảm Miss→Hit latency.


### Q: Ý tưởng cốt lõi SwiftEdit: thay random noise bằng noise từ ảnh gốc?

**Ngày:** 2026-06-19  
**Chủ đề:** #swiftedit #pipeline #one-step #f_theta #sbv2 #core-idea

**Trả lời (tóm tắt):**
- Đúng khung: SBv2 T2I = random noise + prompt → ảnh (1 bước). SwiftEdit = thay random noise bằng inverted noise từ ảnh gốc → pipeline edit 1+1 bước (F_theta invert + SBv2 denoise) thay ~50+50 bước pipeline cũ.
- Bổ sung: (1) noise qua F_theta(VAE(ảnh), src_prompt), không DDIM inversion. (2) SBv2 edit nhận input_sb=alpha*z+sigma*noise_f_src, không noise_f thuần. (3) Cần dst_prompt + IP-Adapter + ARaM(mask từ |noise_f_src-noise_f_dst|), không chỉ đổi prompt. (4) F_theta phải train Stage 1+2 trước — không plug-and-play.
- Câu báo cáo: F_theta (1 bước) suy inverted noise từ ảnh gốc; SBv2 (1 bước) denoise với edit prompt, IP-Adapter, ARaM.

**Ghi chú thêm / link:**
- SwiftEdit_Overview.md §3–4; QA mục 2 (SwiftEdit vs DDIM).


### Q: Tóm tắt notation SwiftEdit: Stage 1, Stage 2 train, và inference edit?

**Ngày:** 2026-06-19  
**Chủ đề:** #swiftedit #notation #stage1 #stage2 #inference #f_theta #sbv2

**Trả lời (tóm tắt):**
- Ký hiệu: noise_svb=noise random SBv2; Img_svb=SBv2(noise_svb,src_prompt); z=VAE encode; noise_f=F_theta(z,prompt).
- STAGE 1 TRAIN (synthetic): noise_svb+src_prompt→SBv2→Img_svb→z; F_theta(z,src_prompt)→noise_f. Loss: |noise_svb-noise_f| + recon SBv2(noise_f,src_prompt)≈Img_svb. Chỉ train F_theta.
- STAGE 2 TRAIN (ảnh thật): Img_real+caption→z→F_theta→noise_f→recon→Img_recon. Loss: DISTS(Img_real,Img_recon)+L_reg. KHÔNG dùng dst_prompt; mục tiêu tái tạo ảnh gốc, không train edit.
- INFERENCE EDIT: z=VAE(Image_src); noise_f_src=F_theta(z,src_prompt); noise_f_dst=F_theta(z,dst_prompt); mask=|src-dst|; input_sb=alpha*z+sigma*noise_f_src; dst_img=SBv2(input_sb, dst_prompt, IP-Adapter(Image_src), ARaM(mask)). Không phải SBv2(noise_f thuần).

**Ghi chú thêm / link:**
- Code: SwiftEdit/infer.py. So sánh QA mục 3 (F_theta training chi tiết).


### Q: SwiftEdit cải tiến chỗ nào mà nhanh hơn DDIM/DDPM?

**Ngày:** 2026-06-19  
**Chủ đề:** #swiftedit #pipeline #ddim #ddpm #one-step #inversion #f_theta

**Trả lời (tóm tắt):**
- SwiftEdit nhanh không phải vì DDIM tốt hơn DDPM — nó đổi cả pipeline chỉnh sửa: từ ~100 lần chạy UNet (50 inversion + 50 edit) xuống 2 forward (1 inversion + 1 edit).
- Pipeline cũ có 2 giai đoạn độc lập: inversion (DDIM ngược, ảnh→noise) rồi sampling+edit (denoise đa bước + P2P). DDIM chỉ rút bước nửa sau (sinh ảnh); editing vẫn phải inversion đắt trước.
- Cải tiến 1 — one-step inversion F_theta: latent z + prompt → eps_hat trong 1 pass, thay chuỗi DDIM inversion; không optimize từng ảnh như Null-text.
- Cải tiến 2 — one-step editing: SwiftBrushv2 (distill one-step) + IP-Adapter G_IP + ARaM; thay vòng lặp p_sample/ddim_sample với edit prompt.
- Cải tiến 3 — self-guided mask M = |eps_hat_source - eps_hat_edit| tận dụng ngay sau F_theta, không cần segmentation model thêm.
- Số liệu PieBench (paper): DDIM+P2P ~26s; TurboEdit 4+4 ~1.3s; SwiftEdit 1+1 ~0.23s.
- Liên hệ học diffusion-from-scratch: q_sample là forward 1 công thức; inversion cũ phải chạy ngược nhiều bước; SwiftEdit học F_theta + dùng backbone one-step thay cả hai vòng lặp.

**Ghi chú thêm / link:**
- Chi tiết: SwiftEdit_Overview.md §3 (F_theta, ARaM), §6 (bảng so sánh); SwiftEdit_DeTai_CS2309.md §1.3; learn/diffusion-from-scratch bước 03–04 (DDPM/DDIM chỉ là sampling, không phải editing pipeline).


### Q: Source prompt có bắt buộc tự viết không?

**Ngày:** 2026-06-01  
**Chủ đề:** #prompt #swiftedit #pipeline

**Trả lời (tóm tắt):**
- Không bắt buộc — optional, khuyến nghị có.
- Code SwiftEdit (infer.py): src_p "could leave it empty".
- Edit prompt là bắt buộc; source mô tả ảnh gốc, hỗ trợ inversion + self-guided mask (so sánh eps_hat source vs edit).
- Không cần câu dài như PieBench: demo dùng ngắn ("woman", "dog"); câu đầy đủ ("An orange cat...") thường cho mask/reconstruction tốt hơn khi sửa cục bộ.
- Có thể caption tự động (BLIP/LLaVA) rồi chỉnh tay — SwiftEdit không tự sinh source prompt.


### Q: Có ảnh minh họa pipeline inversion + editing diffusion không?

**Ngày:** 2026-06-01  
**Chủ đề:** #pipeline #inversion #swiftedit #diffusion

**Trả lời (tóm tắt):**
- Có — mục 1.3.3 SwiftEdit_DeTai_CS2309.md và §2.2 SwiftEdit_Overview.md.
- Ảnh lưu local assets/pipeline/ (hiển thị trong Cursor MD preview).
- Toàn pipeline: sage-fig3-pipeline.png (SAGE Fig. 3).
- Inversion: nulltext-diagram.png.
- Mỗi step: sage-fig4-denoise-steps.png; tutorial HF DDIM Inversion.
- Editing: p2p-cross-attention.png.
- Kết quả: p2p-teaser.png, nulltext-editing-results.png.


### Q: Tại sao các phương pháp diffusion chỉnh sửa ảnh cũ lại tốn nhiều step?

**Ngày:** 2026-06-01  
**Chủ đề:** #swiftedit #pipeline #diffusion #one-step #inversion

**Trả lời (tóm tắt):**
- Pipeline kinh điển gồm 2 giai đoạn độc lập, mỗi giai đoạn đều multi-step: inversion (20–50+) rồi sampling+edit (20–50+).
- Mỗi step = 1 forward pass UNet lớn trên latent → chi phí tính toán nhân đôi so với số step.
- Inversion (DDIM/Null-text): chạy ngược quá trình denoise để tìm noise ban đầu khớp ảnh nguồn; Null-text thêm optimize từng ảnh.
- Editing (P2P, MasaCtrl, PnP): chạy lại denoise đa bước, can thiệp cross/self-attention mỗi step để đổi prompt nhưng giữ layout.
- Các method này xây trên SD multi-step (50 step huấn luyện) — không thể rút còn 1 step mà không distill/học riêng như SBv2.
- Few-step (TurboEdit, ICD) giảm còn 3–8 step nhưng vẫn 2 giai đoạn; SwiftEdit: 1 step inversion + 1 step edit.




---

## 3. Inversion & Noise

*DDIM inversion, Null-text, one-step inversion `F_theta`, inverted noise `eps_hat`, stage 1/2 training, …*

<!-- qa:insert -->
### Q: Fig. 2 sau ảnh synthetic: VAE/z, F_theta, IP-Adapter học gì? Vì sao Stage 2 freeze IP?

**Ngày:** 2026-07-31  
**Chủ đề:** #fig2 #vae #ip-adapter #f-theta #stage2

**Trả lời (tóm tắt):**
- VAE (freeze): x → z latent. F_θ: (z,c_y)→ε̂ ≈ ε đầu vào SBv2; cùng prompt với G.
- IP (Stage1 lửa): cổng phụ trong UNet; train projection+W^K_x/W^V_x; h=Attn_text+s_x·Attn_image. Học bám ảnh gốc khi recon — không thay F_θ.
- Stage1: F_θ+IP cùng lúc. Stage2: chỉ F_θ, IP freeze (giữ prior; tránh overfit; đóng domain gap ảnh thật).
- feedback.md F1 (ngắn).


### Q: Fig. 2 Stage 1 không vẽ prompt: G(.) nhận gì? Công thức ghi ở đâu?

**Ngày:** 2026-07-31  
**Chủ đề:** #fig2 #sbv2 #prompt #equation5 #stage1

**Trả lời (tóm tắt):**
- G = SBv2 (one-step T2I). Prompt có: c_y từ caption JourneyDB; Fig. 2 thường省略 mũi tên text.
- Paper Sec. 4.1 Eq. (5): ε ~ N(0,1), z = G(ε, c_y) — không trên mũi tên hình.
- G ra latent z; ảnh x trên Fig. 2 là sau VAE decode (~512×512). x̂=D(G(...)) là đường đủ tới pixel; Stage 2 nêu D rõ hơn.
- feedback.md F1.


### Q: Stage 1: ε là 1 tensor latent hay noise ngẫu nhiên?

**Ngày:** 2026-07-31  
**Chủ đề:** #epsilon #noise #latent #stage1

**Trả lời (tóm tắt):**
- Cả hai: ε là một tensor trong latent space; giá trị được sample ngẫu nhiên từ N(0,I) mỗi bước train.
- Không phải tensor cố định dùng mãi; cũng không phải nhiễu trên pixel RGB.
- Mỗi bước: sample ε → SBv2 + prompt → ảnh → F_theta học đoán lại đúng ε đó.
- feedback.md F1.


### Q: Trên Fig. 2 Stage 1, ε ~ N(0, I) thì ε, N, 0, I là gì?

**Ngày:** 2026-07-31  
**Chủ đề:** #notation #gaussian #noise #stage1 #fig2

**Trả lời (tóm tắt):**
- ε (epsilon): tensor noise latent đưa vào SBv2.
- N: phân phối Gaussian (Normal).
- 0: mean = vector 0.
- I: ma trận đơn vị (covariance) — nhiễu độc lập, phương sai 1.
- Cả cụm: lấy mẫu white noise chuẩn nhiều chiều. Stage 1 dùng ε làm nhãn dạy F_theta đảo ảnh → đúng noise.
- Chi tiết: feedback.md F1.


### Q: Nguyên lý train SwiftEdit trên SBv2 là gì?

**Ngày:** 2026-07-31  
**Chủ đề:** #training #sbv2 #f-theta #swiftedit #stage2

**Trả lời (tóm tắt):**
- SBv2 backbone đóng băng (T2I one-step).
- Stage 1 (~40k JourneyDB, ~100k iter): F_theta + IP-Adapter học đảo/recon trên synthetic.
- Stage 2 (~5k CommonCanvas, ~180k iter): chỉ tiếp tục train F_theta; paper freeze IP-Adapter; loss DISTS + L_reg.
- Edit chỉ lúc inference (edit_prompt + mask/ARaM). Đề tài dùng checkpoint pretrained.
- Chi tiết Fig. 2: QA “Fig. 2 paper…” + slide §3a-fig.


### Q: Fig. 2 paper: Stage 1 và Stage 2 chạy song song hay tuần tự? Lửa/tuyết nghĩa là gì? Ai được train?

**Ngày:** 2026-07-31  
**Chủ đề:** #fig2 #training #stage1 #stage2 #frozen #ip-adapter #sbv2

**Trả lời (tóm tắt):**
- Tuần tự: Stage 1 (warm-up synthetic) rồi Stage 2 (continue ảnh thật) — không song song. Caption Fig. 2: warm up → shift/continue.
- Lửa = trainable (nhận gradient); tuyết = frozen — quy ước phổ biến CV/ML (ControlNet, IP-Adapter…), không riêng SwiftEdit.
- Train: F_theta lửa cả 2 stage. SBv2 G / VAE / CLIP = tuyết. IP-Adapter (W^K_x, W^V_x): train Stage 1; Stage 2 paper freeze IP (“train only the inversion network”). Edit/ARaM/mask: không train — chỉ inference.
- Minh chứng trước thầy: chiếu Fig. 2 gốc paper + bảng legend; Mermaid slide chỉ tóm tắt đã đối chiếu.
- Nguồn: arXiv 2412.04301 Sec. 4.1 + Fig. 2.


### Q: Các dataset huấn luyện SBv2 và SwiftEdit tên gì?

**Ngày:** 2026-07-24  
**Chủ đề:** #dataset #journeydb #laion #commoncanvas #piebench #sbv2 #swiftedit

**Trả lời (tóm tắt):**
- SBv2: JourneyDB (prompts), LAION (prompts mở rộng), tuỳ chọn LAION-Aesthetic-6.25+ (cặp image regularization).
- SwiftEdit Stage 1: JourneyDB (caption → synthetic qua SBv2).
- SwiftEdit Stage 2: CommonCanvas (ảnh thật + caption).
- Eval đề tài (không phải train): PIE-Bench / PIE-Bench-auto200.


### Q: Tập dữ liệu huấn luyện SBv2 và SwiftEdit khoảng bao nhiêu?

**Ngày:** 2026-07-24  
**Chủ đề:** #sbv2 #swiftedit #training #dataset #journeydb #commoncanvas #laion

**Trả lời (tóm tắt):**
- SBv2 (SwiftBrush v2, T2I one-step — backbone đóng băng):
- ~1.3M prompts JourneyDB; bản mở rộng thêm ~2M LAION → tổng ~3.3M prompts (image-free distill).
- Tuỳ chọn thêm ~200K cặp LAION-Aesthetic cho image regularization (~5% data).
- SwiftEdit (train F_theta + IP-Adapter; không train lại SBv2):
- Stage 1: ~40k caption JourneyDB → ảnh synthetic từ SBv2; ~100k iter (học đảo noise).
- Stage 2: ~5k ảnh thật CommonCanvas + caption; ~180k iter (DISTS tái tạo).
- Đề tài CS2309 dùng checkpoint pretrained (inverse_ckpt-120k, ip_adapter_ckpt-90k, sbv2_0.5) — không train lại.


### Q: Nguyên lý train SwiftEdit trên SBv2 là gì?

**Ngày:** 2026-07-24  
**Chủ đề:** #training #sbv2 #f-theta #swiftedit

**Trả lời (tóm tắt):**
- SBv2 (SwiftBrushv2) là T2I one-step: noise + prompt → ảnh trong 1 bước; backbone này đóng băng khi train SwiftEdit.
- Stage 1 (~40k caption JourneyDB, ~100k iter): SBv2 sinh ảnh synthetic → F_theta học đảo noise (regression |noise_svb − noise_f|).
- Stage 2 (~5k ảnh CommonCanvas, ~180k iter): F_theta + IP-Adapter học tái tạo ảnh thật (DISTS); vẫn không dạy hành vi edit.
- Edit chỉ lúc inference: đổi edit_prompt + self-guided mask + ARaM trên checkpoint pretrained.
- Đề tài không train lại — dùng inverse_ckpt-120k + sbv2_0.5 + ip_adapter_ckpt-90k.


### Q: Nguyên lý training SwiftEdit trên SBv2 là gì?

**Ngày:** 2026-07-24  
**Chủ đề:** #training #sbv2 #f_theta #swiftedit #stage1 #stage2

**Trả lời (tóm tắt):**
- SBv2 (one-step T2I) đóng băng: dùng để sinh cặp (noise, ảnh) synthetic và làm generator lúc infer.
- Train F_theta + IP-Adapter; không train lại SBv2 backbone.
- Stage 1 (~100k): noise_svb + prompt → SBv2 → ảnh → F_theta học hồi quy noise + recon.
- Stage 2 (~180k): ảnh thật → invert → recon; loss DISTS + L_reg — vẫn tái tạo, không train edit.
- Edit chỉ lúc inference: đổi edit_prompt + mask/ARaM trên checkpoint pretrained.
- Slide: report/SLIDE_SwiftEdit.md §3a+.


### Q: Loss perceptual Stage 2 (DISTS) chấm điểm thế nào?

**Ngày:** 2026-06-19  
**Chủ đề:** #f_theta #stage2 #dists #perceptual #commoncanvas #loss

**Trả lời (tóm tắt):**
- Stage 2 không có eps ground truth (ảnh thật) → không dùng regression ||eps_hat-eps|| như Stage 1.
- Perceptual loss paper: L = DISTS(x, x_hat). x = ảnh thật đầu vào; x_hat = ảnh tái tạo sau chuỗi z=VAE(x) → F_theta→eps_hat → G_IP → VAE decode.
- DISTS (Deep Image Structure and Texture Similarity): metric đã train trên ảnh thật, so cấu trúc + texture — robust hơn MSE/PSNR pixel-wise khi lệch sáng/nhẹ pixel.
- Thêm L_reg (gợi ý SDS): giữ eps_hat gần N(0,I). Chỉ DISTS thì noise có thể ôm pattern ảnh nguồn quá mức → khó edit sau.
- Loss Stage 2 tổng: DISTS(x, x_hat) + λ·L_reg(eps_hat). λ=1 trong paper.
- Bảng so sánh: Stage 1 chấm eps_hat vs eps + pixel recon; Stage 2 chấm x_hat vs x (perceptual) + phân phối noise.

**Ghi chú thêm / link:**
- Paper SwiftEdit Sec. 3 Stage 2; arXiv 2412.04301. Metric eval PieBench khác (PSNR/CLIP) — xem QA mục 5.


### Q: Hiểu tổng quan train F_theta: SBv2 sinh mẫu → train F → fine-tune ảnh thật — đúng chỗ nào?

**Ngày:** 2026-06-19  
**Chủ đề:** #f_theta #stage1 #stage2 #sbv2 #training

**Trả lời (tóm tắt):**
- Đúng: (1) Đã có SBv2 sinh ảnh one-step nhưng editing vẫn pipeline SD multi-step (DDIM inv + sampling). (2) Stage 1 lợi dụng SBv2 sinh synthetic có nhãn eps để train F_theta nhảy thẳng latent+prompt → noise. (3) Stage 2 fine-tune trên ảnh thật để invert ảnh chụp, không chỉ ảnh synthetic.
- Chỉnh: F_theta là UNet cỡ SD (unet_inverse), không phải mạng nhỏ. Không học bắt chước từng bước DDIM — thay thế 50 bước bằng 1 forward.
- Stage 2 data là ảnh thật CommonCanvas + caption — không phải SBv2 sinh thêm mẫu mới; SBv2 vẫn dùng trong loss tái tạo (đưa eps_hat qua generator xem có ra lại ảnh gốc).
- Tóm 3 tầng: engine one-step có sẵn → synthetic có nhãn train invert → ảnh thật fine-tune perceptual.

**Ghi chú thêm / link:**
- SwiftEdit_Overview.md §3.1; SwiftEdit_DeTai_CS2309.md §6.4 (đề tài dùng checkpoint, không train lại).


### Q: Stage 1 train F_theta — giải thích chi tiết 5 bước?

**Ngày:** 2026-06-19  
**Chủ đề:** #f_theta #stage1 #synthetic #sbv2 #training

**Trả lời (tóm tắt):**
- Bước 1 — random eps ~ N(0,1) trong latent space + prompt y (caption). eps là hạt giống; cùng eps khác prompt → ảnh khác.
- Bước 2 — SBv2 frozen: eps + y → UNet one-step (t≈999) → latent sạch → VAE decode → ảnh x. Lưu eps làm ground truth.
- Bước 3 — VAE encode x → latent z (như infer.py). z là input F_theta; có thể lệch nhẹ latent trung gian SBv2 do roundtrip VAE.
- Bước 4 — Train F_theta (unet_inverse): input (z, embedding y) → output eps_hat. Chỉ cập nhật θ của F_theta.
- Bước 5 — Loss A regression: ||eps_hat - eps||². Loss B reconstruction: SBv2(eps_hat, y) ≈ x (ảnh tái tạo phải giống ảnh gốc qua SBv2, không chỉ khớp tensor).
- Lặp ~100k iter với cặp (eps,y) khác nhau. Liên hệ DDPM bước 02: cùng tinh thầu predict noise, nhưng input là latent sạch (invert) thay vì x_t đã nhiễu.

**Ghi chú thêm / link:**
- Code SBv2 forward: SwiftEdit/models.py gen_img(). VAE encode: infer.py vae_encode.


### Q: F_theta có phải học mapping step 0 → step end của pipeline inversion cũ không?

**Ngày:** 2026-06-19  
**Chủ đề:** #f_theta #inversion #ddim #one-step

**Trả lời (tóm tắt):**
- Đúng hướng: DDIM inversion cũ chạy ngược ảnh/latent sạch z (đầu inversion) → inverted noise eps (cuối inversion) trong ~50 bước; F_theta học cùng quan hệ đó nhưng 1 forward: F_theta(z, prompt) → eps_hat.
- Sampling sinh ảnh: noise → ảnh sạch. Inversion edit: ảnh sạch → noise — ngược chiều sampling.
- Không mimic từng bước DDIM: không distill trajectory 50 timestep; thay thế cả chuỗi bằng ánh xạ trực tiếp đã học.
- Stage 1: nhãn eps từ SBv2 synthetic (biết đáp án), không cần chạy DDIM làm teacher.
- Tiêu chí: eps_hat phải dùng được với SBv2 tái tạo ảnh — tương đương kết quả cuối inversion mà pipeline cũ cần.

**Ghi chú thêm / link:**
- So sánh pipeline cũ: QA mục 2 (SwiftEdit vs DDIM). Paper Sec. 3.


### Q: Quy trình tạo ra F_theta (inversion network) như thế nào?

**Ngày:** 2026-06-19  
**Chủ đề:** #f_theta #inversion #training #stage1 #stage2 #synthetic #commoncanvas

**Trả lời (tóm tắt):**
- Bối cảnh: F_theta không có sẵn trong SD/DDIM — là UNet inversion do SwiftEdit train riêng, lưu checkpoint inverse_ckpt-120k/unet_ema (code: InverseModel.unet_inverse).
- Tiền đề: cần SwiftBrushv2 (SBv2) one-step đã train sẵn — vì Stage 1 cần biết cặp (noise, latent) ground-truth khi sinh ảnh synthetic.
- Bước 0 — Khởi tạo: lấy kiến trúc UNet có điều kiện text (cùng họ SD), chưa biết invert — giống khung UNet bước 02 diffusion-from-scratch nhưng nhiệm vụ ngược (latent+prompt → noise).
- Stage 1 (synthetic, ~100k iter): (1) random noise eps + prompt → SBv2 sinh ảnh; (2) VAE encode → latent z; (3) train F_theta: input (z, prompt) → output eps_hat; (4) loss: eps_hat ≈ eps (regression) + tái tạo ảnh qua SBv2. Vì biết eps gốc nên dạy được one-step inversion có nhãn.
- Stage 2 (ảnh thực, ~180k iter): dữ liệu CommonCanvas (ảnh thật + caption); fine-tune F_theta với DISTS (perceptual) + regularization giữ phân phối noise — generalize khỏi synthetic.
- Kết quả: checkpoint F_theta — inference 1 forward: ảnh → VAE → z + prompt → eps_hat; không cần DDIM inversion 50 bước.
- Repo CS2309 chỉ inference; không có script train Stage 1/2 — dùng weights pretrained từ release SwiftEdit.

**Ghi chú thêm / link:**
- Code: SwiftEdit/models.py (InverseModel), infer.py (unet_inverse). Paper Sec. 3; SwiftEdit_Overview.md §3.1; SwiftEdit_DeTai_CS2309.md §6.4 (đề tài không train lại).



---

## 4. Mask & ARaM

*Self-guided mask, editing mask, `s_y` / `s_edit` / `s_non-edit`, IP-Adapter, cross-attention, …*

<!-- qa:insert -->
### Q: IP-Adapter sẽ cho ra cái gì? Một ma trận, một số, hay một chuỗi?

**Ngày:** 2026-07-31  
**Chủ đề:** #ip-adapter #embedding #c_x #cross-attention

**Trả lời (tóm tắt):**
- Không phải một số, cũng không phải chuỗi chữ.
- IP-Adapter đưa ra tensor embedding ảnh (image_prompt_embeds): dạng (batch, seq_len, dim) — nhiều token ảnh (thường seq_len=4; IP-Adapter-Plus ~16), mỗi token là vector chiều dim.
- Luồng trong code: CLIP image encoder → clip_image_embeds → image_proj_model → image_prompt_embeds; rồi ghép (concat) với text prompt embeds theo chiều sequence, đưa vào cross-attention (to_k_ip / to_v_ip).
- Trong paper gọi điều kiện ảnh c_x; generator G_IP(ε, c_y, c_x) dùng cả prompt text và embedding ảnh.
- Tóm lại: một dãy vector / ma trận embedding (tokens), không phải scalar hay string.


### Q: Self-guided mask của SwiftEdit là gì và nằm ở bước nào?

**Ngày:** 2026-06-05

**Chủ đề:** #self-guided-mask #swiftedit #aram #pipeline

**Trả lời (tóm tắt):**
- Self-guided mask là mask vùng cần chỉnh sửa do SwiftEdit tự sinh, không cần người dùng vẽ mask.
- Nó được tạo sau bước **one-step inversion** và trước bước **ARaM editing**.
- SwiftEdit encode ảnh nguồn thành latent, rồi chạy inversion network `F_theta` với 2 prompt: `src_p` và `edit_p`.
- Hai noise map được so sánh bằng độ lệch tuyệt đối; vùng nào khác biệt nhiều được coi là vùng cần edit.
- Trong code: `SwiftEdit/infer.py` tính `predict_inverted_code`, tách thành `inverted_noise_1`, `inverted_noise_2`, rồi tạo `mask12 = |noise_source - noise_edit|` sau threshold.
- Sau đó `mask12` được đưa vào `MaskController`, và ARaM dùng mask này để tăng/giảm attention giữa vùng edit và vùng background.

**Pipeline ngắn gọn:**

```text
Ảnh nguồn + source/edit prompt
→ VAE encode ảnh thành latent
→ One-step inversion F_theta
→ Self-guided mask từ |noise_source - noise_edit|
→ ARaM dùng mask để chỉnh vùng edit, giữ vùng nền
→ One-step generation + VAE decode
```


### Q: SAM 3 có còn nên dùng để thay self-guided mask của SwiftEdit không?

**Ngày:** 2026-06-05

**Chủ đề:** #sam3 #mask #swiftedit #aram #segmentation #realtime

**Trả lời (tóm tắt):**
- Không nên chọn làm hướng đào sâu chính nếu luận điểm của đề tài là realtime/instant editing.
- SAM 3 vẫn hợp để phân tích chất lượng mask offline: self-guided mask vs GT mask vs SAM 3 mask.
- Nhưng nếu đưa SAM 3 vào pipeline chính, hệ thống phải load/chạy thêm segmentation model trước SwiftEdit, làm tăng latency và tài nguyên end-to-end.
- Điều này mâu thuẫn với điểm mạnh cốt lõi của SwiftEdit: một bước inversion + một bước editing để đạt tốc độ rất nhanh.
- Hướng hợp lý hơn: giữ SAM 3 như optional/failure analysis, còn hướng chính chuyển sang **SwiftEdit-RT** — profile bottleneck và tối ưu inference.
- Nếu vẫn làm SAM 3, phải ghi rõ đây là preprocessing/offline mask analysis, không phải realtime pipeline.

**Nguồn:** [SAM 3 paper/project](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) · [arXiv](https://arxiv.org/abs/2511.16719) · [GitHub](https://github.com/facebookresearch/sam3)


### Q: Dùng SAM 3 thay self-guided mask có làm chậm và tốn tài nguyên hơn không?

**Ngày:** 2026-06-05

**Chủ đề:** #sam3 #runtime #resource #mask #colab

**Trả lời (tóm tắt):**
- Có. SAM 3 thêm một bước segmentation trước SwiftEdit, nên tổng thời gian và tài nguyên chắc chắn tăng so với self-guided mask gốc.
- Self-guided mask gần như "miễn phí thêm" trong SwiftEdit vì nó tận dụng luôn 2 noise map đã tính trong one-step inversion.
- SAM 3 cần load thêm model segmentation lớn, thêm VRAM/RAM, thời gian inference và có thể cần CUDA/PyTorch mới hơn SwiftEdit.
- Vì vậy không nên quảng bá SAM 3 là nhanh hơn; trong đề tài hiện tại nên xem đây là trade-off chất lượng mask, không phải hướng tối ưu realtime.
- Cách đo công bằng: báo cáo riêng `runtime_swiftedit_s`, `runtime_sam3_s`, `runtime_total_s`; so sánh thêm PSNR/MSE background, CLIP và IoU/Dice.
- Cách triển khai hợp lý: chạy SAM 3 offline/batch để cache mask trước, rồi chạy SwiftEdit nhiều cấu hình trên mask đã lưu. Khi đó inference SwiftEdit vẫn nhanh, nhưng pipeline end-to-end có thêm chi phí SAM 3.
- Kết luận: SAM 3 chỉ nên giữ như phân tích offline/optional; hướng chính realtime nên ưu tiên SwiftEdit-RT để giảm latency của pipeline hiện có.


---

## 5. Đánh giá & PieBench

*PSNR, MSE, CLIP-Whole, CLIP-Edited, IoU/Dice, 10 loại editing, …*

<!-- qa:insert -->
### Q: cuDNN nói Tensor Core gần nhưng không bit-identical FP32 nghĩa là gì?

**Ngày:** 2026-07-31  
**Chủ đề:** #fp16 #tensor-core #cudnn #precision #psnr

**Trả lời (tóm tắt):**
- Bit-identical = từng bit float giống hệt đường FP32 thuần.
- Tensor Core dùng MMA (nhân ma trận gộp), input thường FP16, tích lũy FP16/FP32 rồi có thể down-convert về FP16; thứ tự cộng số thực không kết hợp → làm tròn khác FP32 tuần tự.
- Kết quả rất gần (PSNR ~48 trong bench) nhưng không trùng bit → PSNR hữu hạn, không ∞.
- cuDNN ghi vậy: bật Tensor Core chấp nhận sai số nhỏ đổi tốc độ — không claim output giống hệt FP32 từng bit.
- Nguồn: NVIDIA cuDNN Core Concepts — Notes on Tensor Core Precision.


### Q: PSNR, SSIM, LPIPS, MSE trong Excel so sánh FP16 vs FP32 nghĩa là gì?

**Ngày:** 2026-07-31  
**Chủ đề:** #psnr #ssim #lpips #mse #fidelity #torchmetrics

**Trả lời (tóm tắt):**
- Trong bench đề tài, cả 4 độ đo so output config (vd FP16) với output baseline_fp32 cùng job_id + prompt_idx + seed — không so ảnh gốc.
- MSE↓: trung bình bình phương sai số pixel [0,1]; càng nhỏ càng giống FP32. MSE~2e-5 → lệch rất nhỏ.
- PSNR↑ (dB): 10·log10(1/MSE); cao = sai số nhỏ. ~48 dB = gần trùng; ~22 dB (FP4) = lệch rõ; edit↔nguồn ~19 dB.
- SSIM↑ (0–1): giống cấu trúc/độ tương phản cục bộ (gần mắt người hơn MSE). Gần 1 = cấu trúc giữ.
- LPIPS↓: khoảng cách perceptual qua mạng deep (AlexNet/VGG); thấp = trông gần như nhau. ~0.001 = gần không lệch mắt.
- Đọc cùng nhau: PSNR cao + SSIM gần 1 + LPIPS gần 0 = fidelity precision tốt; chỉ nhìn PSNR dễ hiểu nhầm là "giống ảnh gốc".


### Q: Nén model để làm gì nếu card 24GB vẫn cần dùng card 24GB?

**Ngày:** 2026-07-26  
**Chủ đề:** #fp16 #vram #compression #t4

**Trả lời (tóm tắt):**
- Analog thùng 3L đúng một nửa: nếu luôn có A100 và không cần tốc độ thì ROI nén thấp.
- Đề tài cần FP16 vì fit T4 16GB/Mac (paper ~24GB; FP32 peak ~14.6GB dễ chật), headroom Gradio, latency (~1.7× T4 / ~7× Mac), disk −50%.
- FP4 giảm VRAM thêm nhưng PSNR↔FP32 ~21 dB → không khuyến nghị.


### Q: PSNR 48.6 có phải trick / so ảnh gốc không? Code có sai không?

**Ngày:** 2026-07-26  
**Chủ đề:** #psnr #fidelity #piebench #metrics #audit

**Trả lời (tóm tắt):**
- Không sai code: quality_speed_bench so output FP16 ↔ output FP32 cùng job (REFERENCE=baseline_fp32), không vs ảnh nguồn.
- ~48 dB ≈ MSE ~0.00002 — hai edit gần trùng pixel sau đổi precision; kỳ vọng.
- Spot-check 12 mẫu: FP16↔FP32 ~47 dB; edit↔source ~19 dB → không phải copy gốc.
- Hai lớp metric: (A) fidelity PSNR/SSIM/LPIPS vs FP32; (B) edit quality CLIP + PSNR nền vs source (PieBench subset20: CLIP-W 23.0, PSNR nền 14.0).
- Chi tiết: report/AUDIT_PSNR_FIDELITY.md · EDIT_QUALITY_SUMMARY.md


### Q: PSNR ~48.5 dB có phải số phi thực tế (gần như ảnh gốc) không? Tính đúng không?

**Ngày:** 2026-07-24  
**Chủ đề:** #psnr #ssim #lpips #metrics #fp16 #baseline #precision

**Trả lời (tóm tắt):**
- Tính đúng nhưng reference là output FP32 cùng job — không phải ảnh gốc. Mean PSNR cao dễ bị hiểu nhầm nên đề tài bổ sung thêm độ đo + min/max.
- Nguồn: quality_speed_bench_2026-06-17, improved_fp16_cache, n=600 vs baseline_fp32:
- PSNR↑: mean 48.56 · min 34.97 · max 56.73
- SSIM↑: mean 0.9976 · min 0.9889 · max 0.9994
- LPIPS↓: mean 0.0008 · min 0.0001 · max 0.0063
- MSE↓: mean ~0.000020 · min ~0.000002 · max ~0.00032
- Đối chứng: edit↔ảnh nguồn trên mẫu ~18–21 dB; FP4 cùng protocol PSNR 21.67 (16.59–29.41), SSIM 0.78, LPIPS 0.15 → bộ đo phân biệt được khi lệch.
- Slide/báo cáo: tách bảng tốc độ và bảng chất lượng đa độ đo; nêu rõ ‘vs FP32 output’.


### Q: Dùng 3 prompt_idx thì EditCache sao trùng prompt được?

**Ngày:** 2026-07-24  
**Chủ đề:** #editcache #prompt #piebench #benchmark #cache

**Trả lời (tóm tắt):**
- Không cần trùng edit prompt. Cache khóa theo ảnh + source prompt; 3 prompt_idx chỉ đổi edit prompt.
- Mỗi ảnh PIE-Bench: 1 src_prompt cố định + 3 edit_prompt khác nhau (idx 0/1/2).
- Thứ tự chạy cùng ảnh: idx0 = miss (tính VAE latent, CLIP image embed, source text embed); idx1 và idx2 = hit (tái dùng 3 thành phần trên).
- Phần vẫn tính lại mỗi lần: embed/inversion/generation phụ thuộc edit_prompt — vì 3 edit khác nhau nên output khác nhau.
- Đổi ảnh hoặc đổi source prompt → invalidate cache. Lần edit đầu trên ảnh mới luôn miss.


### Q: PSNR, MSE×10⁴↓, CLIP-Whole↑, CLIP-Edited↑ là độ đo gì, so sánh với cái gì?

**Ngày:** 2026-07-24  
**Chủ đề:** #metrics #psnr #mse #clip #piebench #swiftedit

**Trả lời (tóm tắt):**
- Bốn độ đo chuẩn paper SwiftEdit (Table 1) / PIE-Bench — so các method editing với nhau trên cùng dataset (ảnh nguồn, edit prompt, GT mask).
- PSNR↑ + MSE×10⁴↓: đo bảo toàn background. So ảnh edited với ảnh nguồn trên vùng KHÔNG sửa (1 − mask). PSNR cao = nền giữ tốt; MSE thấp = sai số pixel nhỏ. ×10⁴ = nhân 10000 cho dễ đọc bảng.
- CLIP-Whole↑: CLIPScore giữa TOÀN ảnh edited và edit_prompt — edit có khớp nghĩa prompt không (toàn cục).
- CLIP-Edited↑: CLIPScore giữa vùng TRONG mask của ảnh edited và edit_prompt — vùng được sửa có đúng nội dung prompt không.
- Mũi tên: ↑ càng cao càng tốt; ↓ càng thấp càng tốt.
- Không nhầm với PSNR vs FP32 trên slide tối ưu: cái đó so output config với output baseline_fp32 cùng job (sai số precision), còn PSNR paper là so edited vs source trên background.


### Q: Trong kết quả đề tài SwiftEdit, các độ đo Miss/Hit/Overall/PSNR/SSIM/LPIPS/Disk/VRAM nghĩa là gì?

**Ngày:** 2026-07-24  
**Chủ đề:** #metrics #psnr #ssim #lpips #editcache #precision

**Trả lời (tóm tắt):**
- Miss/cold (s): thời gian 1 edit khi chưa có EditCache (thường prompt_idx=0).
- Hit (s): thời gian 1 edit khi đã cache (cùng ảnh + source; prompt_idx=1–2).
- Overall × vs FP32: trung bình 600 job; t_fp32 / t_config (1.69× = nhanh hơn FP32 ~1.69 lần).
- PSNR / SSIM / LPIPS vs FP32: so **output config ↔ output baseline_fp32** cùng job (không vs ảnh nguồn). Báo mean kèm min–max. FP16 (n=600): PSNR 48.56 (34.97–56.73), SSIM 0.9976 (0.9889–0.9994), LPIPS 0.0008 (0.0001–0.0063).
- Disk (GiB): dung lượng cây swiftedit_weights; FP32→FP16: 9.79→4.94 GiB.
- VRAM/peak (MB): bộ nhớ GPU (CUDA) hoặc peak alloc (MPS) lúc load/edit.
- Slide: bảng tốc độ tách khỏi bảng chất lượng đa độ đo; nguồn `quality_speed_bench_2026-06-17`.

**Ghi chú thêm / link:**
- report/SLIDE_SwiftEdit.md §6a–§6b
- experimental_data/quality_speed_bench_2026-06-17/report.md


### Q: 200 ảnh × 3 prompt: 3 prompt là gì, lấy từ đâu?

**Ngày:** 2026-07-24  
**Chủ đề:** #piebench #prompt #benchmark #cache

**Trả lời (tóm tắt):**
- Mỗi ảnh có 1 source prompt (original_prompt PIE-Bench) và 3 edit prompt.
- Template cố định trong jobs_june17.json / build_june17_jobs.py:
- 0 = {edit} — editing_prompt gốc PIE-Bench (semantic edit).
- 1 = {src} at night, dark lighting — global night.
- 2 = {src} in winter, covered in snow — global winter.
- Dataset: data/PIE-Bench-auto200 (200 ảnh); file jobs: data/jobs_june17.json; khớp quality_speed_bench_2026-06-17.
- prompt_idx=0 dùng cold/fair; idx 1–2 tạo cache hit khi cùng ảnh+source.

**Ghi chú thêm / link:**
- scripts/build_june17_jobs.py; data/jobs_june17.json


### Q: Đánh giá SwiftEdit dùng độ đo nào, đo thế nào, công nhận ở đâu?

**Ngày:** 2026-06-05  
**Chủ đề:** #piebench #metrics #psnr #clip #mse

**Trả lời (tóm tắt):**
- Benchmark chuẩn: PIE-Bench (700 ảnh, 10 loại edit, có GT mask) — từ PnP Inversion (ICLR 2024).
- PSNR + MSE: đo trên vùng KHÔNG sửa (1 - mask) → bảo toàn background. PSNR cao↑, MSE thấp↓.
- CLIP-Whole: CLIPScore(ảnh edited, edit prompt) trên toàn ảnh → độ trung thành chỉnh sửa. Cao↑.
- CLIP-Edited: CLIPScore(vùng mask của ảnh edited, edit prompt) → sửa đúng vùng + đúng prompt. Cao↑.
- Runtime: thời gian 1 lần `edit_image()` sau khi model đã load — điểm mạnh SwiftEdit (~0.23s A100, ~1.3s T4, ~30-50s Mac MPS).
- Bổ sung: LPIPS, SSIM, structure distance (DINO) trong paper; IoU/Dice cho mask tự sinh vs GT.
- Trong repo: `scripts/piebench_metrics.py` dùng `torchmetrics` với CLIP ViT-L/14, PSNR và MSE; `CLIP-Whole` đã được chuẩn hóa để dùng `edit_prompt`, khớp metric `clip_similarity_target_image` của PIE-Bench.
- Công nhận: [PnP Inversion / PIE-Bench ICLR 2024](https://github.com/cure-lab/PnPInversion) (định nghĩa benchmark + `evaluation/evaluate.py`); [SwiftEdit CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_SwiftEdit_Lightning_Fast_Text-Guided_Image_Editing_via_One-Step_Diffusion_CVPR_2025_paper.pdf) (Table 1); [CLIP ICML 2021](https://proceedings.mlr.press/v139/radford21a) và [CLIPScore EMNLP 2021](https://arxiv.org/abs/2104.08718).




---

## 6. Triển khai Mac / Colab

*MPS vs CUDA, cài env, weights, OOM, runtime, …*

<!-- qa:insert -->
### Q: Vì sao thử FP4 và vì sao vẫn chạy được trên T4 dù không có native FP4?

**Ngày:** 2026-07-24  
**Chủ đề:** #fp4 #bitsandbytes #t4 #quantization

**Trả lời (tóm tắt):**
- Thử vì muốn nén mạnh hơn FP16 để giảm VRAM trên T4 16GB (chuỗi FP32→FP16→FP4).
- Đã làm: quantize_unet() trong models.py — sau load UNet, thay nn.Linear bằng bitsandbytes Linear4bit (quant_type=fp4), compute_dtype=fp16; chỉ Linear, Conv/VAE/CLIP không quant, VAE giữ FP32.
- Vẫn chạy vì bitsandbytes dequant 4-bit→FP16 rồi MatMul FP16 trên Turing — không cần Tensor Core FP4.
- Chạy được nhưng không tối ưu: tốc độ không thắng FP16 rõ; PSNR ~21dB vs ~48.5dB của FP16 → giữ làm ablation âm; dừng làm hướng tối ưu (FP4_DECISION_AND_NEXT_PLAN.md).

**Ghi chú thêm / link:**
- SwiftEdit/models.py quantize_unet; experimental_data/FP4_DECISION_AND_NEXT_PLAN.md


### Q: Chỗ nào trong SwiftEdit ép tensor tính toán sang FP16?

**Ngày:** 2026-07-24  
**Chủ đề:** #fp16 #models #vae #precision

**Trả lời (tóm tắt):**
- Không ép cả pipeline. Khi dtype=fp16 trong models.py:
- Ép FP16: Inverse UNet, Generation UNet (SBv2), IP-Adapter (proj + to_k_ip/to_v_ip), CLIP text encoder, CLIP image encoder.
- Giữ FP32: VAE encode/decode (tránh NaN/ảnh đen); trước decode latent được .to(float32).
- Timestep vẫn int64.
- Disk vẫn có thể là FP32 nếu chỉ compute fp16 (improved_fp16_cache); muốn giảm disk phải convert cây weight FP16 (fp16_disk).

**Ghi chú thêm / link:**
- SwiftEdit/models.py; report/BAO_CAO_SwiftEdit.md §6.2.2


### Q: 32 bit trong FP32 là 32 bit của cái gì?

**Ngày:** 2026-07-24  
**Chủ đề:** #fp32 #fp16 #precision #quantization

**Trả lời (tóm tắt):**
- Là 32 bit nhị phân dùng để lưu MỖI một số thực (một weight hoặc một activation), không phải 32 bit của cả mô hình.
- FP32 ≈ 4 byte/số; FP16 ≈ 2 byte/số; FP4 ≈ 0.5 byte/số — tổng dung lượng checkpoint/VRAM giảm vì mô hình có hàng tỷ số như vậy.
- Chuẩn IEEE 754 chia các bit thành dấu + mũ + định trị (FP32 thường 1+8+23); báo cáo chỉ cần nhớ: ít bit hơn → tiết kiệm bộ nhớ nhưng dễ sai số học hơn.
- Trong đề tài: baseline = FP32; tối ưu chính = FP16; FP4 = quant một phần Linear.

**Ghi chú thêm / link:**
- report/BAO_CAO_SwiftEdit.md §6.2.1


### Q: fp16_weight_xformers khác fp16_weight thế nào?

**Ngày:** 2026-07-19  
**Chủ đề:** #xformers #fp16 #swiftedit-rt #colab

**Trả lời (tóm tắt):**
- Cùng cây weights fp16 trên disk + EditCache; không đổi precision.
- fp16_weight_xformers bật xFormers Memory-Efficient Attention.
- Inverse UNet: Diffusers enable_xformers_memory_efficient_attention.
- Gen UNet: self-attn (và nhánh không ARaM controller) dùng xformers.ops; không ghi đè IP processors.
- Cross-attn có mask controller vẫn einsum (ARaM) — MEA không đụng nhánh đó.
- Alias: fp16_weight_xformers → config fp16_disk_xformers; fp16_weight vẫn giữ nguyên.


### Q: Sau khi dừng FP4 trên T4, nên thử hướng nén/tăng tốc nào tiếp (xếp theo đơn giản × hiệu quả)?

**Ngày:** 2026-07-19  
**Chủ đề:** #xformers #tome #tensorrt #swiftedit-rt #colab

**Trả lời (tóm tắt):**
- Giữ baseline FP16 + EditCache (và fp16_weight nếu cần disk).
- Hạng nhanh: (1) xFormers Memory-Efficient Attention — T4 hỗ trợ MEA.
- (2) Token Merging / tomesd — tune ratio, đo PSNR.
- Tiếp: torch.compile, TinyVAE ablation riêng.
- Nặng hơn: TensorRT FP16 cho UNet (effort cao, sau khi 1–2 có số).
- Không lưu checkpoint FP4 trên disk nếu runtime không thắng.
- Chi tiết: experimental_data/FP4_DECISION_AND_NEXT_PLAN.md


### Q: Vì sao FP4 (bitsandbytes) không phải giải pháp nén đúng nghĩa cho SwiftEdit trên Tesla T4?

**Ngày:** 2026-07-19  
**Chủ đề:** #fp4 #quantization #t4 #colab #swiftedit-rt

**Trả lời (tóm tắt):**
- T4 (Turing) không có native FP4: weight FP4 phải dequant sang FP16 rồi mới MatMul.
- UNet ảnh thường compute-bound → chi phí dequant triệt lợi đọc VRAM ít hơn → tốc độ ≈ fp16.
- Repo chỉ quant nn.Linear (bỏ Conv/VAE/CLIP) nên VRAM giảm hạn chế trong khi PSNR tụt mạnh (~21.7 vs ~48.6 của fp16).
- Kết luận đề tài: dừng tối ưu FP4 trên T4; giữ làm ablation âm; baseline = FP16 + EditCache.


### Q: Tại sao đổi sang fp16 compute mà dung lượng checkpoint trên disk gần như không đổi?

**Ngày:** 2026-07-19  
**Chủ đề:** #fp16 #weights #disk #vram #safetensors

**Trả lời (tóm tắt):**
- `dtype=fp16` trong `InverseModel`/`IPSBV2Model` ép tensor trên GPU sau khi đọc file — không rewrite file `.safetensors`/`.bin` trên đĩa.
- Checkpoint Qualcomm (~3.2–3.3GB mỗi thư mục UNet/IP) vẫn lưu precision gốc (thường fp32/bf16 tùy file).
- Muốn giảm disk: phải convert + `save_pretrained` (hoặc tool safetensors) rồi trỏ path mới; load với `torch_dtype=fp16`.
- Muốn giảm peak VRAM lúc load: tránh nạp fp32 rồi mới `.to(fp16)`; load thẳng fp16 hoặc quant (bnb) đúng path.


### Q: Có thể tự quantize checkpoint xuống fp16/fp8/fp4 để giảm disk và VRAM khi load weight không? Dễ không? Liên quan GGUF?

**Ngày:** 2026-07-19  
**Chủ đề:** #quantization #fp16 #fp4 #gguf #vram #weights #swiftedit-rt

**Trả lời (tóm tắt):**
- Ba lớp khác nhau — dễ lẫn:
- 1) Compute fp16 (`.to(fp16)`): đã có; giảm VRAM lúc chạy, nhưng file trên disk vẫn có thể là fp32 → peak VRAM lúc load có thể gần bằng fp32.
- 2) Lưu checkpoint fp16/safetensors: khả thi cao, khá dễ với diffusers (`save_pretrained` + `torch_dtype=fp16`); disk ~−50%; load với `torch_dtype=fp16` giảm peak lúc nạp. Không phải \"quantize sâu\".
- 3) Weight-only fp8/fp4 lúc runtime: repo ĐÃ có `quantize_unet()` (torchao fp8 / bitsandbytes fp4) SAU `from_pretrained` — giảm VRAM sau khi nén, nhưng peak load vẫn gần full weights (không giảm disk trừ khi tự save format quant).
- Kết quả bench Colab T4 (2026-06-17): fp16+cache sweet spot (VRAM −42%, PSNR ~48.6dB); fp8 hỏng (PSNR ~6); fp4 VRAM −48.5% nhưng PSNR ~21.7dB.
- GGUF: format/tooling chủ yếu LLM (llama.cpp). Không phải đường chuẩn cho UNet Diffusers/SwiftEdit. ComfyUI có GGUF SD thử nghiệm nhưng khác stack — không “chạy tool → dùng luôn” với `models.py`.
- Độ dễ: lưu fp16 disk = dễ; runtime fp4 = đã code sẵn; lưu fp4/GGUF production = khó / ít lợi so rủi ro chất lượng với one-step editor.


### Q: Nên ưu tiên các hướng tăng tốc SwiftEdit-RT theo thứ tự nào?

**Ngày:** 2026-06-05

**Chủ đề:** #swiftedit-rt #runtime #optimization #no-training

**Trả lời (tóm tắt):**
- Ưu tiên 0: **bỏ decode `noise_image` không dùng** trong `gen_img()`; đơn giản nhất, không đổi ảnh edited chính, không cần train.
- Ưu tiên 1: **vectorized self-guided mask trên GPU**; thay `.cpu().apply_()` bằng tensor threshold, giảm CPU-GPU sync, gần như không rủi ro chất lượng.
- Ưu tiên 2: **profile latency theo module**; không tăng tốc trực tiếp nhưng bắt buộc để chứng minh bottleneck và speedup.
- Ưu tiên 3: **cache latent/image embedding/source prompt embedding** cho cùng ảnh nhiều prompt; rất hợp demo realtime, không đổi chất lượng nếu cache đúng.
- Ưu tiên 4: **`channels_last` + `fp16`/`bf16` trên CUDA**; benefit có thể tốt, cần so PSNR/CLIP/ảnh để xác nhận không lệch.
- Ưu tiên 5: **`torch.compile`**; có thể nhanh ở warm inference/batch nhưng có compile overhead và rủi ro tương thích.
- Ưu tiên thấp hơn: **TinyVAE/TAESD, TensorRT/Core ML/quantization**; có thể nhanh hơn nhưng có rủi ro đổi chất lượng hoặc tốn công tích hợp.
- Không ưu tiên cho mục tiêu tốc độ: SAM 3/VLM auto-caption/baseline editor mới, vì thêm model/bước xử lý.


### Q: Vì sao đổi hướng từ SAM 3 sang SwiftEdit-RT?

**Ngày:** 2026-06-05

**Chủ đề:** #swiftedit-rt #runtime #optimization #realtime #colab

**Trả lời (tóm tắt):**
- SwiftEdit được thiết kế để chạy rất nhanh; nếu thêm SAM 3 làm segmentation trước khi edit thì pipeline sẽ chậm hơn và tốn VRAM/RAM hơn.
- Khi latency đã tăng, hướng SAM 3 dễ bị phản biện: ở cùng mức tốc độ đó có thể dùng model/editing method khác chất lượng cao hơn.
- SwiftEdit-RT giữ đúng tinh thần bài báo: không thêm model mới vào đường realtime, mà giảm overhead trong pipeline hiện có.
- Các hướng tối ưu chính: profile từng module, vectorized self-guided mask trên GPU, bỏ decode `noise_image` không dùng, cache latent/embedding cho cùng ảnh nhiều prompt, thử `fp16`, `channels_last`, `torch.compile`, TinyVAE/TAESD.
- Cách đánh giá: `runtime_total_s`, latency breakdown từng module, peak memory, speedup, PSNR/MSE background, CLIP-Whole/Edited và so sánh ảnh baseline vs optimized.
- Câu chuyện báo cáo tốt hơn: *sau khi SwiftEdit giảm diffusion step xuống 1+1, bottleneck hệ thống còn nằm ở đâu và tối ưu được bao nhiêu?*


### Q: 15 GB có đủ để dùng SwiftEdit không (VRAM vs ổ đĩa)?

**Ngày:** 2026-06-04  
**Chủ đề:** #colab #storage #drive

**Trả lời (tóm tắt):**
- VRAM T4 ~15GB: đủ chạy inference 512×512 (batch=1); OOM nếu fine-tune/batch lớn.
- Ổ đĩa/Drive 15GB trống: chật — weights Qualcomm ~9.6GB + HF ~3–8GB; nên ≥20–25GB hoặc chỉ lưu weights trên Drive + tải HF tối thiểu (download_hf_models.sh).
- Không snapshot_download full mirror SD2.1 (~30GB+ cache).


### Q: SwiftEdit chạy trên Google Colab T4 được không?

**Ngày:** 2026-06-04  
**Chủ đề:** #colab #t4 #cuda

**Trả lời (tóm tắt):**
- Được — đúng kế hoạch đề tài: T4 ~15GB VRAM đủ inference 512×512, dùng requirements.txt CUDA (cu118).
- Clone repo đề tài CS2309 (có patch mirror SD2.1, map_location ip_adapter), lưu swiftedit_weights trên Drive.
- Nhanh hơn Mac MPS; PieBench/baseline nên chạy Colab.


### Q: Tại sao SwiftEdit chạy chậm trên Mac so với paper (~0.23s)?

**Ngày:** 2026-06-04  
**Chủ đề:** #mac #mps #runtime

**Trả lời (tóm tắt):**
- Paper đo trên NVIDIA A100 + CUDA; Mac dùng MPS, thường fp32, nhiều model load (sd-turbo, SD2.1, IP-Adapter, weights ~9.6GB).
- Thời gian ~90s/ảnh trên M4 thường gồm load model + inference 512×512 — không phải multi-step 50+50.
- Colab T4 nhanh hơn Mac; chỉ A100 mới gần số paper.




---

## 7. Chỉnh sửa & Style

*Semantic editing vs style transfer, prompt tiếng Anh, failure cases, …*

<!-- qa:insert -->

### Q: Xóa vật thể khỏi ảnh bằng SwiftEdit có khả thi không và đánh giá thế nào?

**Ngày:** 2026-06-05

**Chủ đề:** #object-removal #inpainting #mask #metrics #swiftedit

**Trả lời (tóm tắt):**
- Khả thi **trung bình-cao** nếu object nhỏ/vừa và background đơn giản. Đây là local edit nên hợp với self-guided mask + ARaM hơn global style.
- Cách prompt: source có object, edit prompt bỏ object, ví dụ `"a street with a red car"` → `"an empty street, no car"`.
- Nên thử 3 cấu hình: SwiftEdit-SG (mask tự sinh), SwiftEdit-UserMask/GTMask (mask object), SwiftEdit-SG+Dilate (nới mask để tránh viền/ghosting).
- Điểm khó: SwiftEdit không phải inpainting model chuyên dụng; với object lớn hoặc che nhiều nền, có thể còn bóng/viền/texture méo.
- Metric chính: **removal success** bằng detector confidence drop hoặc CLIP margin `"without object"` vs `"with object"`.
- Metric giữ nền: PSNR/SSIM/MSE/LPIPS trên vùng ngoài object mask `(1 - mask)`.
- Metric vùng inpaint: human rating/crop review, MUSIQ/NIQE, hoặc FID/KID nếu có tập target đủ lớn.
- Metric mask: IoU/Dice giữa self-guided mask và object mask nếu có GT/user mask.
- Nên so baseline LaMa nếu kịp, vì LaMa là inpainting chuyên dụng và có thể lấp nền tốt hơn SwiftEdit ở vùng xóa lớn.

**Nguồn / metric nền:** [LaMa](https://openaccess.thecvf.com/content/WACV2022/html/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.html) · [LPIPS](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0299.pdf) · [FID](https://papers.nips.cc/paper/7240-gans-trained-by-a-two-time-scale-update-rule-converge-to-a-local-nash-equilibrium) · [CLIPScore](https://aclanthology.org/2021.emnlp-main.595/)


### Q: Global style/weather edit như ngày↔đêm, mùa, mưa↔nắng nên đánh giá thế nào và SwiftEdit có phù hợp không?

**Ngày:** 2026-06-05

**Chủ đề:** #global-edit #style-transfer #weather #metrics #swiftedit

**Trả lời (tóm tắt):**
- Đây là **global attribute/style transfer**, không phải local object edit. Mask IoU/Dice gần như vô nghĩa vì toàn ảnh đều có thể cần thay đổi.
- PSNR/MSE background của PieBench cũng không phù hợp vì không còn vùng background cần giữ nguyên tuyệt đối.
- Metric nên đổi sang 3 trục: **đúng style/target**, **giữ content/layout**, **ảnh tự nhiên/ít artifact**.
- Đúng target: CLIPScore với `edit_prompt`, zero-shot CLIP label giữa `{day, night}`, `{spring, summer, autumn, winter}`, `{sunny, rainy, overcast}`; có thể dùng ΔCLIP target so với ảnh gốc.
- Giữ content/layout: DINO similarity hoặc CLIP image-image similarity giữa source và edited; LPIPS/SSIM toàn ảnh chỉ dùng phụ vì global edit cần đổi màu/ánh sáng.
- Realism/artifact: human rating 1–5, MUSIQ/NIQE nếu có; nếu có tập target domain đủ lớn thì dùng FID/KID giữa edited set và ảnh thật target domain.
- Độ khả thi với SwiftEdit: **trung bình**. Các edit nhẹ như day→night vừa phải, sunny→overcast, warm/cold tone có khả năng làm được; mùa đông/tuyết, mưa lớn, đêm có đèn/phản chiếu khó hơn.
- Lý do rủi ro: SwiftEdit được thiết kế mạnh cho semantic/local edit với self-guided mask + ARaM; global prompt có thể bị under-edit, đổi không đều hoặc làm lệch texture/layout.
- Nên làm như hướng ứng dụng/phân tích giới hạn, không nên kỳ vọng chắc chắn tốt như đổi object/attribute cục bộ.

**Nguồn / metric nền:** [CLIPScore](https://aclanthology.org/2021.emnlp-main.595/) · [DINO](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper) · [LPIPS](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0299.pdf) · [MUSIQ](https://mlanthology.org/iccv/2021/ke2021iccv-musiq/) · [FID](https://papers.nips.cc/paper/7240-gans-trained-by-a-two-time-scale-update-rule-converge-to-a-local-nash-equilibrium)

---

## 8. Chưa phân loại

*Câu hỏi chưa xếp vào mục — sắp xếp lại sau.*

<!-- qa:insert -->
### Q: Train LoRA day↔night bằng cặp Gemini có khả thi trên T4 không?

**Ngày:** 2026-07-26  
**Chủ đề:** #lora #daynight #gemini #t4 #contribution

**Trả lời (tóm tắt):**
- Khả thi ở mức pilot: LoRA rank 4–16 trên gen UNet (SBv2), đóng băng F_theta; T4 16GB đủ với batch 1 + grad checkpoint.
- Cặp (gốc, Gemini edit) không khớp train F_theta paper; dùng để bias style ngày/đêm trên generator.
- Cần 50–150 cặp sạch + 15–20 hold-out; 1–2 ảnh sẽ overfit.
- Script: train_lora_daynight.py · eval_lora_daynight.py · report/LORA_DAYNIGHT_PILOT.md


### Q: Tại sao không dùng LaTeX trong file MD?

**Ngày:** 2026-06-01  
**Chủ đề:** #markdown #swiftedit

**Trả lời (tóm tắt):**
- Nhiều MD reviewer (GitHub, LMS) không render LaTeX.
- Dùng backtick, code block và Unicode thay thế.
- Skill write-markdown quy định rule này cho repo CS2309.




---

## Tags gợi ý

Dùng trong entry để dễ tìm:

`#diffusion` `#one-step` `#swiftedit` `#inversion` `#noise` `#aram` `#mask` `#attention` `#ip-adapter` `#piebench` `#metrics` `#mac` `#colab` `#style-editing` `#prompt`

---

## Gợi ý câu hỏi có thể hỏi sau

- Diffusion model hoạt động thế nào? One-step khác multi-step ở đâu?
- Inversion là gì? Tại sao SwiftEdit không dùng DDIM Inversion?
- Inverted noise (`eps_hat`) dùng để làm gì khi edit?
- Self-guided mask được tính như thế nào?
- ARaM khác gì so với chỉnh global scale `s_x`?
- PSNR và CLIP đo cái gì? Trade-off giữa chúng?
- Tại sao paper chạy A100 còn mình dùng Mac + Colab?
- SwiftEdit có làm tốt style editing (watercolor, anime) không?

---

*File Q&A — bổ sung dần trong quá trình học và làm đề tài.*
