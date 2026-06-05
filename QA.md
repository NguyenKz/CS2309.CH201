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
| 1 | [Diffusion & Text-to-Image](#1-diffusion--text-to-image) | 1 |
| 2 | [SwiftEdit & Pipeline](#2-swiftedit--pipeline) | 3 |
| 3 | [Inversion & Noise](#3-inversion--noise) | 0 |
| 4 | [Mask & ARaM](#4-mask--aram) | 3 |
| 5 | [Đánh giá & PieBench](#5-đánh-giá--piebench) | 1 |
| 6 | [Triển khai Mac / Colab](#6-triển-khai-mac--colab) | 5 |
| 7 | [Chỉnh sửa & Style](#7-chỉnh-sửa--style) | 2 |
| 8 | [Chưa phân loại](#8-chưa-phân-loại) | 1 |

*(Cập nhật cột "Số câu hỏi" khi thêm entry.)*

---

## 1. Diffusion & Text-to-Image

*Khái niệm nền: diffusion model, one-step vs multi-step, SBv2, CLIP, VAE, …*

<!-- qa:insert -->
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
*(Chưa có câu hỏi.)*

---

## 4. Mask & ARaM

*Self-guided mask, editing mask, `s_y` / `s_edit` / `s_non-edit`, IP-Adapter, cross-attention, …*

<!-- qa:insert -->
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
