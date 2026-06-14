# Tổng hợp hướng phát triển SwiftEdit

Cập nhật: 2026-06-14

File này gom các hướng phát triển đã bàn cho đề tài SwiftEdit. Mỗi hướng được tóm tắt theo 6 ý: mô tả ngắn, benchmark nếu có, độ đo benchmark, các bước implement, độ khả thi, và phân loại ứng dụng/nghiên cứu.

## 1. Bảng tổng hợp nhanh

| Ưu tiên | Hướng | Mô tả ngắn | Benchmark nếu có | Độ khả thi | Loại |
|---:|---|---|---|---:|---|
| 1 | **SwiftEdit-RT: realtime inference acceleration** | Giảm latency pipeline hiện tại, không thêm model nặng, không train lại | PIE-Bench subset, custom latency benchmark | **Cao** | Nghiên cứu + engineering |
| 2 | **Object removal / inpainting** | Xóa vật thể bằng prompt + self/user mask | Custom 20-40 ảnh, COCO mask, PIE-Bench remove/delete nếu có, LaMa baseline | **Trung bình-cao** | Ứng dụng + nghiên cứu |
| 3 | **Global scene/style/weather editing** | Đổi ngày/đêm, mùa, mưa/nắng, tone toàn ảnh | Custom 20-40 ảnh; target-domain set nếu có | **Trung bình** | Ứng dụng + phân tích giới hạn |
| 4 | **Self-guided mask analysis** | Đo self-guided mask có định vị đúng vùng edit không | PIE-Bench có GT mask | **Cao** | Nghiên cứu/đánh giá |
| 5 | **SAM 3 / GroundingDINO+SAM mask analysis** | Dùng segmentation model sinh mask offline để so với self-guided mask | PIE-Bench GT mask, custom object masks | **Trung bình** | Nghiên cứu optional |
| 6 | **Auto source prompt / concept extraction** | Dùng Florence-2/Qwen2.5-VL tự tạo source prompt/concept từ ảnh | PIE-Bench, custom prompt set | **Trung bình** | Ứng dụng optional |
| 7 | **Grounded evaluation beyond CLIP** | Kiểm tra edit đúng đối tượng/thuộc tính bằng detector/VQA thay vì chỉ CLIPScore | Custom grounded set, GIE-style protocol nếu có | **Trung bình** | Nghiên cứu/đánh giá |
| 8 | **Modern baseline comparison** | So với FLUX.1 Kontext, Qwen-Image-Edit, Step1X-Edit | PIE-Bench subset hoặc custom set | **Trung bình-thấp** | So sánh baseline |
| 9 | **Vietnamese prompt support** | Hỗ trợ prompt tiếng Việt bằng dịch prompt hoặc multilingual CLIP | Custom bilingual prompt set | **Trung bình** | Ứng dụng |
| 10 | **Apple Silicon / on-device optimization** | Tối ưu Mac MPS/Core ML/quantization cho chạy local | Custom latency benchmark trên Mac | **Trung bình-thấp** | Engineering |
| 11 | **ControlNet / reference-guided style editing** | Thêm điều kiện edge/depth/reference để giữ layout/style tốt hơn | Custom style set | **Trung bình-thấp** | Nghiên cứu dài hạn |
| 12 | **Fine-tune nhẹ Stage 2** | Fine-tune checkpoint trên domain nhỏ bằng Colab | Custom domain set, CommonCanvas nếu có | **Thấp-trung bình** | Nghiên cứu optional |
| 13 | **Thay one-step generator/backbone** | Thay SBv2 bằng generator mới hơn | PIE-Bench sau khi train/tích hợp lại | **Thấp** | Nghiên cứu dài hạn |
| 14 | **Gradio realtime demo** | Làm UI upload ảnh, nhập prompt, hiện output và runtime | Không cần benchmark học thuật | **Cao** | Ứng dụng/demo |

## 2. Khuyến nghị chọn hướng

**Hướng chính nên chọn:** **SwiftEdit-RT**. Hướng này khớp nhất với tinh thần paper: SwiftEdit bán điểm mạnh là nhanh, nên mở rộng hợp lý nhất là đo bottleneck và giảm latency mà không làm giảm chất lượng.

**Hướng ứng dụng phụ nên chọn:** **Object removal / inpainting**. Hướng này dễ demo, ảnh before/after rõ, metric cũng rõ hơn global style vì vẫn là local edit có mask.

**Hướng phân tích giới hạn nên làm nếu còn thời gian:** **Global scene/style/weather editing**. Hướng này trả lời câu hỏi SwiftEdit có vượt ra ngoài local semantic edit được không, nhưng cần đổi bộ metric vì mask/PSNR-background không còn phù hợp.

**Các hướng nên để optional:** SAM 3, VLM auto-caption, baseline hiện đại, ControlNet, fine-tune, Core ML. Các hướng này có giá trị phân tích nhưng dễ thêm latency, dependency, chi phí GPU/API hoặc vượt phạm vi môn học.

---

## 3. Chi tiết từng hướng

### 3.1. SwiftEdit-RT: realtime inference acceleration

**Mô tả ngắn:**  
Tối ưu tốc độ suy luận của SwiftEdit mà không train lại và không thêm model nặng vào đường realtime. Trọng tâm là tìm bottleneck sau khi SwiftEdit đã rút diffusion về 1 inversion step + 1 editing step.

**Benchmark nếu có:**

- PIE-Bench subset 20-100 mẫu để kiểm tra chất lượng trước/sau tối ưu.
- Custom latency benchmark với model đã load sẵn.
- Interactive benchmark: cùng 1 ảnh nguồn, chạy nhiều edit prompt liên tiếp.

**Độ đo dùng để benchmark:**

| Nhóm | Độ đo | Ý nghĩa |
|---|---|---|
| Tốc độ | `runtime_total_s`, p50/p95 latency | Tổng thời gian mỗi edit |
| Bottleneck | `vae_encode_s`, `text_encoder_s`, `inverse_unet_s`, `mask_s`, `clip_image_encoder_s`, `generation_unet_s`, `vae_decode_s` | Module nào chậm |
| Tài nguyên | peak memory MB/GB | VRAM/RAM tối đa |
| Speedup | baseline time / optimized time | Tối ưu nhanh hơn bao nhiêu lần |
| Chất lượng | PSNR-unedit, MSE-unedit, CLIP-Whole, CLIP-Edited | Tối ưu có làm lệch kết quả không |
| Sai khác ảnh | LPIPS/SSIM hoặc diff baseline vs optimized | Cần khi dùng fp16/TinyVAE |

**Các bước implement:**

1. Thêm profiler theo module trong `edit_image()` và `gen_img()`.
2. Đo baseline trên Mac MPS và Colab CUDA với model đã load sẵn.
3. Giữ patch đã làm: bỏ decode `noise_image` khi caller không dùng.
4. ~~Patch self-guided mask threshold vectorized trên GPU, tránh `.cpu().apply_()`.~~ ✅ Done 2026-06-14
5. Thêm cache latent ảnh nguồn, CLIP image embedding, source prompt embedding cho demo cùng ảnh nhiều prompt.
6. Thử `channels_last`, `fp16`/`bf16`, `torch.compile` trên CUDA.
7. Thử TinyVAE/TAESD như ablation riêng vì có rủi ro đổi output.
8. Lập bảng latency breakdown + speedup + quality metrics.

**Độ khả thi:** **Cao**. Bắt đầu được bằng patch nhỏ, không cần train, có metric rõ. Rủi ro chính nằm ở tối ưu CUDA/MPS và tương thích dependency.

**Loại:** **Nghiên cứu + engineering**. Đây nên là hướng chính.

**Ưu tiên con trong SwiftEdit-RT:**

| Ưu tiên | Tối ưu | Khả thi | Rủi ro chất lượng | Ghi chú |
|---:|---|---:|---:|---|
| 0 | Bỏ decode `noise_image` không dùng | Rất cao | Không | Đã implement; cần đo speedup |
| 1 | Vectorized self-guided mask trên GPU | Rất cao | Không/rất thấp | ✅ 2026-06-14: 12.2→4.6 ms (~2.6×, −7.6 ms/ảnh); mask giống hệt baseline; chỉ ~0.02% tổng pipeline (~72s) nên runtime ~không đổi |
| 2 | Profiler latency theo module | Cao | Không | Bắt buộc để chứng minh |
| 3 | Cache latent/image/source embedding | Cao | Không nếu cache đúng | Rất hợp demo realtime |
| 4 | `channels_last` + fp16/bf16 | Trung bình | Thấp | Thử trên Colab CUDA |
| 5 | `torch.compile` | Trung bình | Thấp | Có warmup/compile overhead |
| 6 | TinyVAE/TAESD | Trung bình | Trung bình | Có thể nhanh hơn nhưng đổi màu/chi tiết |
| 7 | TensorRT/Core ML/quantization | Thấp-trung bình | Thấp-trung bình | Hướng dài hơn |

**Kết quả tối ưu đã đo (Mac M4 / MPS, steady-state):**

| Tối ưu | Trước | Sau | Speedup stage | Tiết kiệm tuyệt đối | % tổng pipeline | Tác động end-to-end |
|---|---:|---:|---:|---:|---:|---|
| Bỏ decode `noise_image` | — | — | — | ~0.0001 s | ~0% | Không đáng kể (caller đã không dùng) |
| Vectorized GPU mask | 12.2 ms | 4.6 ms | ~2.6× | ~7.6 ms/ảnh | ~0.02% | ~Không đổi (~72 s/ảnh); lợi ích tăng theo độ phân giải |

**Phân bố thời gian (run 20 mẫu, để biết nên tối ưu tiếp ở đâu):**

| Stage | % tổng |
|---|---:|
| gen_image_embeds (IP-Adapter) | ~23.5% |
| gen_vae_decode | ~23.2% |
| gen_unet (sinh ảnh) | ~21.7% |
| unet_inverse | ~21.5% |
| inv_text_encode | ~7.5% |
| mask_estimate | ~0.02% |

> Nhận định: các tối ưu mask/noise-decode đúng về kỹ thuật nhưng tác động end-to-end rất nhỏ. Muốn giảm runtime thật sự phải nhắm vào **UNet (2 lần) + IP image embeds + VAE decode** — ưu tiên cache embedding (bước 3), rồi fp16/channels_last/compile (bước 4–5).

### 3.2. Object removal / inpainting

**Mô tả ngắn:**  
Dùng SwiftEdit để xóa vật thể: prompt source có object, prompt edit không còn object. Vì đây là local edit, self-guided mask và ARaM vẫn có ý nghĩa.

**Benchmark nếu có:**

- Custom 20-40 ảnh: người, xe, chai/lon, biển báo, rác, vật trên bàn.
- COCO hoặc tập có segmentation mask để lấy object mask.
- PIE-Bench subset có dạng remove/delete object nếu chọn được.
- Baseline LaMa với cùng mask nếu có thời gian.
- ReMOVE có thể tham khảo như metric reference-free cho object erasure.

**Độ đo dùng để benchmark:**

| Nhóm | Độ đo | Ý nghĩa |
|---|---|---|
| Xóa thành công | Detector confidence drop | Confidence của object sau edit phải giảm |
| Đúng prompt | CLIPScore / CLIP margin | Ảnh edited hợp với `"without object"` hơn `"with object"` |
| Giữ nền | PSNR/MSE/SSIM/LPIPS trên vùng ngoài mask `(1 - mask)` | Vùng không xóa không bị phá |
| Tự nhiên vùng lấp | Human rating, crop review, MUSIQ/NIQE | Vùng inpaint có viền/ghosting/artifact không |
| Chất lượng mask | IoU/Dice self-guided mask vs GT/user mask | Mask có bám đúng object không |
| Tốc độ | runtime SwiftEdit vs LaMa | SwiftEdit có lợi thế latency không |

**Các bước implement:**

1. Chọn 20-40 ảnh có object rõ, chia theo object nhỏ/vừa/lớn và nền đơn giản/phức tạp.
2. Viết prompt dạng `"a scene with [object]"` -> `"the same scene without [object]"`.
3. Chạy SwiftEdit-SG với self-guided mask và lưu ảnh/mask visualization.
4. Nếu có object mask, chạy SwiftEdit-UserMask/GTMask.
5. Thử SwiftEdit-SG+Dilate để giảm viền/ghosting.
6. Nếu kịp, chạy LaMa baseline với cùng mask.
7. Tính detector confidence drop, CLIP margin, PSNR/SSIM/LPIPS ngoài mask, runtime.
8. Làm bảng failure cases: ghost object, nền méo, xóa nhầm, biên mask xấu.

**Độ khả thi:** **Trung bình-cao**. Tốt với object nhỏ/vừa và background đơn giản. Thấp hơn với object lớn vì SwiftEdit không phải inpainting model chuyên dụng.

**Loại:** **Ứng dụng + nghiên cứu**. Rất hợp làm hướng ứng dụng phụ để báo cáo.

### 3.3. Global scene/style/weather editing

**Mô tả ngắn:**  
Dùng SwiftEdit để đổi thuộc tính toàn ảnh như ngày -> đêm, mùa hè -> mùa đông, nắng -> mưa, overcast/golden hour, warm/cold tone.

**Benchmark nếu có:**

- Không có benchmark chuẩn sẵn trong repo cho bài toán global style.
- Custom 20-40 ảnh outdoor/street/landscape.
- Nếu có tập ảnh target-domain đủ lớn, có thể dùng FID/KID giữa edited set và ảnh thật target domain.

**Độ đo dùng để benchmark:**

| Nhóm | Độ đo | Ý nghĩa |
|---|---|---|
| Đúng style/target | CLIPScore với edit prompt | Ảnh có hợp prompt ngày/đêm/mùa/nắng không |
| Phân biệt target | Zero-shot CLIP label, Delta CLIP target | Label target tăng sau edit không |
| Giữ content/layout | DINO similarity, CLIP image-image similarity | Bố cục/semantic gốc có được giữ không |
| Perceptual distance | LPIPS/SSIM toàn ảnh | Chỉ dùng phụ vì global edit được phép đổi màu/ánh sáng |
| Ảnh tự nhiên | MUSIQ/NIQE, human rating 1-5 | Có artifact, đổi không đều, ảnh giả không |
| Domain realism | FID/KID nếu có target-domain set | Edited set có giống ảnh thật của domain target không |
| Tốc độ | runtime_s | Vẫn cần giữ luận điểm realtime |

**Các bước implement:**

1. Chọn 20-40 ảnh outdoor/street/landscape.
2. Tạo 4 nhóm prompt: day-night, season, weather, tone/mood.
3. Chạy prompt nhẹ và prompt mạnh trên cùng ảnh.
4. Chạy SwiftEdit-SG; nếu có external mask, thử full-image mask.
5. Tính CLIP target/zero-shot label, DINO/CLIP image similarity, LPIPS/SSIM phụ, IQA/human rating, runtime.
6. Phân loại failure cases: under-edit, đổi không đều, mất layout, artifact ánh sáng/mưa/tuyết.

**Độ khả thi:** **Trung bình**. Edit nhẹ như sunny -> overcast, day -> night vừa phải có thể làm được. Mùa đông/tuyết, mưa lớn, đêm có đèn/phản chiếu khó hơn.

**Loại:** **Ứng dụng + phân tích giới hạn**. Không nên là hướng chính nếu mục tiêu là kết quả chắc ăn điểm.

### 3.4. Self-guided mask analysis

**Mô tả ngắn:**  
Phân tích mask tự sinh của SwiftEdit có định vị đúng vùng cần edit không, và mask ảnh hưởng thế nào đến background preservation/prompt fidelity.

**Benchmark nếu có:** PIE-Bench, vì có source/edit prompt, ảnh, GT mask và 10 loại edit.

**Độ đo dùng để benchmark:**

| Nhóm | Độ đo | Ý nghĩa |
|---|---|---|
| Chất lượng mask | IoU, Dice | Self-guided mask có trùng GT mask không |
| Giữ nền | PSNR/MSE trên `(1 - mask)` | Mask tốt có giữ background tốt hơn không |
| Đúng prompt | CLIP-Whole, CLIP-Edited | Edit có đúng yêu cầu không |
| Tốc độ | mask_s, runtime_total_s | Mask tự sinh tốn bao nhiêu latency |

**Các bước implement:**

1. Tải PIE-Bench subset có GT mask.
2. Chạy SwiftEdit với self-guided mask và lưu mask tự sinh.
3. Decode GT mask về cùng kích thước 512x512.
4. Tính IoU/Dice giữa self-guided mask và GT mask.
5. Chạy thêm cấu hình GT mask nếu code hỗ trợ external mask.
6. So sánh SwiftEdit-SG vs SwiftEdit-GTMask bằng PSNR/MSE/CLIP/runtime.

**Độ khả thi:** **Cao**. Repo đã có script metric PIE-Bench, chỉ cần thêm/hoàn thiện phần log mask tự sinh nếu chưa có.

**Loại:** **Nghiên cứu/đánh giá**. Hợp để giải thích ARaM và pipeline.

### 3.5. SAM 3 / GroundingDINO+SAM mask analysis

**Mô tả ngắn:**  
Dùng segmentation model sinh mask từ concept prompt, rồi so với self-guided mask của SwiftEdit. Sau khi đổi hướng sang realtime, hướng này chỉ nên là offline analysis, không đưa vào pipeline realtime chính.

**Benchmark nếu có:** PIE-Bench GT mask, custom object masks, hoặc object removal set.

**Độ đo dùng để benchmark:**

| Nhóm | Độ đo | Ý nghĩa |
|---|---|---|
| Mask quality | IoU/Dice SAM vs GT, SwiftEdit-SG vs GT | Model nào định vị vùng edit tốt hơn |
| Editing quality | PSNR/MSE background, CLIP-Whole/Edited | Mask tốt có làm edit tốt hơn không |
| Chi phí | `runtime_sam_s`, `runtime_swiftedit_s`, `runtime_total_s`, peak memory | Thêm SAM có đáng với latency không |

**Các bước implement:**

1. Chọn 5-20 mẫu có concept/object rõ.
2. Trích concept từ edit prompt hoặc viết tay.
3. Chạy SAM 3 offline để sinh mask, cache mask ra file.
4. Chạy SwiftEdit với self-guided mask, GT mask và SAM mask.
5. Tính IoU/Dice, PSNR/MSE/CLIP và runtime riêng từng module.
6. Kết luận rõ: SAM 3 có thể tốt hơn mask, nhưng làm chậm pipeline end-to-end.

**Độ khả thi:** **Trung bình**. Ý tưởng rõ nhưng thêm dependency/model nặng, có nguy cơ tốn Colab/GPU setup.

**Loại:** **Nghiên cứu optional**, không nên chọn làm hướng chính.

### 3.6. Auto source prompt / concept extraction

**Mô tả ngắn:**  
Dùng VLM như Florence-2 hoặc Qwen2.5-VL để tự sinh source prompt/caption hoặc concept cần edit, giảm phụ thuộc vào người dùng viết source prompt.

**Benchmark nếu có:** PIE-Bench để so source prompt gốc với source prompt sinh tự động; custom prompt set với ảnh tự thu thập.

**Độ đo dùng để benchmark:**

| Nhóm | Độ đo | Ý nghĩa |
|---|---|---|
| Chất lượng edit | PSNR/MSE background, CLIP-Whole/Edited | Source prompt tự động có làm edit kém prompt tay không |
| Chất lượng caption | CLIP image-text score hoặc human rating | Caption có mô tả đúng ảnh gốc không |
| Mask | IoU/Dice self-guided mask | Source prompt tốt có giúp mask tốt hơn không |
| Chi phí | runtime_vlm_s + runtime_swiftedit_s | VLM thêm latency bao nhiêu |

**Các bước implement:**

1. Chọn 20-50 ảnh có source prompt tay.
2. Chạy VLM để sinh caption/source prompt.
3. Chạy SwiftEdit với source prompt tay và source prompt tự động.
4. So sánh PSNR/MSE/CLIP/mask/runtime.
5. Nếu latency cao, đặt VLM thành preprocessing offline, không tính vào realtime pipeline.

**Độ khả thi:** **Trung bình**. Dễ demo, nhưng thêm latency và dependency lớn.

**Loại:** **Ứng dụng optional**. Hợp nếu muốn hệ thống dễ dùng hơn, không hợp nếu mục tiêu là tốc độ end-to-end.

### 3.7. Grounded evaluation beyond CLIP

**Mô tả ngắn:**  
Bổ sung cách đánh giá edit có đúng đối tượng/thuộc tính thật không, vì CLIPScore có thể cao nhưng ảnh vẫn sai chi tiết.

**Benchmark nếu có:** Custom grounded set, mỗi mẫu có edit instruction và câu hỏi kiểm tra như "car removed?", "dog mouth opened?", "is it nighttime?". Có thể dùng GIE-style protocol nếu tìm được benchmark phù hợp.

**Độ đo dùng để benchmark:**

| Nhóm | Độ đo | Ý nghĩa |
|---|---|---|
| Functional correctness | Detector/attribute classifier accuracy | Object/attribute target có xuất hiện/biến mất không |
| VQA correctness | VQA yes/no accuracy | Trả lời câu hỏi về kết quả edit |
| Text-image | CLIPScore/CLIP margin | So với metric cũ |
| Human eval | success/partial/fail | Kiểm tra mẫu khó/ambiguous |

**Các bước implement:**

1. Tạo 20-50 mẫu có instruction rõ và câu hỏi kiểm tra.
2. Chọn tool kiểm tra: detector, classifier, VQA hoặc human rating.
3. Chạy SwiftEdit và ghi output.
4. Tính success rate và so với CLIPScore để xem CLIP có bị ảo không.
5. Đưa vào báo cáo như phần "beyond CLIP".

**Độ khả thi:** **Trung bình**. Dùng human rating thì dễ; dùng detector/VQA tự động thì thêm model và công setup.

**Loại:** **Nghiên cứu/đánh giá**.

### 3.8. Modern baseline comparison

**Mô tả ngắn:**  
So sánh SwiftEdit với các editor hiện đại như FLUX.1 Kontext, Qwen-Image-Edit, Step1X-Edit để đặt SwiftEdit vào bối cảnh mới.

**Benchmark nếu có:** PIE-Bench subset nếu baseline cho phép batch inference; custom 10-30 mẫu chung nếu chỉ so định tính.

**Độ đo dùng để benchmark:**

| Nhóm | Độ đo | Ý nghĩa |
|---|---|---|
| Chất lượng | CLIPScore, human rating, failure cases | Model nào đúng prompt và ảnh đẹp hơn |
| Giữ nền | PSNR/SSIM/LPIPS ngoài mask nếu có mask | Model nào phá background ít hơn |
| Tốc độ | runtime_s, chi phí API/GPU | SwiftEdit có lợi thế tốc độ/chi phí không |
| Tài nguyên | VRAM, model size | Khả năng chạy local |

**Các bước implement:**

1. Chọn 10-30 mẫu chung.
2. Chạy SwiftEdit local/Colab.
3. Chạy baseline bằng API/demo/weights nếu khả thi.
4. Lập grid ảnh side-by-side và bảng runtime/chi phí.
5. Nếu không có metric đồng nhất, ghi rõ chỉ so sánh định tính.

**Độ khả thi:** **Trung bình-thấp**. Phụ thuộc API, license, VRAM và khả năng chạy baseline.

**Loại:** **So sánh baseline**, nên làm nếu còn thời gian.

### 3.9. Vietnamese prompt support

**Mô tả ngắn:**  
Cho phép người dùng nhập prompt tiếng Việt. Cách thực tế nhất là dịch prompt Việt -> Anh trước khi đưa vào SwiftEdit; thay text encoder bằng multilingual CLIP là hướng khó hơn.

**Benchmark nếu có:** Custom bilingual set: cùng một ảnh, prompt tiếng Anh và tiếng Việt tương đương.

**Độ đo dùng để benchmark:**

| Nhóm | Độ đo | Ý nghĩa |
|---|---|---|
| Chất lượng edit | CLIP-Whole/Edited với prompt Anh đã dịch | Prompt Việt có đạt kết quả như prompt Anh không |
| Human rating | Đúng ý/câu lệnh, tự nhiên, giữ nền | Cần vì CLIP tiếng Việt có thể yếu |
| Tốc độ | translation_s + swiftedit_s | Dịch prompt thêm bao nhiêu latency |

**Các bước implement:**

1. Tạo 20-30 cặp prompt Việt/Anh.
2. Cách đơn giản: dịch prompt Việt sang Anh bằng model/API hoặc dịch tay.
3. Chạy SwiftEdit với prompt Anh gốc và prompt Việt đã dịch.
4. So sánh output và runtime.
5. Nếu thử multilingual CLIP/text encoder, tách thành ablation riêng vì có rủi ro không tương thích checkpoint.

**Độ khả thi:** **Trung bình** với cách dịch prompt; **thấp** nếu thay text encoder.

**Loại:** **Ứng dụng**.

### 3.10. Apple Silicon / on-device optimization

**Mô tả ngắn:**  
Tối ưu chạy SwiftEdit trên Mac M-series/MPS hoặc xuất sang Core ML/quantization để hướng đến on-device editing.

**Benchmark nếu có:** Custom latency benchmark trên Mac MPS; so sánh Mac MPS vs Colab T4 vs paper A100.

**Độ đo dùng để benchmark:**

| Nhóm | Độ đo | Ý nghĩa |
|---|---|---|
| Tốc độ | cold start, warm inference, p50/p95 latency | Trải nghiệm local |
| Tài nguyên | RAM/unified memory peak, model size | Có chạy được trên máy cá nhân không |
| Chất lượng | So output với baseline CUDA nếu có | Tối ưu có làm lệch output không |

**Các bước implement:**

1. Đo baseline Mac MPS hiện tại.
2. Tách cold start và warm inference.
3. Thử các tối ưu nhẹ: cache, giảm decode thừa, giảm CPU sync.
4. Nếu còn thời gian, khảo sát Core ML/quantization cho từng module.
5. Báo cáo rõ khác biệt Mac MPS vs CUDA, không so trực tiếp với A100 nếu điều kiện khác nhau.

**Độ khả thi:** **Trung bình-thấp**. Đo latency trên Mac thì dễ; Core ML/quantization cho pipeline phức tạp thì tốn công.

**Loại:** **Engineering/on-device**.

### 3.11. ControlNet / reference-guided style editing

**Mô tả ngắn:**  
Thêm điều kiện edge/depth/reference image để giữ layout tốt hơn khi làm global style/weather edit.

**Benchmark nếu có:** Custom style/weather set, dùng lại metric của global style editing.

**Độ đo dùng để benchmark:**

| Nhóm | Độ đo | Ý nghĩa |
|---|---|---|
| Giữ layout | DINO similarity, SSIM, edge/depth consistency | Control/reference có giữ cấu trúc tốt hơn không |
| Đúng style | CLIP target, zero-shot CLIP label | Có đổi đúng style không |
| Tự nhiên | human rating, MUSIQ/NIQE | Ảnh có artifact không |
| Tốc độ | runtime_total_s | Thêm ControlNet/reference làm chậm bao nhiêu |

**Các bước implement:**

1. Xác định có thể tích hợp ControlNet/reference vào backbone hiện tại không.
2. Nếu không tương thích, chỉ để làm related-work/hướng tương lai.
3. Nếu tích hợp được, chạy style/weather set trước/sau ControlNet.
4. Đo CLIP target, DINO/SSIM và runtime.

**Độ khả thi:** **Trung bình-thấp**. Có nguy cơ vượt phạm vi vì SwiftEdit checkpoint không được train với ControlNet.

**Loại:** **Nghiên cứu dài hạn**.

### 3.12. Fine-tune nhẹ Stage 2

**Mô tả ngắn:**  
Fine-tune nhẹ trên domain nhỏ để xem reconstruction/editing có cải thiện không. Chỉ nên làm trên Colab/GPU, không làm trên Mac.

**Benchmark nếu có:** Custom domain set 200-500 ảnh + caption; CommonCanvas nếu dùng lại được theo setup gốc; PIE-Bench/custom subset trước-sau fine-tune.

**Độ đo dùng để benchmark:**

| Nhóm | Độ đo | Ý nghĩa |
|---|---|---|
| Reconstruction | PSNR, LPIPS, SSIM | Fine-tune có tái tạo ảnh domain tốt hơn không |
| Editing | CLIP-Whole/Edited, human rating | Edit có tốt hơn không |
| Overfit | Kết quả trên ảnh ngoài domain | Có bị kém tổng quát không |
| Tài nguyên | training time, VRAM, checkpoint size | Có đáng làm trong môn học không |

**Các bước implement:**

1. Chuẩn bị 200-500 ảnh + caption.
2. Chạy Stage 2 vài nghìn iterations trên Colab.
3. Lưu checkpoint riêng, không ghi đè checkpoint gốc.
4. Chạy cùng subset trước/sau fine-tune.
5. Báo cáo reconstruction/editing/failure cases.

**Độ khả thi:** **Thấp-trung bình**. Có thể OOM/T4 chậm; bỏ qua vẫn đủ cho đề tài.

**Loại:** **Nghiên cứu optional**.

### 3.13. Thay one-step generator/backbone

**Mô tả ngắn:**  
Thay SBv2 bằng one-step generator mới hơn hoặc backbone khác để tăng chất lượng generation/editing.

**Benchmark nếu có:** PIE-Bench sau khi tích hợp và/hoặc train lại; custom set để so sánh chất lượng.

**Độ đo dùng để benchmark:** PSNR/MSE/CLIP/LPIPS, runtime, VRAM/model size/training cost.

**Các bước implement:**

1. Khảo sát generator mới có API/latent space tương thích không.
2. Nếu không tương thích, cần train/tune inversion network và IP-Adapter lại.
3. Chạy benchmark sau khi tích hợp.

**Độ khả thi:** **Thấp** trong phạm vi môn học. Rủi ro phải train lại nhiều module.

**Loại:** **Nghiên cứu dài hạn**, không nên chọn.

### 3.14. Gradio realtime demo

**Mô tả ngắn:**  
Làm giao diện demo: upload ảnh, nhập source/edit prompt, hiện output, mask và runtime.

**Benchmark nếu có:** Không cần benchmark học thuật; có thể đo app latency và memory local.

**Độ đo dùng để benchmark:** `runtime_total_s`, p50/p95 latency, RAM peak, success/partial/fail human label.

**Các bước implement:**

1. Tạo Gradio app upload ảnh + source prompt + edit prompt.
2. Hiện edited image, optional self-guided mask và runtime.
3. Thêm mode cache cùng ảnh nhiều prompt nếu đã làm SwiftEdit-RT.
4. Lưu output demo vào folder riêng để đưa vào slide.

**Độ khả thi:** **Cao**. Tốt để trình bày, nhưng không thay thế được phần benchmark/nghiên cứu.

**Loại:** **Ứng dụng/demo**.

---

## 4. Gợi ý cấu trúc benchmark chung

### 4.1. Benchmark chính cho SwiftEdit/PieBench

| Metric | Cách đo | Hướng tốt |
|---|---|---|
| PSNR-unedit | PSNR giữa source và edited trên background `(1 - mask)` | Cao |
| MSE-unedit | Sai số pixel trên background `(1 - mask)` | Thấp |
| CLIP-Whole | CLIPScore giữa toàn ảnh edited và `edit_prompt` | Cao |
| CLIP-Edited | CLIPScore giữa vùng edit mask và `edit_prompt` | Cao |
| IoU/Dice | Self-guided mask vs GT mask | Cao |
| Runtime | Thời gian gọi edit với model đã load | Thấp |

### 4.2. Benchmark cho hướng tốc độ

| Config | Device | Runtime total | Speedup | Peak memory | PSNR | MSE | CLIP-Whole | CLIP-Edited |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base | Mac/Colab |  | 1.00x |  |  |  |  |  |
| No-noise-decode | Mac/Colab |  |  |  |  |  |  |  |
| GPU-mask | Colab |  |  |  |  |  |  |  |
| Cache | Mac/Colab |  |  |  |  |  |  |  |
| fp16/channels_last | Colab |  |  |  |  |  |  |  |
| TinyVAE | Colab |  |  |  |  |  |  |  |

### 4.3. Benchmark cho hướng ứng dụng

| Hướng | Dataset tối thiểu | Metric chính | Metric phụ |
|---|---|---|---|
| Object removal | 20-40 ảnh có object | Detector confidence drop, CLIP margin | PSNR/SSIM/LPIPS ngoài mask, human rating, runtime |
| Global style/weather | 20-40 ảnh outdoor | CLIP target, zero-shot CLIP label | DINO similarity, LPIPS/SSIM, MUSIQ/NIQE, human rating, runtime |
| Vietnamese prompt | 20-30 cặp Việt/Anh | Human rating, CLIP với prompt Anh dịch | runtime translation + edit |
| Demo app | 5-10 case thuyết phục | runtime, success/partial/fail | RAM peak |

---

## 5. Kết luận chọn hướng

Nếu chỉ chọn **1 hướng chính**, nên chọn:

> **SwiftEdit-RT: realtime inference acceleration**

Lý do: khả thi cao, không cần train, có metric rõ, bảo vệ điểm mạnh realtime của SwiftEdit.

Nếu chọn **1 hướng ứng dụng phụ**, nên chọn:

> **Object removal / inpainting**

Lý do: ứng dụng dễ hiểu, có ảnh before/after thuyết phục, vẫn dùng được mask/ARaM, metric rõ hơn global style.

Nếu cần **1 hướng phân tích giới hạn**, chọn:

> **Global scene/style/weather editing**

Lý do: trả lời được câu hỏi SwiftEdit có vượt ra ngoài local semantic edit được không, nhưng cần chấp nhận metric khác PIE-Bench và rủi ro under-edit.

## 6. Tài liệu/benchmark liên quan

- [SwiftEdit CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_SwiftEdit_Lightning_Fast_Text-Guided_Image_Editing_via_One-Step_Diffusion_CVPR_2025_paper.pdf)
- [SwiftEdit repository](https://github.com/Qualcomm-AI-research/SwiftEdit)
- [PIE-Bench / PnP Inversion ICLR 2024](https://github.com/cure-lab/PnPInversion)
- [CLIP ICML 2021](https://proceedings.mlr.press/v139/radford21a)
- [CLIPScore EMNLP 2021](https://arxiv.org/abs/2104.08718)
- [SAM 3](https://github.com/facebookresearch/sam3)
- [Florence-2](https://www.microsoft.com/en-us/research/publication/florence-2-advancing-a-unified-representation-for-a-variety-of-vision-tasks/)
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)
- [FLUX.1 Kontext](https://arxiv.org/abs/2506.15742)
- [Step1X-Edit](https://arxiv.org/abs/2504.17761)
- [Qwen-Image-Edit](https://qwenlm.github.io/zh/blog/qwen-image-edit/)
- [AutoencoderTiny / TAESD](https://huggingface.co/docs/diffusers/en/api/models/autoencoder_tiny)
- [Diffusers optimization guide](https://huggingface.co/docs/diffusers/main/optimization/fp16)
- [PyTorch torch.compile and Diffusers](https://docs.pytorch.org/devlogs/inductor/2026-05-11-torch-compile-and-diffusers/)
- [TensorRT Stable Diffusion acceleration](https://developer.nvidia.com/blog/tensorrt-accelerates-stable-diffusion-nearly-2x-faster-with-8-bit-post-training-quantization/)
- [DINO](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper)
- [LPIPS](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0299.pdf)
- [MUSIQ](https://mlanthology.org/iccv/2021/ke2021iccv-musiq/)
- [FID](https://papers.nips.cc/paper/7240-gans-trained-by-a-two-time-scale-update-rule-converge-to-a-local-nash-equilibrium)
- [LaMa WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.html)
- [ReMOVE object erasure metric](https://arxiv.org/abs/2409.00707)
