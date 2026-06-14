# Nhật ký làm việc — CS2309 SwiftEdit

> Ghi chép tiến độ đề tài. Tự động cập nhật bởi skill `.cursor/skills/update-readme-progress/`.
> Đồng bộ tóm tắt sang [`SwiftEdit_DeTai_CS2309.md`](./SwiftEdit_DeTai_CS2309.md) Mục 8.1.

---

## Tóm tắt nhanh

| Ngày | Giai đoạn | Công việc | Kết quả / Ghi chú | Môi trường |
|---|---|---|---|---|
| 2026-06-14 | 3c | Đo timing từng công đoạn (StageTimer) + eval PIE-Bench subset 20 mẫu | 20 mẫu/Apple M4 MPS: TB 69.0s/ảnh (steady 73.6s); UNet x2 ~43%, IP embeds ~24%, VAE decode ~23%; CLIP-Whole 23.02, CLIP-Edited 21.46, PSNR nền 14.01 (9/20); lưu experimental_data/piebench_subset20_2026-06-14/ | Mac M4 (MPS); .venv; torch 2.12.0 |
| 2026-06-05 | 4d | Đề xuất hướng ứng dụng xóa vật thể / inpainting | Cập nhật đề tài/README/Overview/QA: dùng SwiftEdit để xóa object bằng prompt + self/user mask; đánh giá khả thi trung bình-cao với object nhỏ/vừa; metric detector confidence drop, CLIP margin, PSNR/SSIM/LPIPS ngoài mask, realism/human rating; so LaMa nếu kịp. | Mac M4; tài liệu |
| 2026-06-05 | 4c | Đề xuất hướng ứng dụng global style/weather edit | Cập nhật đề tài/README/Overview/QA: dùng SwiftEdit cho ngày↔đêm, mùa, mưa↔nắng; đánh giá khả thi trung bình; bỏ mask metric, dùng CLIP target, zero-shot CLIP label, DINO/CLIP image similarity, LPIPS/SSIM phụ, IQA/human rating, FID/KID nếu có target domain. | Mac M4; tài liệu |
| 2026-06-05 | 4d | Chốt hướng SwiftEdit-RT đầu tiên: bỏ decode `noise_image` | Patch `IPSBV2Model.gen_img()` thêm `return_noise_image=False`, mặc định không decode ảnh nhiễu không dùng; cập nhật README/QA/đề tài với ranking ưu tiên các hướng tăng tốc không train. | Mac M4; code + tài liệu |
| 2026-06-05 | 4d | Đổi hướng đào sâu sang SwiftEdit-RT realtime inference acceleration | Cập nhật đề tài/README/Overview/QA: loại SAM 3 khỏi pipeline chính vì tăng latency; chọn hướng profile bottleneck, GPU mask threshold, bỏ decode thừa, cache latent/embedding, thử fp16/channels_last/torch.compile và TinyVAE/TAESD. | Mac M4; tài liệu |
| 2026-06-05 | 4d | Khảo sát ban đầu: SwiftEdit + SAM 3 concept-guided mask replacement | Đã từng cân nhắc SAM 3 để thay self-guided mask; sau phản biện realtime, hướng này chuyển thành optional/offline mask analysis thay vì hướng chính. | Mac M4; tài liệu |
| 2026-06-05 | 3c | Chuẩn hóa tài liệu đánh giá SwiftEdit/PieBench và sửa `CLIP-Whole` trong metric code | Ghi rõ PSNR/MSE vùng nền, CLIP-Whole/Edited theo edit prompt, runtime; nguồn: PIE-Bench/PnP Inversion ICLR 2024, SwiftEdit CVPR 2025, CLIP/CLIPScore. `piebench_metrics.py` đã sửa `clip_whole` dùng `edit_prompt`; compileall OK. | Mac M4 (MPS); .venv |
| 2026-06-05 | 3c | Viết scripts đánh giá PIE-Bench (`piebench_utils.py`, `piebench_metrics.py`, `run_piebench_eval.py`) | Smoke eval 1 mẫu chạy OK trên Mac MPS (~50s/ảnh, CLIP-Whole=18.76, CLIP-Edited=22.64); `metrics.csv` sinh đúng; PIE-Bench đầy đủ tải qua form PnP Inversion. | Mac M4 (MPS); .venv; torchmetrics 1.9.0 |
| 2026-06-05 | 2a. Mac | Chạy preset 02.jpg dog→dog with mouth opened trên Mac MPS (cùng prompt Colab T4) | edit_image 30.1s; results/notebook/nb_dog_dog_to_dog_with_mouth_opened.png; T4 1.3s cùng preset → Mac ~23× chậm hơn T4; | Mac M4 (MPS); pyenv 3.12.10; .venv |
| 2026-06-04 | 2b. Colab | Chạy notebook CS2309_SwiftEdit_test trên Colab T4 (extension): preset dog→dog wi | edit_image 1.3s; output results/notebook/nb_dog_dog_to_dog_with_mouth_opened.png; so với Mac MPS ~91s (woman) và paper A | Google Colab — Tesla T4 (Colab extension) |
| 2026-06-04 | 2b. Colab | Patch notebook + requirements (GPU T4, HF stack mới, upload path) | Sẵn sàng chạy Colab extension; fix EncoderDecoderCache/numpy/upload; chưa log runtime T4 OK | Colab extension + Colab web |
| 2026-06-04 | 2a. Mac | Notebook notebooks/CS2309_SwiftEdit_test.ipynb (preset, upload ipywidgets 8, inf | Notebook chạy OK trên Mac (.venv); upload widget sửa tuple ipywidgets 8 | Mac M4 (MPS); Jupyter .venv |
| 2026-06-04 | 2a. Mac | Clone SwiftEdit; pyenv 3.12.10 + .venv + requirements-mac.txt (PyTorch MPS) | Demo woman→Taylor Swift OK; output SwiftEdit/result_woman->Taylor Swift.png; ~91s/ảnh trên MPS | Mac M4 (MPS); pyenv 3.12.10; .venv |
| 2026-06-01 | 0. Khởi tạo project | Tạo README, đề tài chi tiết, skill Cursor hỗ trợ nhật ký | Repo CS2309.CH201 sẵn sàng; skill sync README + NHAT_KY + §8.1 | Mac M4 |

---

## Chi tiết theo phiên làm việc

*(Các entry chi tiết xuất hiện bên dưới, mới nhất ở trên cùng.)*

### 2026-06-14 — [3c] Đo timing từng công đoạn + eval PIE-Bench subset 20 mẫu (Mac M4)

**Môi trường:** Mac M4 (MPS); .venv; torch 2.12.0

**Công việc đã làm:**
- Thêm StageTimer (SwiftEdit/timing.py) đo timing từng bước edit_image/gen_img, ghi JSONL; tạo subset 20 mẫu từ HF PIE_Bench_pp; viết summarize_timing.py; fix shape mask trong piebench_metrics.py

**Kết quả:**
- 20 mẫu/Apple M4 MPS: trung bình 69.0s/ảnh (steady 73.6s); UNet x2 ~43%, IP embeds ~24%, VAE decode ~23%; CLIP-Whole 23.02, CLIP-Edited 21.46, PSNR nền 14.01 (9/20). Lưu experimental_data/piebench_subset20_2026-06-14/

**Task README đã đánh [x]:**
- Tìm hiểu PieBench benchmark
- Lưu `results/piebench/metrics.csv`

**Vấn đề / cách xử lý:**
return_noise_image bỏ đi không cải thiện tốc độ (chỉ ~0.0001s); bottleneck thật là 2x UNet + IP embeds + VAE decode

**Bước tiếp theo:**
- Chạy PIE-Bench đầy đủ/nhiều mẫu hơn; so sánh baseline

---

### 2026-06-05 — [4d] Hướng ứng dụng: object removal / inpainting

**Môi trường:** Mac M4; tài liệu đề tài

**Công việc đã làm:**
- Thêm hướng dùng SwiftEdit để xóa vật thể khỏi ảnh.
- Tách rõ đây là local edit/inpainting: mask vẫn có ý nghĩa, khác với global style/weather edit.
- Đề xuất cấu hình SwiftEdit-SG, SwiftEdit-UserMask/GTMask, SwiftEdit-SG+Dilate và LaMa baseline nếu còn thời gian.
- Cập nhật `SwiftEdit_DeTai_CS2309.md`, `README.md`, `SwiftEdit_Overview.md`, `QA.md`.

**Kết quả:**
- Độ khả thi đánh giá là **trung bình-cao** với object nhỏ/vừa và nền đơn giản.
- Rủi ro chính: object lớn che nhiều background, viền/ghosting, texture nền bị méo; LaMa có thể mạnh hơn vì là inpainting model chuyên dụng.
- Metric đề xuất: detector confidence drop, CLIP margin `"without object"` vs `"with object"`, PSNR/SSIM/LPIPS ngoài mask, realism/human rating, IoU/Dice mask nếu có GT.

**Bước tiếp theo:**
- Chọn 20–40 ảnh có object cần xóa.
- Chạy SwiftEdit-SG trước, visualize mask; sau đó thử user/GT mask và mask dilation nếu cần.

---

### 2026-06-05 — [4c] Hướng ứng dụng: global style/weather edit

**Môi trường:** Mac M4; tài liệu đề tài

**Công việc đã làm:**
- Đánh giá hướng dùng SwiftEdit cho global scene/style editing: ngày↔đêm, mùa xuân/hạ/thu/đông, mưa↔nắng, overcast/golden hour.
- Ghi rõ mask IoU/Dice và PSNR/MSE background không phù hợp vì global edit tác động lên toàn ảnh.
- Đề xuất metric mới theo 3 trục: đúng target/style, giữ content/layout, realism/artifact.
- Cập nhật `SwiftEdit_DeTai_CS2309.md`, `README.md`, `SwiftEdit_Overview.md`, `QA.md`.

**Kết quả:**
- Độ khả thi đánh giá là **trung bình**: hợp để khảo sát ứng dụng và failure cases, đặc biệt với edit nhẹ như day→night, sunny→overcast, warm/cold tone.
- Rủi ro cao hơn ở mùa đông/tuyết, mưa lớn, đêm có nguồn sáng/phản chiếu vì cần thay đổi ánh sáng, texture và chi tiết toàn cảnh.

**Bước tiếp theo:**
- Chọn 20–40 ảnh outdoor/street/landscape.
- Chạy prompt nhẹ/mạnh; nếu có full-image mask thì so SwiftEdit-SG vs SwiftEdit-FullMask.
- Tạo bảng metric: CLIP target, zero-shot CLIP label, DINO/CLIP image similarity, LPIPS/SSIM phụ, human rating và runtime.

---

### 2026-06-05 — [4d] Chốt hướng đầu: bỏ decode `noise_image`

**Môi trường:** Mac M4; code + tài liệu

**Công việc đã làm:**
- Chọn hướng SwiftEdit-RT đầu tiên vì đơn giản nhất và không làm giảm chất lượng: không tạo `noise_image` trong `IPSBV2Model.gen_img()` khi caller không dùng.
- Patch `SwiftEdit/models.py`: thêm cờ `return_noise_image=False`; chỉ decode `noise_image` khi bật cờ này.
- Patch `SwiftEdit/infer.py`: gọi `gen_img(..., return_noise_image=False)` rõ ràng.
- Cập nhật `README.md`, `QA.md`, `SwiftEdit_DeTai_CS2309.md` với thứ tự ưu tiên các hướng tăng tốc không train.

**Kết quả:**
- Output ảnh edited chính giữ nguyên công thức decode; chỉ bỏ một VAE decode phụ vốn bị discard bởi `res_gen_img, _`.
- Thứ tự ưu tiên còn lại: GPU mask threshold → profiler → cache latent/embedding → channels_last/fp16 → torch.compile → TinyVAE/TAESD → TensorRT/Core ML/quantization.

**Bước tiếp theo:**
- Chạy đo baseline vs `SwiftEdit-no-noise-decode` trên Mac MPS và Colab T4.
- Sau đó patch GPU mask threshold để bỏ `.cpu().apply_()`.

---

### 2026-06-05 — [4d] Đổi hướng đào sâu: SwiftEdit-RT

**Môi trường:** Mac M4; tài liệu đề tài

**Công việc đã làm:**
- Rà lại luận điểm cốt lõi của SwiftEdit: realtime/instant editing.
- Đánh giá lại hướng SAM 3: tuy dễ làm và có định lượng mask, nhưng thêm segmentation model sẽ tăng latency và tài nguyên end-to-end.
- Chọn hướng mới: **SwiftEdit-RT: Realtime-Oriented Inference Acceleration**.
- Cập nhật `SwiftEdit_DeTai_CS2309.md`, `README.md`, `SwiftEdit_Overview.md`, `QA.md` theo hướng profiling + tối ưu inference.

**Kết quả:**
- Hướng chính mới gồm: latency breakdown theo module, vectorized self-guided mask trên GPU, bỏ decode `noise_image` không dùng, cache latent/embedding cho cùng ảnh nhiều prompt, thử `fp16`, `channels_last`, `torch.compile`, TinyVAE/TAESD.
- SAM 3 được hạ xuống optional/offline mask quality analysis, không tính là pipeline realtime.

**Bước tiếp theo:**
- Viết profiler cho `edit_image()` để đo từng module.
- Patch nhanh 2 tối ưu ít rủi ro: GPU mask threshold và `return_noise_image=False`.
- Chạy so sánh baseline vs optimized trên Mac MPS và Colab T4.

---

### 2026-06-05 — [4d] Khảo sát ban đầu: SwiftEdit + SAM 3

**Môi trường:** Mac M4; tài liệu đề tài

**Công việc đã làm:**
- Khảo sát các hướng thay module pipeline: SAM 3 mask, GroundingDINO+SAM2, auto source prompt bằng Florence-2/Qwen2.5-VL, grounded evaluation, baseline FLUX.1 Kontext/Qwen-Image-Edit/Step1X-Edit.
- Ban đầu chọn hướng **SwiftEdit + SAM 3: Concept-guided Mask Replacement** để phân tích mask.
- Cập nhật `SwiftEdit_DeTai_CS2309.md`, `README.md`, `SwiftEdit_Overview.md`, `QA.md` với pipeline mới, research questions RQ8/RQ9, đóng góp C13, kế hoạch thí nghiệm và tài liệu tham khảo.

**Kết quả:**
- Hướng SAM 3 được định nghĩa rõ: trích concept → SAM 3 sinh mask → SwiftEdit ARaM dùng external mask → so sánh self-guided / GT / SAM 3.
- Sau phản biện realtime, hướng này được chuyển thành optional/offline analysis, không còn là hướng chính.
- Metric: IoU/Dice mask, PSNR/MSE background, CLIP-Whole/Edited, runtime; có failure cases cho add object/style/global edit.

**Bước tiếp theo:**
- Chỉ quay lại hướng này nếu còn thời gian và cần phân tích chất lượng mask.
- Không đưa SAM 3 vào pipeline realtime chính của đề tài.

---

### 2026-06-05 — [3c] Chốt bộ độ đo đánh giá SwiftEdit/PieBench

**Môi trường:** Mac M4 (MPS); .venv

**Công việc đã làm:**
- Ghi rõ bộ metric chính: PSNR/MSE trên vùng không edit `(1 - mask)`, CLIP-Whole và CLIP-Edited theo `edit_prompt`, runtime theo thời gian gọi `edit_image()`.
- Cập nhật `QA.md`, `SwiftEdit_Overview.md`, `SwiftEdit_DeTai_CS2309.md`, `README.md` để thống nhất nguồn công bố/công nhận: PIE-Bench/PnP Inversion ICLR 2024; SwiftEdit CVPR 2025 Table 1; CLIP/CLIPScore.
- Sửa `scripts/piebench_metrics.py`: `clip_whole` dùng `edit_prompt` thay vì `src_prompt`, khớp metric `clip_similarity_target_image` của PIE-Bench.

**Kết quả:**
- Định nghĩa metric trong tài liệu khớp pipeline hiện tại: background preservation = PSNR/MSE vùng nền; edit fidelity = CLIP toàn ảnh/vùng edit với target prompt.
- `source .venv/bin/activate && python -m compileall -q scripts/piebench_metrics.py` chạy OK.

**Bước tiếp theo:**
- Chạy lại smoke eval hoặc subset PIE-Bench sau khi có dữ liệu đầy đủ để cập nhật lại `metrics.csv` theo định nghĩa `CLIP-Whole` mới.

---

### 2026-06-05 — [3c] Xây pipeline đánh giá PieBench (metrics + eval)

**Môi trường:** Mac M4 (MPS); .venv; torchmetrics 1.9.0

**Công việc đã làm:**
- Viết scripts/piebench_utils.py (đọc mapping_file.json, RLE mask decode, chọn subset), piebench_metrics.py (PSNR/MSE vùng background, CLIP-Whole/Edited qua torchmetrics CLIP ViT-L/14), run_piebench_eval.py (loop edit_image + ghi metrics.csv, có resume). Thêm download_piebench.sh + create_piebench_smoke.py (2 ảnh demo test pipeline khi chưa tải form). Thêm cell 3b/3d vào notebook phase3. Ghi QA.md mục 5 về bộ độ đo.

**Kết quả:**
- Smoke eval 1 mẫu chạy OK trên Mac MPS (~50s/ảnh, CLIP-Whole=18.76, CLIP-Edited=22.64); metrics.csv sinh đúng. PieBench đầy đủ tải qua form PnP Inversion (không có trên git). Độ đo theo chuẩn ICLR 2024 (PIE-Bench) + SwiftEdit CVPR 2025 Table 1.

**Task README đã đánh [x]:**
- *(không đánh task README)*

**Vấn đề / cách xử lý:**
Thiếu torchmetrics trong venv cũ → cài + script báo lỗi sớm trước khi load model. Bug permute tensor→PIL và mask broadcast 512×512 → fix bằng resize + Image.open.

**Bước tiếp theo:**
- Tải PieBench đầy đủ qua form; chạy 50-100 mẫu trên Colab T4; so sánh Table 1 paper

---

### 2026-06-05 — [2a. Mac] Runtime cùng preset dog — Mac MPS 30.1s vs T4 1.3s

**Môi trường:** Mac M4 (MPS); pyenv 3.12.10; .venv

**Công việc đã làm:**
- Chạy preset 02.jpg dog→dog with mouth opened trên Mac MPS (cùng prompt Colab T4)

**Kết quả:**
- `Edit 'dog' -> 'dog with mouth opened' in 30.1s` (inference-only, model đã load)
- Lưu: `results/notebook/nb_dog_dog_to_dog_with_mouth_opened.png`
- Cùng ảnh/prompt với Colab T4 (2026-06-04)

**Bảng runtime (preset `02.jpg`, 512×512, inference-only):**

| | Paper A100 | Colab T4 | Mac M4 MPS |
|---|---:|---:|---:|
| `edit_image` | ~0.23 s | **1.3 s** | **30.1 s** |
| Tốc độ tương đối | 1× (ref.) | ~5.7× chậm hơn A100 | ~131× chậm hơn A100; ~23× chậm hơn T4 |

*Ghi chú báo cáo:* Mac MPS dùng được cho demo/ablation nhỏ; benchmark lớn (PieBench) → Colab T4.

**Task README đã đánh [x]:**
- Bảng runtime 3 cột: **Mac MPS | Colab T4 | Paper A100**

**Bước tiếp theo:**
- PieBench 50 mẫu Colab; ablation hyperparameter Mac

---

### 2026-06-04 — [2b. Colab] Inference Colab T4 — dog preset 1.3s

**Môi trường:** Google Colab — Tesla T4 (Colab extension)

**Công việc đã làm:**
- Chạy notebook CS2309_SwiftEdit_test trên Colab T4 (extension): preset dog→dog with mouth opened (assets/imgs_demo/02.jpg)

**Kết quả:**
- `Edit 'dog' -> 'dog with mouth opened' in 1.3s` (chỉ `edit_image`, model đã load)
- Lưu: `/content/CS2309.CH201/results/notebook/nb_dog_dog_to_dog_with_mouth_opened.png`
- Ảnh: `assets/imgs_demo/02.jpg` — preset `dog` / `dog with mouth opened`

**So sánh runtime (cùng SwiftEdit 512×512, khác preset/ảnh):**

| Môi trường | Thời gian | Ghi chú |
|---|---|---|
| Paper A100 | ~0.23 s | Báo cáo gốc |
| **Colab T4** | **1.3 s** | Phiên này — inference-only |
| Mac M4 MPS | ~91 s | Demo woman→Taylor Swift (có thể gồm overhead MPS lần đầu) |

**Task README đã đánh [x]:**
- Ghi GPU name (T4/A100) và runtime Colab
- Chạy demo CUDA — kết quả khớp Mac
- Cài `requirements.txt` (CUDA)

**Vấn đề / cách xử lý:**
Không dùng Drive — weights/HF trên /content

**Bước tiếp theo:**
- Ghi bảng runtime 3 cột Mac|Colab T4|Paper; PieBench batch Colab

---

### 2026-06-04 — [2b. Colab] Notebook Colab extension: GPU, upload, stack HF mới

**Môi trường:** Colab extension (Cursor/VS Code) + Colab web (T4)

**Công việc đã làm:**
- **`notebooks/CS2309_SwiftEdit_test.ipynb`**
  - Metadata `accelerator: GPU`, `colab.gpuType: T4` (Colab web tự chọn T4)
  - Cell 1: `_check_colab_gpu()` (`nvidia-smi`) trước clone/tải weights; hướng dẫn extension **New Colab Server → GPU → T4**
  - Cell Setup: `pip install -U -r requirements.txt` — **không** pin `numpy==1.26.4` trên Colab; kiểm tra `transformers≥4.46`, `diffusers≥0.32`, import `EncoderDecoderCache`
  - Upload ảnh: ô **Đường dẫn** + **Upload to Colab** (Explorer); nút demo `02.jpg`; **ẩn** `files.upload()` / FileUpload trên extension (widget rỗng `value={}`)
- **`SwiftEdit/requirements.txt`**: stack mới `transformers≥4.46`, `diffusers≥0.32`, `peft`, `accelerate` — bỏ pin torch `cu118` / `2.2.1`
- **`requirements-mac.txt`**: đồng bộ stack mới; `numpy≥1.26`
- **`SwiftEdit/models.py`**: `torch.load(..., weights_only=True)` cho `ip_adapter.bin`

**Kết quả:**
- Patch để Colab extension **có thể chạy** sau quy trình: Restart kernel → cell 1 → Setup → Load models → Áp dụng ảnh (path) → inference
- Chưa ghi nhận inference Colab T4 thành công end-to-end trong phiên này — cần chạy lại sau khi push/pull notebook mới

**Task README đã đánh [x]:**
- *(chưa đánh [x] — chờ xác nhận chạy inference Colab OK)*

**Vấn đề / cách xử lý:**

| Lỗi | Nguyên nhân | Cách xử lý |
|-----|-------------|------------|
| `Colab chưa có GPU` | Server CPU / Auto Connect | Extension: **New Colab Server → T4**; web: Runtime → T4 |
| `numpy.dtype size changed` | Hạ `numpy==1.26.4` trên Colab có torchvision build NumPy 2 | Bỏ pin numpy trên Colab |
| `EncoderDecoderCache` ImportError | Colab `diffusers` mới + `transformers==4.37` cũ | Nâng stack (`transformers≥4.46`, `diffusers≥0.32`), không hạ về 0.22 |
| Upload widget rỗng / `files.upload` disabled | Colab extension không có browser session Colab | **Upload to Colab** → dán `/content/...` hoặc nút **Dùng ảnh demo** |

**Bước tiếp theo:**
- Restart kernel → Setup → Load → chạy preset hoặc upload path → ghi runtime inference T4 (s) vào nhật ký
- Đánh `[x]` README mục Colab khi có bằng chứng (ảnh output + `torch`/`GPU` name)

---

### 2026-06-04 — [2a. Mac] Notebook test, tài liệu Colab/T4, ghi chú runtime

**Môi trường:** Mac M4 (MPS); Jupyter .venv

**Công việc đã làm:**
- Notebook notebooks/CS2309_SwiftEdit_test.ipynb (preset, upload ipywidgets 8, inference)
- README: hướng dẫn cài đặt/pyenv/scripts; .gitignore loại weights
- Ghi nhận: Mac MPS ~91s vs paper A100 ~0.23s; Colab T4 VRAM 15GB đủ inference; Drive ~15GB chật (weights 9.6GB+HF)

**Kết quả:**
- Notebook chạy OK trên Mac (.venv); upload widget sửa tuple ipywidgets 8
- Chưa chạy Colab thực tế — đã ghi kế hoạch T4 + lưu weights Drive trong đề tài/README

**Task README đã đánh [x]:**
- Tạo notebook `CS2309_SwiftEdit_test.ipynb`

**Vấn đề / cách xử lý:**
matplotlib thiếu kernel → %pip trong notebook; FileUpload value tuple (ipywidgets 8)

**Bước tiếp theo:**
- Chạy thử Colab T4: IN_COLAB=True, weights trên Drive; đo runtime inference-only; PieBench batch

---

### 2026-06-04 — [2a. Mac] Setup Mac (pyenv/venv) và chạy demo SwiftEdit

**Môi trường:** Mac M4 (MPS); pyenv 3.12.10; .venv

**Công việc đã làm:**
- Clone SwiftEdit; pyenv 3.12.10 + .venv + requirements-mac.txt (PyTorch MPS)
- Tải swiftedit_weights (~9.6GB); patch infer.py (get_device mps); patch models.py (SD2.1 mirror Manojb, map_location CPU)
- Tải HF: sd-turbo, Manojb/stable-diffusion-2-1-base, h94/IP-Adapter image_encoder
- Bổ sung tài liệu: §1.3 pipeline cũ, ảnh assets/pipeline/, QA.md

**Kết quả:**
- Demo woman→Taylor Swift OK; output SwiftEdit/result_woman->Taylor Swift.png; ~91s/ảnh trên MPS
- Scripts: scripts/run_swiftedit.sh, download_swiftedit_weights.sh

**Task README đã đánh [x]:**
- Cài PyTorch Mac (MPS) — **không** dùng bản CUDA
- Cài dependencies còn lại
- Clone repo SwiftEdit
- Tải checkpoint → `swiftedit_weights/`
- Sửa `device = "mps"` trong code
- Chạy demo `assets/imgs_demo` thành công
- Ghi thời gian/ảnh và RAM peak (Activity Monitor)

**Vấn đề / cách xử lý:**
stabilityai/stable-diffusion-2-1-base 401 → mirror Manojb
IP-Adapter image_encoder timeout → tải riêng model.safetensors
ip_adapter.bin load CUDA → map_location=cpu

**Bước tiếp theo:**
- Ghi RAM peak (Activity Monitor); setup Colab notebook
- Ablation hyperparameter; PieBench eval trên Colab

---

### 2026-06-01 — [0. Khởi tạo project] Khởi tạo cấu trúc đề tài

**Môi trường:** Mac M4

**Công việc đã làm:**
- Tạo README, đề tài chi tiết, skill Cursor hỗ trợ nhật ký

**Kết quả:**
- Repo CS2309.CH201 sẵn sàng; skill sync README + NHAT_KY + §8.1

**Task README đã đánh [x]:**
- *(không đánh task README)*

**Bước tiếp theo:**
Clone SwiftEdit; cài env Mac MPS

---
