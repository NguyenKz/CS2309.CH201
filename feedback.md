# Feedback thầy — ghi chú trả lời & minh chứng

> Lưu góp ý của thầy và câu trả lời đã đối chiếu paper / code / slide.  
> Thêm mục mới phía trên (mới nhất trước). Không dùng LaTeX.

| | |
|---|---|
| **Paper** | SwiftEdit — arXiv [2412.04301](https://arxiv.org/abs/2412.04301) · CVPR 2025 |
| **Slide liên quan** | [`report/SLIDE_SwiftEdit.md`](report/SLIDE_SwiftEdit.md) §3a-fig · §6b fidelity |
| **Báo cáo** | [`report/BAO_CAO_SwiftEdit.md`](report/BAO_CAO_SwiftEdit.md) hộp training |
| **QA** | [`QA.md`](QA.md) mục training (Fig. 2) |

---

## F2. PSNR ~48.5 “quá đẹp” — đa độ đo, FP16/Tensor Core, bằng chứng ảnh

**Ngày ghi:** 2026-07-31

### Góp ý thầy (tóm tắt)

- Số PSNR ~48.5 dB bất thường / “quá tốt” — có phải so nhầm / trick?
- Cần thêm độ đo khác + giải thích FP16 vs FP32 (hardware).
- Cùng prompt, cùng ảnh input: kết quả ra sao? Show ảnh PSNR thấp / cao.
- Có trang online khách quan để thầy tự so? Thư viện nào trong project?

### Kết luận một dòng

**48.5 dB là fidelity precision (FP16 edit ↔ FP32 edit cùng job), không phải “ảnh edit giống ảnh gốc”.** Edit vẫn đổi nội dung mạnh so với nguồn (~19 dB). SSIM/LPIPS/MSE cùng protocol xác nhận hai output gần trùng pixel.

Chi tiết audit: [`report/AUDIT_PSNR_FIDELITY.md`](report/AUDIT_PSNR_FIDELITY.md).

### 1. Protocol so cái gì?

| | |
|---|---|
| **Bench** | `experimental_data/quality_speed_bench_2026-06-17/` |
| **Producer** | `notebooks/CS2309_SwiftEdit_quality_speed_bench.ipynb` |
| **Reference** | `baseline_fp32` — ảnh **đã edit** FP32 cùng `(job_id, prompt_idx)` |
| **Test** | `improved_fp16_cache` (và các config FP16 cùng precision) |

Trong notebook: `REFERENCE = "baseline_fp32"` rồi `qm.compare(pc[REFERENCE], pc[c])` — **không** so với JPEG trong `sample_source/` / `source_images/`.

Công thức (`data_range=1`): `PSNR ≈ 10 · log10(1 / MSE)`. MSE mean ~0.000020 → PSNR ~47–48 dB là **kỳ vọng** khi hai ảnh gần trùng pixel.

### 2. Đa độ đo (đã đo) — không chỉ PSNR

Nguồn: `quality_raw.csv`, config `improved_fp16_cache`, **n=600**. Seed: `250101049`.

**Ý nghĩa từng độ đo (lớp A — so output ↔ FP32 cùng job, không vs ảnh gốc):**

- **MSE ↓** — Sai số bình phương trung bình từng pixel. Càng nhỏ càng giống FP32.
- **PSNR ↑ (dB)** — Đổi MSE ra thang log. Cao (~48) = gần trùng; thấp (~22) = lệch rõ.
- **SSIM ↑ (0–1)** — Giống cấu trúc / tương phản cục bộ. Gần 1 = cấu trúc giữ tốt.
- **LPIPS ↓** — “Nhìn có giống không” (mạng deep). Gần 0 = gần như không lệch mắt.

Miss / Hit / Overall / VRAM / Disk = tốc độ · bộ nhớ (xem slide §6a).

| Độ đo | Mean (min–max) | Đọc |
|---|---|---|
| **PSNR** (dB) | **48.56 (34.97–56.73)** | Cao vì lệch pixel rất nhỏ |
| **SSIM** | **0.9976 (0.9889–0.9994)** | Cấu trúc gần như giữ |
| **LPIPS** ↓ | **0.0008 (0.0001–0.0063)** | Gần như không lệch mắt |
| **MSE** | **0.000020 (0.000002–0.00032)** | RMSE ~1/255 |

Hai lớp độ đo **không trộn**:

| Lớp | Câu hỏi | Độ đo | So cái gì | Số điển hình |
|---|---|---|---|---|
| **A. Fidelity tối ưu** | FP16 lệch FP32 không? | PSNR/SSIM/LPIPS/MSE | edit_FP16 ↔ edit_FP32 | PSNR ~48.5 |
| Spot-check vs nguồn | Edit có đổi ảnh không? | PSNR | edit ↔ **source** | ~19 dB |
| **B. Edit quality** (PieBench) | Đúng prompt / giữ nền? | CLIP-W/E; PSNR nền | edited ↔ prompt; edited ↔ source `(1−mask)` | CLIP-W ~23 · CLIP-E ~21.5 · bg PSNR ~14 |

PieBench: [`experimental_data/piebench_subset20_2026-06-14/EDIT_QUALITY_SUMMARY.md`](experimental_data/piebench_subset20_2026-06-14/EDIT_QUALITY_SUMMARY.md).

### 3. Vì sao FP16 gần FP32? (Tensor Core — không phải TensorFlow)

Stack đề tài: **PyTorch + CUDA (Tesla T4)**, không dùng TensorFlow.

1. **FP32** — precision mặc định đầy đủ trên GPU; chậm hơn, tốn VRAM hơn.
2. **FP16** — GPU NVIDIA hỗ trợ **native** qua **Tensor Core** (matmul/conv half precision): cùng kiến trúc model, cùng weight (cast), cùng ảnh + cùng prompt → đường suy luận gần FP32, chỉ sai số làm tròn nhỏ → output gần trùng → PSNR cao, LPIPS rất thấp.
3. Bench còn `channels_last` + EditCache → tăng tốc / giảm VRAM; **fidelity** vẫn đo bằng so với FP32 cùng job.
4. **Cùng input / prompt / seed:** cặp file cùng `job_id` + `prompt_idx`; seed bench **`250101049`** (`eval_seed_base` · `data/jobs_june17.json` / slide §6). Cùng weights + pipeline one-step + cùng ảnh/prompt/seed → so sánh FP16↔FP32 công bằng.

**Nguồn (tách hai lớp — không trộn):**

| Phần câu | Nguồn | Ghi chú |
|---|---|---|
| Tensor Core nhận **FP16** input cho matmul (MMA); tích lũy thường FP32 | NVIDIA: [Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/); [cuDNN — Tensor Core / Notes on precision](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/core-concepts.html) | Fact phần cứng / thư viện |
| Turing (gồm **T4**) có Tensor Core FP16 | NVIDIA [Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html) | T4 = sm_75 Turing |
| Kết quả Tensor Core **gần nhưng không bit-identical** FP32 thuần | Cùng blog/cuDNN: mixed-precision ≠ bit-equivalent | Giải thích vì sao PSNR cao nhưng hữu hạn (~35–57), không ∞ |
| Cùng model + cast weight + cùng ảnh/prompt/seed → output gần trùng → **PSNR ~48 / LPIPS ~0.001** | **Bench đề tài** `quality_speed_bench_2026-06-17` + `report/AUDIT_PSNR_FIDELITY.md` | **Không** phải paper SwiftEdit; là đo thực nghiệm FP16↔FP32 |

Nhìn mắt: hai ảnh FP16/FP32 **trông gần như một**; diff chỉ lộ khi zoom hoặc tool metric — khớp SSIM ~0.99 và LPIPS ≪ 0.01.

### 4. Ảnh minh họa: PSNR thấp / cao / trung bình

Thư mục demo (đã copy sẵn): [`experimental_data/quality_speed_bench_2026-06-17/psnr_extreme_examples/`](experimental_data/quality_speed_bench_2026-06-17/psnr_extreme_examples/).

| Vai trò | job | PSNR | SSIM | LPIPS | MSE | Cặp file |
|---|---|---|---|---|---|---|
| **Thấp** | `111000000004_0` | 34.97 | 0.9889 | 0.0036 | 0.000319 | `low_psnr/fp32_…` vs `fp16_…` |
| **Trung bình** | `000000000000_0` | 46.48 | 0.9979 | 0.0007 | 0.000022 | `mid_sample/fp32_…` vs `fp16_…` |
| **Cao** | `111000000006_0` | 56.73 | 0.9990 | 0.0001 | 0.000002 | `high_psnr/fp32_…` vs `fp16_…` |

**Nhấn mạnh:** ngay cả PSNR thấp nhất (~35) vẫn SSIM cao / LPIPS thấp → lệch **precision**, không phải hai ảnh khác nội dung.

### 5. Thư viện trong project + trang so sánh online

**Thư viện đo trong repo (lớp A):** [`torchmetrics`](https://torchmetrics.readthedocs.io/)

- `PeakSignalNoiseRatio`
- `StructuralSimilarityIndexMeasure`
- `LearnedPerceptualImagePatchSimilarity`
- MSE tự tính trên tensor `[0,1]`

PieBench (lớp B): [`scripts/piebench_metrics.py`](scripts/piebench_metrics.py) — CLIP + PSNR/MSE vùng nền; **khác vai trò**, không giải thích số 48.5.

**Trang online để thầy tự upload cặp FP16/FP32** (không tự dựng trang):

- **Chính:** [Img2Go — Compare Images](https://www.img2go.com/compare-image) — PSNR, SSIM, MAE/RMSE, diff visual trên browser.
- Phụ: [DualView Quality Metrics](https://www.dualview.ai/quality-metrics/) — SSIM/PSNR + xem cạnh nhau.

Số online có thể lệch nhẹ so với `torchmetrics` (scale 0–1 vs 0–255, cửa sổ SSIM) — hướng vẫn “rất giống”. Số báo cáo chính thức lấy từ CSV / `torchmetrics`.

### Việc đã làm trong repo

- Audit + bảng mean (min–max) trên slide §6b.
- Excel từng ảnh: `report/ket_qua_chat_luong_tung_anh.xlsx`.
- Cặp ảnh extreme: `experimental_data/quality_speed_bench_2026-06-17/psnr_extreme_examples/`.

---

## F1. Sơ đồ training paper (Fig. 2) — lửa / tuyết, Stage 1–2, ai được train

**Ngày ghi:** 2026-07-31

### Góp ý thầy (tóm tắt)

- Paper đã có hình minh họa đẹp, đủ ý — vì sao không dùng figure gốc mà tự vẽ lại (và chưa chắc vẽ đúng)?
- Fig. 2 chia 2 stage: **song song hay tuần tự**? Cái nào trước / sau?
- Ký hiệu **ngọn lửa** và **bông tuyết** nghĩa là gì?
- Module nào được **train** trong pipeline, cái nào không, vì sao?

### Figure gốc trong repo SwiftEdit (dùng khi trả lời thầy)

Hình minh họa two-stage training từ README official ([`SwiftEdit/README.md`](SwiftEdit/README.md) — `swiftedit_diagram.png`):

![SwiftEdit two-stage training diagram (official)](SwiftEdit/assets/swiftedit_diagram.png)

Nguồn: `SwiftEdit/assets/swiftedit_diagram.png` (cùng file README upstream hiển thị dưới mục *Two-stage training strategy*). Khi thuyết trình: chiếu ảnh này + bảng legend bên dưới — không thay bằng Mermaid tự vẽ.

### Câu trả lời đã chốt

#### 1. Hai stage: tuần tự, không song song

- **Stage 1 rồi Stage 2.** Caption Fig. 2 + Sec. 4.1: Stage 1 *warm up* `F_theta` trên synthetic từ SBv2 → Stage 2 *shift / continue* trên ảnh thật CommonCanvas (khởi tạo từ trọng số Stage 1).
- Màu cam / teal trên Fig. 2 = **hai pha khác thời điểm**, không phải hai nhánh chạy đồng thời trên GPU.

```
Init F_theta từ SBv2 weights
  → Stage 1 synthetic warm-up (~100k iter, JourneyDB)
  → Stage 2 real images continue (~180k iter, CommonCanvas)
  → Inference edit (không train)
```

##### Ký hiệu `ε ~ N(0, I)` trên Stage 1 (Fig. 2)

Trên sơ đồ thường viết `ε ~ N(0, I)` (chữ e/ε trong font paper). Đây là **nhiễu Gaussian chuẩn** đưa vào SBv2 để sinh ảnh synthetic:

| Ký hiệu | Đọc | Nghĩa |
|---|---|---|
| **ε** (epsilon; trên hình đôi khi giống chữ **e**) | “eps” | Tensor **noise** trong latent space (cùng shape mà SBv2 nhận, vd. 4×64×64) |
| **N** | “Normal” | Phân phối **Gaussian** (chuẩn) |
| **0** | “zero” | **Vector kỳ vọng (mean)** = toàn số 0 — trung bình nhiễu bằng 0 |
| **I** | “identity” | **Ma trận đơn vị** — hiệp phương sai = I ⇒ mỗi chiều nhiễu độc lập, phương sai = 1 |

Đọc cả cụm: **ε được lấy mẫu từ phân phối chuẩn nhiều chiều có mean 0 và covariance I** (isotropic Gaussian / white noise).

**Làm sao biết `N` = Gaussian?** Đây là **ký hiệu toán xác suất chuẩn**, không phải chữ viết tắt riêng của SwiftEdit:

- Trong sách/paper ML–thống kê, biến ngẫu nhiên Gaussian thường viết **`X ~ N(μ, σ²)`** (1D) hoặc **`X ~ N(μ, Σ)`** (nhiều chiều): đọc “*X follows a Normal distribution with mean μ and variance/covariance …*”.
- **`N`** = **Normal** (= **Gaussian**; hai tên cùng một họ phân phối). Tiếng Việt hay gọi “phân phối chuẩn”.
- Cặp tham số sau `N` luôn là **(mean, covariance)**: paper viết `N(0, I)` → mean = **0**, covariance = **I** — đúng khuôn mẫu Normal, không phải ký hiệu khác (vd. Uniform thường là `U(a,b)`).
- Trong **diffusion** (DDPM, SD, SBv2…), forward process / noise khởi tạo mặc định là **`ε ~ N(0, I)`** — cùng công thức lặp lại khắp literature; SwiftEdit Fig. 2 chỉ dùng lại convention đó.

Không suy từ icon lửa/tuyết; suy từ **cách viết `~ N(·, ·)` trong xác suất**.

##### Mean và covariance là gì? (trong `N(0, I)`)

Hai tham số **mô tả hình dạng** của phân phối nhiễu — “nhiễu trung bình bằng bao nhiêu” và “các chiều nhiễu lan rộng / liên quan nhau thế nào”.

| Thuật ngữ | Tiếng Việt (thường dùng) | Ý nghĩa trực giác | Trong `N(0, I)` |
|---|---|---|---|
| **Mean** (μ) | Trung bình / kỳ vọng | “Trung tâm” của phân phối — giá trị trung bình khi lấy mẫu nhiều lần | **0** = vector toàn 0 → nhiễu **không lệch** về phía dương hay âm |
| **Covariance** (Σ) | Hiệp phương sai | Ma trận mô tả **độ lan** từng chiều và **tương quan** giữa các chiều | **I** (ma trận đơn vị) = mỗi chiều phương sai 1, **không** tương quan với chiều khác |

Ví dụ 1D đơn giản: `N(0, 1)` = chuông Gaussian tâm tại 0, độ rộng “chuẩn”.  
Nhiều chiều (latent diffusion): `ε` là vector dài (vd. hàng nghìn số); `N(0, I)` = mỗi phần tử ~ độc lập `N(0,1)`.

- **Mean = 0:** không đẩy ảnh theo một “bias” cố định khi thêm nhiễu.  
- **Covariance = I:** nhiễu “trắng” — không có pattern cấu trúc sẵn trong noise; model phải học cấu trúc từ data/prompt.

**Số 1 lấy từ đâu nếu chỉ thấy chữ `I`?**  
`I` = **ma trận đơn vị** (identity matrix): đường chéo toàn **1**, ngoài đường chéo toàn **0**. Ví dụ 3 chiều:

```
I = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
```

**Ma trận đơn vị là gì?**  
Ma trận vuông đóng vai trò như số **1** trong phép nhân số: nhân với mọi vector/ma trận tương thích thì **không đổi** kết quả.

- Ký hiệu thường: **`I`** hoặc **`I_n`** (cỡ n×n).
- Định nghĩa: `I_ij = 1` nếu `i = j`, còn lại `I_ij = 0`.
- Tính chất chính: với mọi vector `x` cùng số chiều, **`I · x = x`** (nhân với I như nhân với 1).
- Ví dụ 2×2: `I = [[1,0],[0,1]]`. Không phải mọi ma trận có số 1 đều là I — chỉ khi **đúng** 1 trên đường chéo và 0 mọi chỗ khác.

Trong `N(0, I)`, dùng `I` làm covariance vì muốn mỗi chiều nhiễu “chuẩn hóa” giống nhau và độc lập — chọn sẵn ma trận đơn giản nhất thỏa điều đó.

**Có trường hợp đường chéo không phải 1 không?**  
Có — nhưng lúc đó **không còn gọi là ma trận đơn vị `I`**.

| Trường hợp | Ví dụ | Ý nghĩa |
|---|---|---|
| Đúng `I` | `N(0, I)` | Đường chéo **luôn** 1 (định nghĩa). Không có “`I` ngược” (0 trên đường chéo, 1 ngoài) mà vẫn gọi là `I`. |
| Covariance co giãn đều | `N(0, σ² I)` | Đường chéo = **σ²** (có thể ≠ 1); vẫn độc lập giữa các chiều. |
| Covariance tổng quát | `N(0, Σ)` với `Σ = diag(2, 0.5, 1, …)` | Mỗi chiều **phương sai khác nhau**; đường chéo tùy chọn (phải > 0). |
| Có tương quan | `Σ` có số ≠ 0 ngoài đường chéo | Các chiều nhiễu “dính” nhau — ít dùng làm noise khởi tạo diffusion chuẩn. |

Trong SwiftEdit / SBv2 / DDPM mặc định vẫn **`ε ~ N(0, I)`** (hoặc tương đương sau khi scale lịch trình nhiễu). Muốn nhiễu mạnh/yếu hơn người ta thường nhân hệ số (`σ · ε`) chứ không đổi định nghĩa `I`.

**Còn ma trận “lộn ngược đường chéo” thì sao?** Ví dụ:

```
[[0, 0, 1],
 [0, 1, 0],
 [1, 0, 0]]
```

- **Có** — đây là ma trận toán học hợp lệ (thường gọi *exchange / reversal / anti-diagonal permutation*), dùng khi muốn **đảo thứ tự** tọa độ (`x ↦` phần tử đảo ngược).
- **Không** phải ma trận đơn vị `I` (I chỉ có 1 trên đường chéo chính `i=i`).
- **Không** dùng làm **covariance** trong `N(0, Σ)`: covariance phải **đối xứng xác định không âm** (mọi eigenvalue ≥ 0). Ma trận đảo trên có eigenvalue âm → **không** phải hiệp phương sai hợp lệ.

Tóm lại: Fig. 2 chỉ dùng `I` chuẩn; ma trận 1 nằm trên đường chéo phụ là chuyện khác, không thay cho `I` trong `ε ~ N(0, I)`.

Trong ma trận hiệp phương sai Σ:

- **Phần tử đường chéo** `Σ_ii` = **phương sai** của chiều thứ i → `I_ii = 1` ⇒ mỗi chiều có **variance = 1**.
- **Phần tử ngoài đường chéo** `Σ_ij` (i≠j) = hiệp phương sai giữa chiều i và j → `I_ij = 0` ⇒ **không tương quan** (độc lập trong Gaussian).

Vậy “phương sai = 1” **không** viết sẵn cạnh chữ `I` trên Fig. 2 — nó **nằm trong định nghĩa** ma trận đơn vị (đường chéo là 1). Viết `N(0, I)` gọn hơn `N(0, diag(1,1,…,1))`.

- Stage 1: random `ε` + prompt → SBv2 → ảnh synthetic; `ε` đó trở thành **nhãn** để `F_theta` học đảo (ảnh → đúng `ε`).
- Không nhầm **N** với “số lượng mẫu” hay **I** với “ảnh”; đây là ký hiệu xác suất chuẩn trong diffusion.

##### `ε` là “1 tensor latent” hay “noise ngẫu nhiên”?

**Cả hai — không đối lập.**

| Cách nói | Đúng? | Giải thích |
|---|---|---|
| `ε` là **một tensor** trong **latent space** | Đúng | Shape cố định theo model (vd. batch×4×64×64), không phải ảnh RGB pixel. |
| `ε` là **noise ngẫu nhiên** | Đúng | **Giá trị** trong tensor được **lấy mẫu mới** mỗi bước train từ `N(0, I)` — không phải một tensor cố định dùng mãi. |

Tóm lại từng bước Stage 1 (ý Fig. 2):

1. **Sample** `ε ~ N(0, I)` → được **1 tensor nhiễu ngẫu nhiên** (mỗi lần khác nhau).
2. Cùng `ε` + text prompt → SBv2 sinh ảnh synthetic.
3. VAE encode ảnh → `z`; `F_theta(z, prompt)` dự đoán `ε_hat`.
4. Loss kéo `ε_hat` về đúng **`ε` đã sample** ở bước 1 (regression).

“1 tensor” = **một mẫu nhiễu** (một lần draw), không phải “chỉ có đúng một noise duy nhất trong cả quá trình train”. Train lặp hàng nghìn bước → hàng nghìn `ε` khác nhau.

##### Vì sao phải ngẫu nhiên **theo phân phối chuẩn**, không phải “ngẫu nhiên kiểu bất kỳ”?

- SBv2 / diffusion **được train** với giả định nhiễu đầu vào `ε ~ N(0, I)`. Train `F_theta` phải dùng **cùng loại nhiễu** thì mới khớp generator.
- Không chọn Uniform/Laplace… làm mặc định vì: (1) toán DDPM dựng trên Gaussian; (2) cộng nhiều nhiễu nhỏ → gần Gaussian (CLT); (3) hệ sinh thái SD/SBv2 đã chuẩn hóa vậy — đổi phân phối thì phải train lại generator.
- Tóm lại: phải **đúng phân phối generator đã học**, không phải ngẫu nhiên tùy ý.

##### `G(·)`, prompt `c_y`, và vì sao Fig. 2 trông như chỉ `ε → G → x`

- **`G(·)`** = generator **SBv2** (one-step T2I), đóng băng lúc train SwiftEdit.
- **Prompt:** caption JourneyDB → CLIP text → embedding **`c_y`**. Paper **có** `c_y` trong công thức, dù mũi tên Fig. 2 thường **không vẽ** nhánh prompt (dễ hiểu nhầm thành `G(ε)` thuần).
- **Chỗ ghi trong paper:** Sec. 4.1 Stage 1, **Equation (5)**:

```
ε ~ N(0, 1),    z = G(ε, c_y)
```

- Eq. (5) output của `G` là **latent `z`**. Fig. 2 vẽ *Synthetic image x* = ảnh sau VAE decode (đường đủ tới pixel). Viết tắt `x̂ = D(G(ε, c_y))` đúng ý “ra ảnh”, nhưng paper Stage 1 nhấn cặp `(ε, z)`. `x̂ = D(ẑ)` nêu rõ hơn ở Stage 2 (loss DISTS).
- Kích thước ảnh mặc định khung SD 2.1 / SBv2: **512×512** (latent ~4×64×64).

##### Sau ảnh synthetic: nhánh VAE → `z` và nhánh IP (lửa)

- **VAE (tuyết):** Encoder nén ảnh `x` → latent **`z = E(x)`**. Latent = biểu diễn ẩn/nén, không phải RGB. Không train VAE.
- **`F_θ` (lửa):** Inversion Net — `ε̂ = F_θ(z, c_y)`. Cùng prompt `c_y` với `G`. Học đoán lại **`ε`** đã đưa vào SBv2 từ `(z, c_y)`.
- **Text Encoder (tuyết):** CLIP text; input caption `y`, output `c_y`.

##### IP-Adapter (lửa Stage 1) — dễ hiểu

- **Là gì:** “cổng phụ” trong generator: thêm nhánh attention **ảnh** cạnh nhánh **text** → `G` thành **`G_IP`**. Không phải bước pipeline tách ngoài UNet.
- **Cộng (Eq. 3):** `h = Attn_text + s_x · Attn_image` (cùng Query; cộng hai hướng dẫn).
- **Train cái gì:** projection + `W^K_x`/`W^V_x` (cách đưa ảnh gốc `c_x` vào máy vẽ). Học “đọc ảnh gốc hỗ trợ tái tạo” — không học đoán `ε`.
- **Vì sao vẫn cần `F_θ`?** IP = nhớ/bám ảnh; `ε̂` = **hạt giống input đúng** cho SBv2 để output ổn định, bám scene. Chỉ random `ε` + IP ≈ sinh biến thể, không phải đảo+edit kiểm soát được.
- **Thứ tự train:** Stage 1 train **`F_θ` + IP cùng lúc** (loss regr + recon). Stage 2 **chỉ `F_θ`**, **IP freeze**.

##### Vì sao Stage 2 + vì sao freeze IP

- Stage 1 chỉ synthetic → **domain gap** ảnh thật → cần Stage 2 (CommonCanvas, DISTS + L_reg).
- Freeze IP: giữ image prior Stage 1; chỉ tinh chỉnh đảo; tránh IP/`ε̂` overfit ảnh thật → khó edit.

#### 2. Lửa và tuyết là gì?

Quy ước **phổ biến trong paper CV/ML** (ControlNet, IP-Adapter, nhiều sơ đồ Diffusers) — không riêng SwiftEdit:

| Ký hiệu | Nghĩa |
|---|---|
| Ngọn lửa (fire) | **Trainable** — nhận gradient, cập nhật trọng số |
| Bông tuyết (snowflake) | **Frozen** — không train, chỉ forward |

Paper HTML ít khi viết chữ “fire/snowflake”; icon trên Fig. 2 đúng convention cộng đồng.

#### 3. Ai được train / ai đóng băng?

Theo Sec. 4.1 + implementation details:

| Module | Stage 1 | Stage 2 | Inference |
|---|---|---|---|
| **`F_theta` (inversion UNet)** | Train (lửa) | Train tiếp (lửa) | Freeze, chỉ forward |
| **SBv2 generator `G`** | Freeze (tuyết) — sinh synthetic / recon | Freeze | Freeze |
| **IP-Adapter** (`W^K_x`, `W^V_x`) | Train (paper: chỉ 2 ma trận nhánh ảnh; phần còn lại của `G` freeze) | **Freeze** — *“train only the inversion network, keeping the IP-Adapter branch … frozen”* | Freeze; dùng ARaM scales |
| **VAE / CLIP text / CLIP image** | Freeze | Freeze | Freeze |
| **Edit / ARaM / self-guided mask** | Không có trong train | Không có | Chỉ lúc infer |

**Vì sao freeze SBv2?** Giữ prior one-step đã distill; chỉ học đảo (`F_theta`), không phá generator.

**Vì sao Stage 2 chỉ train `F_theta`?** Giữ image prior IP đã học ở Stage 1; Stage 2 đóng domain gap ảnh thật (DISTS + `L_reg` trên inverted noise).

**Vì sao không train “edit”?** Train chỉ dạy **invert + reconstruct**. Edit = đổi `edit_prompt` + mask/ARaM trên checkpoint đã train.

#### 4. Cách minh chứng trước thầy (tránh tranh “vẽ lại đúng không”)

1. **Chiếu Figure 2 gốc paper** + citation arXiv/CVPR.
2. Chồng bảng legend lửa/tuyết + mũi tên Stage1→Stage2.
3. Trích caption: *“In stage 1, we warm up… At stage 2, we shift… continue to train…”*.
4. Sơ đồ Mermaid trên slide chỉ là **tóm tắt đã đối chiếu** — ghi “tương đương Fig. 2”; **không thay** figure gốc khi giải thích training.

### Việc đã làm trong repo

- Slide [`report/SLIDE_SwiftEdit.md`](report/SLIDE_SwiftEdit.md) §3a-fig + sửa §3a+/§3a++ (Stage 2 không còn ghi “train IP”).
- Báo cáo hộp training khớp paper (Stage 2 freeze IP).
- QA: *Fig. 2 paper: Stage 1 và Stage 2…* và nguyên lý train SBv2.

---

## Mục lục góp ý khác (tham chiếu nhanh)

Các góp ý trước (đã xử lý / đang theo dõi) — chi tiết nằm file chuyên biệt:

| # | Góp ý ngắn | File / chỗ trả lời |
|---|---|---|
| PSNR ~48.6 “quá đẹp” | Fidelity FP16↔FP32; đa metric; Tensor Core; ảnh extreme; torchmetrics + Img2Go | **F2** (mục trên) · [`report/AUDIT_PSNR_FIDELITY.md`](report/AUDIT_PSNR_FIDELITY.md) · PieBench [`EDIT_QUALITY_SUMMARY.md`](experimental_data/piebench_subset20_2026-06-14/EDIT_QUALITY_SUMMARY.md) |
| Demo phải khớp paper (chỉ prompt) | Tab Paper demo Gradio | `scripts/app_gradio.py` · `app_gradio_t4_xformers.py` |
| Nén model để làm gì? | Fit T4/Mac, latency, disk — không chỉ “thùng dư chỗ” | Slide §6b+ · BAO_CAO §8 |
| Đóng góp LoRA day↔night | Dataset sẵn night2day; pilot script | [`report/LORA_DAYNIGHT_PILOT.md`](report/LORA_DAYNIGHT_PILOT.md) |
