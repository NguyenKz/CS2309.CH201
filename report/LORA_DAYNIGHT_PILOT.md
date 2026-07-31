# Pilot LoRA day ↔ night cho SwiftEdit

> Đóng góp ứng dụng/domain adapter — **không** retrain full Stage1/2 paper.  
> Mục tiêu: cải thiện edit ngày→đêm (và ngược lại) trên backbone gen, đo bằng độ đo chỏi nhau.

## 1. Khả thi?

| Câu hỏi | Trả lời ngắn |
|---|---|
| T4 16GB đủ train không? | **Có** — LoRA rank 4–16, batch 1, gradient checkpointing, 512² |
| Cặp (ảnh gốc, Gemini edit) train `F_θ` được không? | **Không khớp** recipe paper (Stage1 cần noise synthetic; Stage2 DISTS recon) |
| Cách làm thực dụng? | LoRA trên **Generation UNet (SBv2)**; đóng băng `F_θ` + IP-Adapter |
| Bao nhiêu ảnh? | **50–150 cặp sạch** train + **15–20** hold-out; 1–2 cặp sẽ overfit |
| Thời gian T4? | ~2–8 giờ cho 1k–5k step (Colab) |

## 2. Layout dataset

```
data/daynight_lora/
  README.md                 ← hướng dẫn Gemini (file này trỏ xuống dưới)
  meta.jsonl                ← mỗi dòng 1 cặp
  train/
    0001_day.jpg
    0001_night.jpg
    ...
  holdout/
    h01_day.jpg
    h01_night.jpg
```

`meta.jsonl` ví dụ:

```json
{"id":"0001","split":"train","day":"train/0001_day.jpg","night":"train/0001_night.jpg","src_day":"a daytime street photo","edit_night":"the same street at night, realistic lighting","src_night":"a nighttime street photo","edit_day":"the same street in daylight, sunny"}
```

## 3. Nguồn dữ liệu — dataset sẵn (ưu tiên) + Gemini (tuỳ chọn)

**Không bắt buộc tự tạo bằng Gemini.** Có bộ paired day↔night công khai:

| Dataset | Quy mô | Link | Ghi chú cho pilot |
|---|---|---|---|
| **night2day** (pix2pix / Transient Attributes) | ~20k cặp | [Hugging Face huggan/night2day](https://huggingface.co/datasets/huggan/night2day) | **Nên dùng trước** — subsample ~80 train + ~20 hold-out |
| Transient Attributes (gốc) | Webcam scenes | [transattr.cs.brown.edu](http://transattr.cs.brown.edu/) | Nguồn của night2day |
| paired-N2D | Paired street | [github.com/isurushanaka/paired-N2D](https://github.com/isurushanaka/paired-N2D) | Night thường synthetic (ToDayGAN) |
| N2D250K | ~250k cặp | [github.com/isurushanaka/N2D250K](https://github.com/isurushanaka/N2D250K) | Quá lớn — chỉ subsample |

**Tránh làm paired nếu chưa xử lý:** ACDC, BDD100K, Dark Zurich — có day/night nhưng không cùng góc máy từng cặp.

**Gemini** (tuỳ chọn, domain riêng): chọn ~40–80 scene; prompt giữ layout:

```
Edit this photo to nighttime. Keep the exact same camera angle, objects,
and layout. Only change illumination: darker sky, artificial lights,
realistic night colors. Do not add or remove objects.
```

Lọc tay → **50–150 cặp sạch** + **15–20** hold-out.

## 4. Train

```bash
# Ví dụ Colab T4 / local CUDA
python scripts/train_lora_daynight.py \
  --data-root data/daynight_lora \
  --pretrained SwiftEdit/swiftedit_weights/sbv2_0.5 \
  --output-dir results/lora_daynight \
  --rank 8 --lr 1e-4 --steps 2000 --batch-size 1
```

Script dùng Diffusers + PEFT LoRA trên UNet; resolution 512; fp16 + gradient checkpointing.

## 5. Eval (độ đo chỏi nhau)

```bash
python scripts/eval_lora_daynight.py \
  --data-root data/daynight_lora \
  --split holdout \
  --baseline-outdir results/daynight_eval/baseline \
  --lora-outdir results/daynight_eval/lora \
  --lora-path results/lora_daynight
```

| Độ đo | Kỳ vọng nếu LoRA hữu ích |
|---|---|
| CLIP(edit, “night…”) / zero-shot night label | ↑ so baseline không LoRA |
| DINO / CLIP-I(source, edit) | không sụp mạnh |
| LPIPS toàn ảnh | phụ (lighting được phép đổi) |
| Human 1–5 trên 20 hold-out | báo cả case fail |

## 6. Gắn vào SwiftEdit inference

Sau khi có `pytorch_lora_weights.safetensors` (hoặc thư mục PEFT):

```python
from scripts.lora_utils import attach_lora_to_unet
attach_lora_to_unet(ip_sb_model.unet, "results/lora_daynight")
```

Hoặc merge weight rồi load checkpoint gen như cũ (fallback nếu PEFT lệch version).

## 7. Rủi ro nói thẳng với thầy

- Đây là **adapter domain**, không phải cải tiến thuật toán SwiftEdit gốc.
- Gemini synthetic có bias; cần báo failure cases.
- Nếu LoRA không cải CLIP night trên hold-out → ghi nhận ablation âm (vẫn là đóng góp thực nghiệm).
