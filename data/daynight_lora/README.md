# Dataset LoRA day ↔ night

Xem hướng dẫn đầy đủ: [`report/LORA_DAYNIGHT_PILOT.md`](../../report/LORA_DAYNIGHT_PILOT.md).

## Dataset sẵn (không cần Gemini)

Ưu tiên subsample từ **[huggan/night2day](https://huggingface.co/datasets/huggan/night2day)** (~20k cặp pix2pix / Transient Attributes) → ~80 train + ~20 hold-out.

Thêm: [paired-N2D](https://github.com/isurushanaka/paired-N2D), [N2D250K](https://github.com/isurushanaka/N2D250K) (subsample).  
Không paired sẵn: ACDC / BDD / Dark Zurich.

## Cấu trúc

```
daynight_lora/
  meta.jsonl
  train/
  holdout/
```

Copy `meta.example.jsonl` → `meta.jsonl` rồi thay đường dẫn ảnh thật.

## Prompt Gemini gợi ý

**Day → night**

```
Edit this photo to nighttime. Keep the exact same camera angle, objects,
and layout. Only change illumination: darker sky, artificial lights on,
realistic night colors. Do not add or remove any objects or people.
```

**Night → day**

```
Edit this photo to bright daylight. Keep the exact same camera angle,
objects, and layout. Only change illumination to natural sunny daylight.
Do not add or remove any objects or people.
```

## Checklist lọc

- [ ] Cùng góc máy / bố cục
- [ ] Không thêm/bớt object
- [ ] Lighting đổi rõ (day↔night)
- [ ] Đủ 50–150 cặp train + 15–20 hold-out
