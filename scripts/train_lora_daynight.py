#!/usr/bin/env python3
"""
Train LoRA day↔night trên Generation UNet (SBv2 / SD-family).

Thiết kế cho Colab T4 16GB: batch=1, rank≤16, gradient checkpointing, fp16.

Ví dụ:
  python scripts/train_lora_daynight.py \\
    --data-root data/daynight_lora \\
    --pretrained SwiftEdit/swiftedit_weights/sbv2_0.5 \\
    --text-encoder-id stabilityai/stable-diffusion-2-1-base \\
    --output-dir results/lora_daynight \\
    --rank 8 --steps 2000

Xem report/LORA_DAYNIGHT_PILOT.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]


class DayNightPairDataset(Dataset):
    """Mỗi mẫu: ảnh target (night hoặc day) + caption tương ứng."""

    def __init__(self, data_root: Path, split: str = "train", size: int = 512):
        self.root = data_root
        meta_path = data_root / "meta.jsonl"
        if not meta_path.exists():
            example = data_root / "meta.example.jsonl"
            raise FileNotFoundError(
                f"Thiếu {meta_path}. Copy từ {example} và thêm ảnh train/holdout."
            )
        rows = []
        with meta_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("split", "train") != split:
                    continue
                # Hai hướng: day→night và night→day
                rows.append(
                    {
                        "image": self.root / row["night"],
                        "prompt": row.get("edit_night")
                        or "the same scene at night, realistic lighting",
                    }
                )
                rows.append(
                    {
                        "image": self.root / row["day"],
                        "prompt": row.get("edit_day")
                        or "the same scene in daylight, sunny",
                    }
                )
        if not rows:
            raise RuntimeError(f"Không có mẫu split={split} trong {meta_path}")
        missing = [str(r["image"]) for r in rows if not r["image"].exists()]
        if missing:
            raise FileNotFoundError(
                "Thiếu ảnh (thêm file hoặc sửa meta.jsonl):\n- " + "\n- ".join(missing[:8])
            )
        self.rows = rows
        self.tf = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(row["image"]).convert("RGB")
        return {"pixel_values": self.tf(img), "prompt": row["prompt"]}


def parse_args():
    p = argparse.ArgumentParser(description="Train LoRA day↔night trên gen UNet")
    p.add_argument("--data-root", type=Path, default=ROOT / "data" / "daynight_lora")
    p.add_argument(
        "--pretrained",
        type=Path,
        default=ROOT / "SwiftEdit" / "swiftedit_weights" / "sbv2_0.5",
        help="Thư mục UNet SBv2 (subfolder unet nếu cần)",
    )
    p.add_argument(
        "--text-encoder-id",
        default="stabilityai/stable-diffusion-2-1-base",
        help="HF id hoặc path có tokenizer + text_encoder + vae",
    )
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "lora_daynight")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--seed", type=int, default=250101049)
    p.add_argument("--save-every", type=int, default=500)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from peft import LoraConfig, get_peft_model
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as exc:
        print(
            "Cần: pip install diffusers peft transformers accelerate",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] Không có CUDA — train sẽ rất chậm (demo logic only).")

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds = DayNightPairDataset(args.data_root, split="train", size=args.resolution)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    unet_path = args.pretrained
    if (unet_path / "unet").exists():
        unet_path = unet_path / "unet"
    unet = UNet2DConditionModel.from_pretrained(str(unet_path))
    tokenizer = CLIPTokenizer.from_pretrained(args.text_encoder_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.text_encoder_id, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(args.text_encoder_id, subfolder="vae")
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.text_encoder_id, subfolder="scheduler"
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    text_encoder.to(device)
    unet.to(device)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.enable_gradient_checkpointing()
    unet.train()

    opt = torch.optim.AdamW(
        (p for p in unet.parameters() if p.requires_grad),
        lr=args.lr,
    )

    print(
        f"[train] samples={len(ds)} steps={args.steps} rank={args.rank} "
        f"device={device} out={args.output_dir}"
    )
    step = 0
    data_iter = iter(loader)
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        pixel = batch["pixel_values"].to(
            device, dtype=vae.dtype if device.type == "cuda" else torch.float32
        )
        prompts = batch["prompt"]
        with torch.no_grad():
            latents = vae.encode(pixel).latent_dist.sample() * vae.config.scaling_factor
            tokens = tokenizer(
                list(prompts),
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            encoder_hidden = text_encoder(tokens.input_ids.to(device))[0]

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
        ).long()
        noisy = noise_scheduler.add_noise(latents.float(), noise.float(), timesteps)

        pred = unet(noisy.to(unet.dtype), timesteps, encoder_hidden.to(unet.dtype)).sample
        loss = F.mse_loss(pred.float(), noise.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        step += 1
        if step % 50 == 0 or step == 1:
            print(f"  step {step}/{args.steps}  loss={loss.item():.4f}")
        if step % args.save_every == 0 or step == args.steps:
            save_dir = args.output_dir / f"checkpoint-{step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            unet.save_pretrained(str(save_dir))
            print(f"  saved {save_dir}")

    unet.save_pretrained(str(args.output_dir))
    (args.output_dir / "train_meta.json").write_text(
        json.dumps(
            {
                "rank": args.rank,
                "steps": args.steps,
                "lr": args.lr,
                "pretrained": str(args.pretrained),
                "n_train_images": len(ds),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] LoRA → {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
