"""
Simple U-Net for diffusion models.

Architecture (for 28x28 MNIST):
  Encoder:  inc(1→64) → down1(64→128, 14x14) → down2(128→256, 7x7)
  Middle:   mid(256→256, 7x7) with self-attention
  Decoder:  up1(256+128→128, 14x14) → up2(128+64→64, 28x28) → out(64→1)

Time conditioning via sinusoidal embeddings projected to each block.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal positional encoding for timestep t."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class ResBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Simple self-attention for spatial features."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.out = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q.transpose(1, 2) @ k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (v @ attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.out(out)


class Down(nn.Module):
    """Downsample: ResBlock + MaxPool."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block = ResBlock(in_ch, out_ch, time_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t_emb):
        x = self.block(x, t_emb)
        return self.pool(x), x  # return skip connection before pooling


class Up(nn.Module):
    """Upsample: Interpolate + Concat skip + ResBlock."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block = ResBlock(in_ch, out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t_emb)


class UNet(nn.Module):
    """U-Net noise prediction network for 28x28 grayscale images."""

    def __init__(self, in_ch=1, base_ch=64, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Encoder
        self.inc_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1) # 1→base_ch
        self.inc = ResBlock(base_ch, base_ch, time_dim)          # 28x28
        self.down1 = Down(base_ch, base_ch * 2, time_dim)       # → 14x14
        self.down2 = Down(base_ch * 2, base_ch * 4, time_dim)   # → 7x7

        # Middle
        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, time_dim)
        self.mid_attn = SelfAttention(base_ch * 4)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_dim)

        # Decoder (concat channels: bottleneck + skip)
        self.up1 = Up(base_ch * 4 + base_ch * 4, base_ch * 2, time_dim)  # 7→14
        self.up2 = Up(base_ch * 2 + base_ch * 2, base_ch, time_dim)      # 14→28

        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, 1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Encoder
        x1 = self.inc(self.inc_conv(x), t_emb)  # 28x28, base_ch
        x2, skip1 = self.down1(x1, t_emb)      # 14x14, base_ch*2
        x3, skip2 = self.down2(x2, t_emb)      # 7x7,   base_ch*4

        # Middle
        x3 = self.mid1(x3, t_emb)
        x3 = self.mid_attn(x3)
        x3 = self.mid2(x3, t_emb)

        # Decoder
        x = self.up1(x3, skip2, t_emb)         # 14x14
        x = self.up2(x, skip1, t_emb)          # 28x28

        return self.out(x)
