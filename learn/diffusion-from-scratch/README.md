# diffusion-from-scratch

Minimal DDPM and DDIM implementation in PyTorch — train a diffusion model on MNIST from scratch in ~200 lines of core code.

## Results

### Training Loss
![Training Loss](assets/training_loss.png)

### DDPM Samples (1000 steps)
![DDPM Samples](assets/ddpm_samples.png)

### DDIM Samples (50 steps, deterministic)
![DDIM 50 Steps](assets/ddim_50_samples.png)

### DDIM Samples (20 steps, deterministic)
![DDIM 20 Steps](assets/ddim_20_samples.png)

### Denoising Process Visualization
![Denoising Steps](assets/denoising_steps.png)

### Forward Diffusion Process
![Forward Process](assets/forward_process.png)

### Noise Schedule
![Noise Schedule](assets/noise_schedule.png)

## What's Inside

| File | Description |
|------|-------------|
| `diffusion/schedule.py` | Linear beta schedule, forward process q(x_t\|x_0), DDPM & DDIM reverse sampling |
| `diffusion/unet.py` | U-Net with sinusoidal time embeddings, residual blocks, self-attention |
| `01_forward_process.py` | Visualize how noise is progressively added |
| `02_train_ddpm.py` | Train the denoising model on MNIST (15 epochs, ~45 min on MPS) |
| `03_sample_ddpm.py` | Generate images with full 1000-step DDPM sampling |
| `04_sample_ddim.py` | Fast sampling with DDIM (50 and 20 steps) |
| `train_all.py` | Run the entire pipeline end-to-end |

## Key Concepts

**DDPM (Denoising Diffusion Probabilistic Models)**
- Forward process: gradually add Gaussian noise over T=1000 steps
- Reverse process: learn to denoise step by step
- Training: predict the noise added at each timestep (simple MSE loss)

**DDIM (Denoising Diffusion Implicit Models)**
- Same trained model, different sampling procedure
- Skip timesteps for 20x faster generation (1000 → 50 steps)
- Deterministic sampling when eta=0

**Architecture**
- U-Net with encoder-decoder structure and skip connections
- Sinusoidal positional embeddings for timestep conditioning
- Self-attention at the bottleneck (7x7 resolution)
- GroupNorm + SiLU activations throughout
- ~1.4M parameters

## Quick Start

```bash
pip install -r requirements.txt

# Visualize forward process
python 01_forward_process.py

# Train DDPM on MNIST
python 02_train_ddpm.py

# Generate samples
python 03_sample_ddpm.py   # DDPM (1000 steps)
python 04_sample_ddim.py   # DDIM (50 & 20 steps)

# Or run everything at once
python train_all.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision, matplotlib, numpy, tqdm

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (Song et al., 2020)

## Related

- Blog post: [Diffusion Models from Scratch: DDPM Training in PyTorch](https://tildalice.io/diffusion-models-from-scratch-ddpm-pytorch/) on TildAlice
