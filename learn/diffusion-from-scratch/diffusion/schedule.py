"""
Noise schedule for diffusion models.

Implements the linear beta schedule from the DDPM paper:
  beta_t linearly increases from beta_start to beta_end over T timesteps.

Key quantities:
  alpha_t     = 1 - beta_t
  alpha_bar_t = cumulative product of alpha_1 ... alpha_t

Forward process: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
"""

import torch
import numpy as np


class LinearSchedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, T, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # Precompute useful quantities
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_alpha = torch.sqrt(self.alpha)

        # For posterior q(x_{t-1} | x_t, x_0)
        alpha_bar_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_bar[:-1]])
        self.posterior_var = self.beta * (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bar)

    def q_sample(self, x_0, t, noise=None):
        """Forward process: sample x_t given x_0 and timestep t.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return sqrt_ab * x_0 + sqrt_1_ab * noise, noise

    def p_sample(self, model, x_t, t):
        """Reverse process (DDPM): denoise one step from x_t to x_{t-1}."""
        with torch.no_grad():
            t_tensor = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)
            eps_pred = model(x_t, t_tensor)

            beta_t = self.beta[t]
            alpha_t = self.alpha[t]
            sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t]
            sqrt_a = self.sqrt_alpha[t]

            # Mean of p(x_{t-1} | x_t)
            mean = (1.0 / sqrt_a) * (x_t - (beta_t / sqrt_1_ab) * eps_pred)

            if t > 0:
                noise = torch.randn_like(x_t)
                sigma = torch.sqrt(self.posterior_var[t])
                return mean + sigma * noise
            else:
                return mean

    def ddim_sample(self, model, x_t, t, t_prev, eta=0.0):
        """DDIM sampling step: from x_t to x_{t_prev}.

        eta=0: deterministic (DDIM)
        eta=1: stochastic (equivalent to DDPM)
        """
        with torch.no_grad():
            t_tensor = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)
            eps_pred = model(x_t, t_tensor)

            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_prev = self.alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0)

            # Predict x_0
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-1, 1)

            # Direction pointing to x_t
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps_pred

            x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt
            if eta > 0 and t_prev >= 0:
                x_prev += sigma * torch.randn_like(x_t)
            return x_prev
