from __future__ import annotations

import torch
from torch.nn import functional as F

from .cfg import apply_cfg


class Diffusion:
    def __init__(self, steps: int, beta_start: float, beta_end: float, device: torch.device):
        if steps < 2 or not 0 < beta_start < beta_end < 1:
            raise ValueError("Invalid linear diffusion schedule")
        self.steps = steps
        betas = torch.linspace(beta_start, beta_end, steps, device=device, dtype=torch.float64)
        self.alpha_bar = torch.cumprod(1 - betas, dim=0).float()

    def loss(self, model, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # images/noise: [B,C,H,W], t/labels: [B]
        batch = images.shape[0]
        t = torch.randint(self.steps, (batch,), device=images.device)
        noise = torch.randn_like(images)
        alpha = self.alpha_bar[t, None, None, None]
        x_t = alpha.sqrt() * images + (1 - alpha).sqrt() * noise
        prediction = model(x_t, t, labels)
        return F.mse_loss(prediction.float(), noise.float())

    @torch.inference_mode()
    def sample(
        self,
        model,
        labels: torch.Tensor,
        image_size: int,
        channels: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        *,
        sampler: str = "ddim",
        clip_denoised: bool = False,
    ) -> tuple[torch.Tensor, int]:
        if not 1 <= num_steps <= self.steps:
            raise ValueError("sample steps must be between 1 and diffusion steps")
        if sampler not in ("ddim", "ddpm"):
            raise ValueError("sampler must be ddim or ddpm")
        device = labels.device
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn((labels.numel(), channels, image_size, image_size), device=device, generator=generator)
        times = torch.linspace(self.steps - 1, 0, num_steps, device=device).round().long().tolist()
        nfe = 0
        for i, step in enumerate(times):
            t = torch.full_like(labels, step)
            if guidance_scale == 0:
                eps = model(x, t, torch.full_like(labels, model.num_classes))
                nfe += 1
            elif guidance_scale == 1:
                eps = model(x, t, labels)
                nfe += 1
            else:
                eps_uncond = model(x, t, torch.full_like(labels, model.num_classes))
                eps_cond = model(x, t, labels)
                eps = apply_cfg(eps_uncond, eps_cond, guidance_scale)
                nfe += 2
            alpha_t = self.alpha_bar[step]
            alpha_prev = self.alpha_bar[times[i + 1]] if i + 1 < len(times) else torch.ones_like(alpha_t)
            # A float32 scalar times a bf16 tensor can still produce bf16.
            # New sampler options use FP32 coefficients/predictions; legacy DDIM
            # retains its arithmetic so historical default outputs remain valid.
            if sampler == "ddpm" or clip_denoised:
                eps = eps.float()
            x0 = (x - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
            if clip_denoised:
                x0 = x0.clamp(-1, 1)
            if sampler == "ddpm":
                # Exact q(x_s | x_t, x0) for the effective respaced beta.
                beta = 1 - alpha_t / alpha_prev
                mean = beta * alpha_prev.sqrt() / (1 - alpha_t) * x0 + (1 - alpha_prev) * (1 - beta).sqrt() / (1 - alpha_t) * x
                variance = ((1 - alpha_prev) / (1 - alpha_t) * beta).clamp_min(0)
                noise = torch.randn(x.shape, device=device, generator=generator)
                x = mean + variance.sqrt() * noise
            else:
                # Deterministic DDIM (eta=0); default retains final-only clipping.
                if clip_denoised:
                    eps = (x - alpha_t.sqrt() * x0) / (1 - alpha_t).sqrt()
                x = alpha_prev.sqrt() * x0 + (1 - alpha_prev).sqrt() * eps
        return x.clamp(-1, 1), nfe
