from __future__ import annotations

import torch


def apply_cfg(uncond: torch.Tensor, cond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    """The sole definition of classifier-free guidance in this project."""
    # Avoid additional low-precision rounding in the conditional difference.
    if uncond.dtype in (torch.float16, torch.bfloat16) or cond.dtype in (torch.float16, torch.bfloat16):
        uncond, cond = uncond.float(), cond.float()
    return uncond + guidance_scale * (cond - uncond)
