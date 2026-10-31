from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    experiment_name: str
    data_root: str
    split_dir: str
    image_size: int
    in_channels: int
    num_classes: int | None
    model_channels: int
    channel_mult: tuple[int, ...]
    num_res_blocks: int
    attention_resolutions: tuple[int, ...]
    dropout: float
    batch_size: int
    gradient_accumulation: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    num_train_steps: int
    warmup_steps: int
    grad_clip: float
    ema_decay: float
    class_dropout: float
    prediction_type: str
    noise_schedule: str
    diffusion_steps: int
    beta_start: float
    beta_end: float
    sample_steps: int
    guidance_scale: float
    amp_dtype: str
    seed: int
    log_interval: int
    save_interval: int
    sample_interval: int
    output_dir: str
    learning_rate_schedule: str = "constant"
    horizontal_flip_probability: float = 0.0
    augmentation_degrees: float = 0.0
    augmentation_translate: float = 0.0
    augmentation_scale_delta: float = 0.0
    extra_ema_decays: tuple[float, ...] = ()
    train_manifest: str | None = None
    development_reference_manifest: str | None = None
    sample_sampler: str = "ddim"
    sample_clip_denoised: bool = False
    diagnostic_interval: int = 0
    diagnostic_labels: tuple[int, ...] = ()
    activation_checkpoint_resolutions: tuple[int, ...] = ()
    cache_training_images: bool = False


def load_config(path: str | Path) -> tuple[Config, dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a JSON object")
    raw["channel_mult"] = tuple(raw["channel_mult"])
    raw["attention_resolutions"] = tuple(raw["attention_resolutions"])
    raw["extra_ema_decays"] = tuple(raw.get("extra_ema_decays", ()))
    raw["diagnostic_labels"] = tuple(raw.get("diagnostic_labels", ()))
    raw["activation_checkpoint_resolutions"] = tuple(raw.get("activation_checkpoint_resolutions", ()))
    cfg = Config(**raw)
    if cfg.prediction_type != "epsilon":
        raise ValueError("This baseline implements epsilon prediction only")
    if cfg.noise_schedule != "linear":
        raise ValueError("This baseline implements a linear beta schedule only")
    if cfg.amp_dtype not in {"bf16", "fp16", "none"}:
        raise ValueError("amp_dtype must be bf16, fp16, or none")
    if not 0 <= cfg.class_dropout <= 1:
        raise ValueError("class_dropout must be in [0, 1]")
    if cfg.batch_size < 1 or cfg.gradient_accumulation < 1:
        raise ValueError("batch_size and gradient_accumulation must be positive")
    if cfg.learning_rate_schedule not in {"constant", "linear"}:
        raise ValueError("learning_rate_schedule must be constant or linear")
    if not 0 <= cfg.ema_decay < 1 or any(not 0 <= d < 1 for d in cfg.extra_ema_decays):
        raise ValueError("EMA decays must be in [0, 1)")
    if len(set((cfg.ema_decay, *cfg.extra_ema_decays))) != len(cfg.extra_ema_decays) + 1:
        raise ValueError("EMA decays must be unique")
    if cfg.sample_sampler not in {"ddim", "ddpm"} or cfg.diagnostic_interval < 0:
        raise ValueError("Invalid diagnostic sampling configuration")
    if cfg.num_train_steps < 1 or cfg.warmup_steps < 0:
        raise ValueError("Require positive training steps and nonnegative warmup_steps")
    if not 0 <= cfg.horizontal_flip_probability <= 1:
        raise ValueError("horizontal_flip_probability must be in [0, 1]")
    if not 0 <= cfg.augmentation_degrees <= 15 or not 0 <= cfg.augmentation_translate <= 0.1 or not 0 <= cfg.augmentation_scale_delta <= 0.1:
        raise ValueError("Augmentation must be mild: degrees <= 15, translation/scale delta <= 0.1")
    return cfg, raw
