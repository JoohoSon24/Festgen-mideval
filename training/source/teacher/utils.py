from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .config import Config
from .model import ClassConditionalUNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_model(cfg: Config, num_classes: int) -> ClassConditionalUNet:
    return ClassConditionalUNet(
        image_size=cfg.image_size,
        in_channels=cfg.in_channels,
        num_classes=num_classes,
        model_channels=cfg.model_channels,
        channel_mult=cfg.channel_mult,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        dropout=cfg.dropout,
        activation_checkpoint_resolutions=cfg.activation_checkpoint_resolutions,
    )


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def checkpoint_weights(checkpoint: dict[str, Any], name: str):
    if name in {'model','ema'}:
        return checkpoint[name]
    if name.startswith('ema:') and name[4:] in checkpoint.get('ema_tracks',{}):
        return checkpoint['ema_tracks'][name[4:]]
    raise ValueError(f'Unknown weights {name}; use model, ema, or ema:<tracked decay>')


def save_image_grid(images: torch.Tensor, path: Path, columns: int = 4) -> None:
    images = ((images.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    n, channels, height, width = images.shape
    if channels != 3:
        raise ValueError("PNG grid requires RGB images")
    rows = (n + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * height))
    for i, image in enumerate(images):
        canvas.paste(Image.fromarray(image.permute(1, 2, 0).numpy()), ((i % columns) * width, (i // columns) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
