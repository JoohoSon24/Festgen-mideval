"""Inference helpers for the unchanged FastGen base Model interface."""
from __future__ import annotations
import contextlib
import json
from pathlib import Path
import torch
from teacher.model import ClassConditionalUNet
from teacher.diffusion import Diffusion
from .core import sample_velocity

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "submission_model.json"


def submission_config():
    with _CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def submission_backbone():
    config = submission_config()
    return ClassConditionalUNet(
        image_size=config["image_size"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        model_channels=config["model_channels"],
        channel_mult=tuple(config["channel_mult"]),
        num_res_blocks=config["num_res_blocks"],
        attention_resolutions=tuple(config["attention_resolutions"]),
        dropout=config["dropout"],
        activation_checkpoint_resolutions=tuple(config.get("activation_checkpoint_resolutions", [])),
    )


def initialize_buffers(model,device,steps):
    if steps not in (1,4):raise ValueError('Submission supports exactly 1 or 4 NFE')
    config = submission_config()
    schedule=Diffusion(config["diffusion_steps"],config["beta_start"],config["beta_end"],torch.device(device))
    grid=torch.tensor(config["sample_grids"][str(steps)],device=device,dtype=torch.long)
    model.register_buffer('alpha_bar',schedule.alpha_bar)
    model.register_buffer('sampling_times',grid)
    model.register_buffer('sampling_seed',torch.tensor(42,device=device))
    model._batch_index=0
    model.nfe_per_sample=steps
    model.to(device)


@torch.inference_mode()
def submission_sample(model,shape,device,category,**kwargs):
    device=torch.device(device)
    if tuple(shape[1:])!=(3,64,64) or category is None or len(category)!=shape[0]:
        raise ValueError('Expected categories and shape [B,3,64,64]')
    category=category.to(device=device,dtype=torch.long)
    if torch.any((category<0)|(category>=151)):raise ValueError('Invalid class')
    model.to(device).eval()
    seed=int(kwargs.get('seed',int(model.sampling_seed)+model._batch_index))
    if 'seed' not in kwargs:model._batch_index+=1
    amp=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
    with amp:
        return sample_velocity(model.backbone,model.alpha_bar,model.sampling_times,category,seed,False)
