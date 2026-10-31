from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
from datetime import datetime, timezone
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.utils import count_parameters

from .config import Config, load_config
from .data import TrainImages
from .diffusion import Diffusion
from .utils import load_checkpoint, make_model, save_checkpoint, save_image_grid, set_seed, write_json


def learning_rate_at_step(cfg: Config, completed_steps: int) -> float:
    """LR for the next optimizer update; derived from checkpointed global step."""
    if completed_steps < cfg.warmup_steps:
        factor = (completed_steps + 1) / cfg.warmup_steps
    elif cfg.learning_rate_schedule == "linear":
        decay_updates = cfg.num_train_steps - cfg.warmup_steps
        factor = max(0.0, 1.0 - (completed_steps - cfg.warmup_steps) / max(1, decay_updates - 1))
    else:
        factor = 1.0
    return cfg.learning_rate * factor


def drop_classes(labels: torch.Tensor, num_classes: int, probability: float) -> torch.Tensor:
    result = labels.clone()
    result[torch.rand(labels.shape, device=labels.device) < probability] = num_classes
    return result


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for target, source in zip(ema.parameters(), model.parameters(), strict=True):
        target.lerp_(source.detach(), 1 - decay)
    for target, source in zip(ema.buffers(), model.buffers(), strict=True):
        target.copy_(source)


def epoch_loader(dataset: TrainImages, batch_size: int, workers: int, seed: int, epoch: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed + epoch)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    return DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=False,
        generator=torch.Generator().manual_seed(seed + epoch + 1_000_000),
    )


@torch.no_grad()
def training_diagnostics(model, ema, extra_emas, diffusion, cfg, run_dir, step, device):
    labels=torch.tensor(cfg.diagnostic_labels or tuple(range(min(4,model.num_classes))),device=device)
    if torch.any((labels<0)|(labels>=model.num_classes)):
        raise ValueError("Invalid diagnostic class label")
    labels=labels.repeat_interleave(2)
    directory=run_dir/'diagnostics'/f'step_{step:07d}'
    records=[]
    was_training=model.training
    model.eval()
    try:
        tracks={'raw':model,f'ema_{cfg.ema_decay:g}':ema,
                **{f'ema_{key}':net for key,net in extra_emas.items()}}
        for name,network in tracks.items():
            for sampler,clip in [('ddim',False),('ddim',True),('ddpm',True)]:
                amp=torch.autocast('cuda',dtype=torch.bfloat16 if cfg.amp_dtype=='bf16' else torch.float16) if device.type=='cuda' and cfg.amp_dtype!='none' else contextlib.nullcontext()
                with amp:
                    samples,nfe=diffusion.sample(network,labels,cfg.image_size,3,cfg.sample_steps,1,cfg.seed,
                                                  sampler=sampler,clip_denoised=clip)
                filename=f'{name}_{sampler}_clip{int(clip)}_s1.png'
                save_image_grid(samples,directory/filename)
                records.append({'file':filename,'weights':name,'sampler':sampler,'clip_denoised':clip,'guidance_scale':1,'nfe':nfe})
        for sampler in ('ddim','ddpm'):
            for scale in (0,2,3):
                amp=torch.autocast('cuda',dtype=torch.bfloat16 if cfg.amp_dtype=='bf16' else torch.float16) if device.type=='cuda' and cfg.amp_dtype!='none' else contextlib.nullcontext()
                with amp:
                    samples,nfe=diffusion.sample(ema,labels,cfg.image_size,3,cfg.sample_steps,scale,cfg.seed,
                                                  sampler=sampler,clip_denoised=True)
                filename=f'ema_{cfg.ema_decay:g}_{sampler}_clip1_s{scale}.png'
                save_image_grid(samples,directory/filename)
                records.append({'file':filename,'weights':f'ema_{cfg.ema_decay:g}','sampler':sampler,
                                'clip_denoised':True,'guidance_scale':scale,'nfe':nfe})
        write_json(directory/'metadata.json',{'step':step,'seed':cfg.seed,'labels':labels.tolist(),'variants':records})
    finally:
        model.train(was_training)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    cfg, raw = load_config(args.config)
    set_seed(cfg.seed)
    dataset = TrainImages(cfg.data_root, cfg.split_dir, cfg.image_size, cfg.num_classes,
                          cfg.horizontal_flip_probability, cfg.augmentation_degrees,
                          cfg.augmentation_translate, cfg.augmentation_scale_delta, split_file=cfg.train_manifest,
                          cache=cfg.cache_training_images)
    train_paths_hash=hashlib.sha256('\n'.join(dataset.paths).encode()).hexdigest()
    if cfg.in_channels != 3:
        raise ValueError("TrainImages currently produces RGB images; set in_channels: 3")
    num_classes = len(dataset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and cfg.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support bf16; choose fp16")
    model = make_model(cfg, num_classes).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    extra_emas={f'{decay:g}':copy.deepcopy(ema) for decay in cfg.extra_ema_decays}
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.amp_dtype == "fp16")
    diffusion = Diffusion(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end, device)
    run_dir = Path(cfg.output_dir) / cfg.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "config.json")
    source_dir = run_dir / "source_snapshot" / "teacher"
    if not args.resume and source_dir.exists():
        raise FileExistsError(f"Run source snapshot already exists: {source_dir}")
    if not source_dir.exists():
        shutil.copytree(Path(__file__).parent, source_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip() or None
    git_dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False).stdout.strip())
    metrics = {"experiment_name": cfg.experiment_name, "git_commit": commit, "seed": cfg.seed,
               "parameter_count": parameters, "train_images": len(dataset), "num_classes": num_classes,
               "class_names": dataset.classes, "device": str(device), "peak_vram_bytes": None,
               "completed_steps": 0, "fid": None, "git_dirty_at_start": git_dirty,
               "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "optimizer": {"name":"torch.optim.AdamW","betas":[0.9,0.999],"eps":1e-8,"weight_decay":cfg.weight_decay},
               "ema_decays":[cfg.ema_decay,*cfg.extra_ema_decays],"train_manifest":cfg.train_manifest,
               "train_paths_sha256":train_paths_hash,
               "fid_note": "Teacher FID is reported separately with teacher.evaluate; no training-time validation tuning."}
    step = epoch = batch_in_epoch = images_seen = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, "cpu")
        if checkpoint["class_names"] != dataset.classes:
            raise ValueError("Checkpoint class mapping differs from training data")
        if checkpoint.get('train_paths_sha256',train_paths_hash)!=train_paths_hash or checkpoint['config'].get('train_manifest')!=cfg.train_manifest:
            raise ValueError('Checkpoint training split differs from config')
        if set(checkpoint.get('ema_tracks',{}))!=set(extra_emas):
            raise ValueError('Checkpoint EMA tracks differ from config')
        architecture_keys = ("image_size", "in_channels", "num_classes", "model_channels", "channel_mult",
                             "num_res_blocks", "attention_resolutions", "dropout", "prediction_type",
                             "noise_schedule", "diffusion_steps", "beta_start", "beta_end")
        for key in architecture_keys:
            current = raw[key]
            previous = checkpoint["config"][key]
            if current != previous:
                raise ValueError(f"Checkpoint {key} differs from config")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        for key,network in extra_emas.items():
            network.load_state_dict(checkpoint['ema_tracks'][key])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        step, epoch, batch_in_epoch = checkpoint["step"], checkpoint["epoch"], checkpoint["batch_in_epoch"]
        images_seen = checkpoint.get("images_seen", step * cfg.batch_size * cfg.gradient_accumulation)
        metrics["resumed_from"] = str(args.resume)
        metrics["initial_step"] = step
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    log_file = run_dir / "train.log"
    accumulated_loss = 0.0
    accumulated_microbatches = 0
    def checkpoint_state() -> dict:
        return {"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                "ema_tracks":{key:network.state_dict() for key,network in extra_emas.items()},
                "train_paths_sha256":train_paths_hash,
                "scaler": scaler.state_dict(), "step": step, "epoch": epoch, "batch_in_epoch": batch_in_epoch,
                "images_seen": images_seen,
                "config": raw, "class_names": dataset.classes, "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
    while step < cfg.num_train_steps:
        loader = epoch_loader(dataset, cfg.batch_size, cfg.num_workers, cfg.seed, epoch)
        for batch_number, (images, labels) in enumerate(loader):
            if batch_number < batch_in_epoch:
                continue
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            labels = drop_classes(labels, num_classes, cfg.class_dropout)
            if batch_number == batch_in_epoch and step == 0:
                assert images.shape[1:] == (3, cfg.image_size, cfg.image_size)
                assert images.min() >= -1 and images.max() <= 1
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16) if device.type == "cuda" and cfg.amp_dtype != "none" else contextlib.nullcontext()
            with amp:
                loss = diffusion.loss(model, images, labels) / cfg.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at optimizer step {step + 1}")
            scaler.scale(loss).backward()
            accumulated_loss += float(loss.detach()) * cfg.gradient_accumulation
            accumulated_microbatches += 1
            images_seen += images.shape[0]
            batch_in_epoch = batch_number + 1
            # Also update at an epoch boundary so small datasets can train.
            last_batch = batch_in_epoch == len(loader)
            accumulation_boundary = batch_in_epoch % cfg.gradient_accumulation == 0 or last_batch
            if not accumulation_boundary:
                continue
            scaler.unscale_(optimizer)
            if accumulated_microbatches < cfg.gradient_accumulation:
                correction = cfg.gradient_accumulation / accumulated_microbatches
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at optimizer step {step + 1}")
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at_step(cfg, step)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step_loss = accumulated_loss / accumulated_microbatches
            accumulated_loss = 0.0
            accumulated_microbatches = 0
            if scaler.get_scale() < previous_scale:
                continue
            update_ema(ema, model, cfg.ema_decay)
            for key,network in extra_emas.items():
                update_ema(network,model,float(key))
            step += 1
            metrics["completed_steps"] = step
            metrics["last_loss"] = step_loss
            metrics["elapsed_seconds"] = round(time.monotonic() - start, 2)
            metrics["peak_vram_bytes"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
            metrics["images_seen"] = images_seen
            metrics["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            line = json.dumps({"step": step, "loss": step_loss, "lr": optimizer.param_groups[0]["lr"],
                               "grad_norm": float(grad_norm), "images_seen": images_seen,
                               "elapsed_seconds": metrics["elapsed_seconds"],
                               "peak_vram_bytes": metrics["peak_vram_bytes"]})
            with log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            if step % cfg.log_interval == 0 or step == 1:
                print(line, flush=True)
            if cfg.sample_interval > 0 and step % cfg.sample_interval == 0:
                ema.eval()
                sample_labels = torch.arange(min(4, num_classes), device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16) if device.type == "cuda" and cfg.amp_dtype != "none" else contextlib.nullcontext():
                    samples, nfe = diffusion.sample(ema, sample_labels, cfg.image_size, 3, cfg.sample_steps, cfg.guidance_scale, cfg.seed,
                                                    sampler=cfg.sample_sampler,clip_denoised=cfg.sample_clip_denoised)
                save_image_grid(samples, run_dir / f"samples_step_{step:07d}.png")
                metrics["sample_nfe"] = nfe
                metrics["sample_guidance_scale"] = cfg.guidance_scale
                metrics['sample_sampler']=cfg.sample_sampler
            if cfg.diagnostic_interval>0 and step%cfg.diagnostic_interval==0:
                training_diagnostics(model,ema,extra_emas,diffusion,cfg,run_dir,step,device)
            if step % cfg.save_interval == 0 or step == cfg.num_train_steps:
                save_checkpoint(run_dir / "latest.pt", checkpoint_state())
            write_json(run_dir / "metrics.json", metrics)
            if step >= cfg.num_train_steps:
                save_checkpoint(run_dir / "final.pt", checkpoint_state())
                break
        else:
            epoch += 1
            batch_in_epoch = 0
    print(f"final checkpoint: {run_dir / 'final.pt'}", flush=True)


if __name__ == "__main__":
    main()
