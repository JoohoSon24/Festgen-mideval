from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.utils import count_parameters

from .cfg import apply_cfg
from .config import load_config
from .data import TrainImages
from .diffusion import Diffusion
from .train import drop_classes, update_ema
from .utils import load_checkpoint, make_model, save_checkpoint, save_image_grid, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/teacher/debug.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg, raw = load_config(args.config)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    data_path = Path(cfg.data_root) / "PokemonData"
    if data_path.is_dir():
        dataset = TrainImages(cfg.data_root, cfg.split_dir, cfg.image_size, cfg.num_classes, split_file=cfg.train_manifest)
        images, labels = next(iter(DataLoader(dataset, batch_size=cfg.batch_size)))
        num_classes = len(dataset.classes)
        data_result = {"source": "real training directory", "count": len(dataset), "classes": num_classes}
    else:
        num_classes = cfg.num_classes or 151
        images = torch.rand((cfg.batch_size, 3, cfg.image_size, cfg.image_size)) * 2 - 1
        labels = torch.arange(cfg.batch_size) % num_classes
        data_result = {"source": "synthetic fallback", "dataset_batch": "SKIPPED: training directory absent"}
    images, labels = images.to(device), labels.to(device)
    assert images.shape == (cfg.batch_size, 3, cfg.image_size, cfg.image_size)
    assert images.min() >= -1 and images.max() <= 1
    model = make_model(cfg, num_classes).to(device)
    ema = make_model(cfg, num_classes).to(device).eval().requires_grad_(False)
    ema.load_state_dict(model.state_dict())
    t = torch.zeros((cfg.batch_size,), device=device, dtype=torch.long)
    conditional = model(images, t, labels)
    unconditional = model(images, t, torch.full_like(labels, num_classes))
    assert conditional.shape == images.shape == unconditional.shape
    assert torch.equal(apply_cfg(unconditional, conditional, 0), unconditional)
    assert torch.equal(drop_classes(labels, num_classes, 1), torch.full_like(labels, num_classes))
    assert torch.equal(drop_classes(labels, num_classes, 0), labels)
    diffusion = Diffusion(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    initial_loss = None
    losses = []
    # Fixed noisy batch gives a useful overfit sanity check without training on validation data.
    noise = torch.randn_like(images)
    fixed_t = torch.full_like(labels, cfg.diffusion_steps // 2)
    alpha = diffusion.alpha_bar[fixed_t, None, None, None]
    noisy = alpha.sqrt() * images + (1 - alpha).sqrt() * noise
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        pred = model(noisy, fixed_t, labels)
        loss = F.mse_loss(pred.float(), noise.float())
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        losses.append(float(loss.detach()))
        optimizer.step()
        update_ema(ema, model, cfg.ema_decay)
    initial_loss = losses[0]
    assert losses[-1] < initial_loss, f"Fixed batch did not overfit: {losses}"
    with torch.no_grad():
        conditioned_after = model(images, t, labels)
        null_after = model(images, t, torch.full_like(labels, num_classes))
    assert (conditioned_after - null_after).abs().max() > 1e-7
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "smoke.pt"
        save_checkpoint(path, {"model": model.state_dict(), "ema": ema.state_dict(), "config": raw, "class_names": list(range(num_classes))})
        restored = load_checkpoint(path, device)
        copy_model = make_model(cfg, num_classes).to(device)
        copy_model.load_state_dict(restored["model"])
        assert torch.equal(copy_model(images, t, labels), model(images, t, labels))
        sampled, nfe = diffusion.sample(ema, labels[:2], cfg.image_size, 3, cfg.sample_steps, cfg.guidance_scale, cfg.seed)
        assert sampled.shape == images[:2].shape and torch.isfinite(sampled).all()
        sampled_again, _ = diffusion.sample(ema, labels[:2], cfg.image_size, 3, cfg.sample_steps, cfg.guidance_scale, cfg.seed)
        assert torch.equal(sampled, sampled_again)
        save_image_grid(sampled, Path(temp) / "samples.png")
        assert (Path(temp) / "samples.png").is_file()
        uncond_samples, uncond_nfe = diffusion.sample(ema, labels[:2], cfg.image_size, 3, cfg.sample_steps, 0, cfg.seed)
        null_labels = torch.full_like(labels[:2], num_classes)
        direct_samples, _ = diffusion.sample(ema, null_labels, cfg.image_size, 3, cfg.sample_steps, 1, cfg.seed)
        assert torch.equal(uncond_samples, direct_samples)
        assert uncond_nfe == cfg.sample_steps
    result = {"status": "pass", "data": data_result, "parameter_count": count_parameters(model),
              "fixed_batch_losses": losses, "sample_nfe": nfe, "unconditional_nfe": uncond_nfe,
              "peak_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              "device": str(device)}
    if args.output:
        write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
