from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import torch

from .config import Config
from .diffusion import Diffusion
from .utils import checkpoint_weights, load_checkpoint, make_model, save_image_grid, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", default="ema")
    parser.add_argument("--labels", type=int, nargs="+", required=True)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--clip-denoised", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    raw = checkpoint["config"].copy()
    raw["channel_mult"] = tuple(raw["channel_mult"])
    raw["attention_resolutions"] = tuple(raw["attention_resolutions"])
    cfg = Config(**raw)
    if any(label < 0 or label >= len(checkpoint["class_names"]) for label in args.labels):
        raise ValueError("Sample labels must be class indices from checkpoint class_names")
    set_seed(args.seed)
    model = make_model(cfg, len(checkpoint["class_names"])).to(device).eval()
    model.load_state_dict(checkpoint_weights(checkpoint,args.weights))
    diffusion = Diffusion(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end, device)
    labels = torch.tensor(args.labels, device=device)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16) if device.type == "cuda" and cfg.amp_dtype != "none" else contextlib.nullcontext()
    with amp:
        images, nfe = diffusion.sample(model, labels, cfg.image_size, cfg.in_channels, args.steps, args.guidance_scale, args.seed,
                                      sampler=args.sampler, clip_denoised=args.clip_denoised)
    save_image_grid(images, Path(args.output))
    print(json.dumps({"output": args.output, "nfe": nfe, "guidance_scale": args.guidance_scale, "seed": args.seed,
                      "sampler": args.sampler, "clip_denoised": args.clip_denoised,
                      "weights":args.weights,
                      "class_names": [checkpoint["class_names"][i] for i in args.labels]}))


if __name__ == "__main__":
    main()
