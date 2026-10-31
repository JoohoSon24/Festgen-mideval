"""Teacher FID reporting through unchanged FastGen directory metric functions.

This is not a one/few-NFE submission entry point. Fixed validation FID is for
reporting a frozen configuration, never for selecting hyperparameters.
"""
from __future__ import annotations

import argparse
import contextlib
from importlib.metadata import version
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from types import SimpleNamespace

from dataset import PokemonDataModule
from evaluate import compute_fid, generate_samples, prepare_reference_set
from src.utils import count_parameters
from .config import Config
from .diffusion import Diffusion
from .data import TrainImages
from .utils import checkpoint_weights, load_checkpoint, make_model, set_seed, write_json


class TeacherEvaluationAdapter(nn.Module):
    def __init__(self, backbone: nn.Module, diffusion: Diffusion, cfg: Config,
                 steps: int, scale: float, seed: int, precision: str,
                 sampler: str = "ddim", clip_denoised: bool = False):
        super().__init__()
        self.backbone = backbone
        self.diffusion = diffusion
        self.cfg = cfg
        self.steps, self.scale, self.seed, self.precision = steps, scale, seed, precision
        self.batch_index = 0
        self.sampler, self.clip_denoised = sampler, clip_denoised
        self.nfe_per_sample = None

    @torch.inference_mode()
    def sample(self, shape: tuple[int, ...], *, device="cuda", category=None, **kwargs) -> torch.Tensor:
        if category is None or tuple(shape[1:]) != (3, 64, 64):
            raise ValueError("FastGen teacher evaluation requires categories and 64x64 RGB")
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda" and self.precision == "bf16" else contextlib.nullcontext()
        with amp:
            images, nfe = self.diffusion.sample(self.backbone, category, 64, 3,
                                               self.steps, self.scale, self.seed + self.batch_index,
                                               sampler=self.sampler, clip_denoised=self.clip_denoised)
        self.batch_index += 1
        self.nfe_per_sample = nfe
        return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--reference-split", choices=("official", "development"), default="official")
    parser.add_argument("--weights", default="ema", help="model, ema, or ema:<tracked decay>")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--precision", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--clip-denoised", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--score-only", action="store_true", help="Score previously generated images with matching metadata")
    parser.add_argument("--smoke", action="store_true", help="Generate 2 samples for 2 classes; do not compute FID")
    args = parser.parse_args()
    output = Path(args.output_dir)
    generated = output / "generated"
    if args.score_only and (args.smoke or args.generate_only):
        raise ValueError("score-only cannot be combined with smoke or generate-only")
    if generated.exists() and any(generated.iterdir()) and not args.score_only:
        raise FileExistsError(f"Use a fresh output directory: {generated}")
    checkpoint = load_checkpoint(args.checkpoint, "cpu")
    raw = checkpoint["config"].copy()
    raw["channel_mult"] = tuple(raw["channel_mult"])
    raw["attention_resolutions"] = tuple(raw["attention_resolutions"])
    cfg = Config(**raw)
    if cfg.image_size != 64 or len(checkpoint["class_names"]) != 151:
        raise ValueError("Full FastGen reporting requires 64x64 and 151 classes")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = make_model(cfg, 151).to(device).eval()
    backbone.load_state_dict(checkpoint_weights(checkpoint,args.weights))
    model = TeacherEvaluationAdapter(backbone, Diffusion(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end, device),
                                     cfg, args.steps, args.guidance_scale, args.seed, args.precision,
                                     args.sampler, args.clip_denoised).eval()
    categories = torch.arange(2 if args.smoke else 151).repeat_interleave(2 if args.smoke else 20)
    metrics = {"checkpoint": args.checkpoint, "checkpoint_step": checkpoint["step"], "weights": args.weights,
               "reference_split":args.reference_split,
               "seed": args.seed, "batch_size": args.batch_size, "precision": args.precision,
               "steps": args.steps, "guidance_scale": args.guidance_scale, "sample_count": len(categories),
               "sampler": args.sampler, "clip_denoised": args.clip_denoised,
               "parameter_count": count_parameters(model, trainable_only=False), "fid": None,
               "submission_eligible": False, "smoke": args.smoke,
               "versions": {name: version(name) for name in ("torch", "clean-fid", "scipy", "numpy")}}
    del checkpoint
    if args.score_only:
        previous = json.loads((output / "metrics.json").read_text())
        for key in ("checkpoint", "checkpoint_step", "weights", "seed", "batch_size", "precision", "steps", "guidance_scale", "sample_count"):
            if metrics[key] != previous[key]:
                raise ValueError(f"Existing generation metadata differs: {key}")
        if previous.get("sampler", "ddim") != args.sampler or previous.get("clip_denoised", False) != args.clip_denoised:
            raise ValueError("Existing sampler/clipping metadata differs")
        if previous.get('reference_split','official')!=args.reference_split:
            raise ValueError('Existing reference split differs')
        expected_names = {f"{label:04d}_{i:02d}.png" for label in range(151) for i in range(20)}
        if {p.name for p in generated.glob("*.png")} != expected_names:
            raise ValueError("Generated directory must contain exactly 20 PNGs for each of 151 classes")
        metrics["nfe_per_sample"] = previous["nfe_per_sample"]
    else:
        generate_samples(model, generated, categories, args.batch_size, device)
        metrics["nfe_per_sample"] = model.nfe_per_sample
    write_json(output / "metrics.json", metrics)
    if not args.generate_only and not args.smoke:
        if args.reference_split=='development':
            if not cfg.development_reference_manifest or not cfg.train_manifest:
                raise ValueError('Checkpoint requires a declared training-only holdout')
            heldout=TrainImages(cfg.data_root,cfg.split_dir,64,151,split_file=cfg.development_reference_manifest)
            trained=TrainImages(cfg.data_root,cfg.split_dir,64,151,split_file=cfg.train_manifest)
            if set(heldout.paths)&set(trained.paths):
                raise ValueError('Development images overlap trained images')
            expected=len(heldout)
            reference=Path(args.reference_dir or output/'development_reference')
            data=SimpleNamespace(val_dataloader=lambda:DataLoader(heldout,batch_size=args.batch_size,num_workers=4))
        else:
            reference=Path(args.reference_dir or 'results/reference')
            expected=3020
            data=None
        existing = len(list(reference.glob("*.png")))
        if existing not in (0, expected):
            raise ValueError(f"Reference must have 0 or {expected} PNGs; found {existing}")
        if existing == 0:
            if data is None:
                data = PokemonDataModule(data_root=cfg.data_root, split_dir=cfg.split_dir,
                                         batch_size=args.batch_size, num_workers=4)
            prepare_reference_set(data, reference)
        metrics["reference_count"] = len(list(reference.glob("*.png")))
        if metrics["reference_count"] != expected:
            raise ValueError("Incomplete reference set")
        metrics["fid"] = compute_fid(generated, reference, device, args.batch_size)
        write_json(output / "metrics.json", metrics)
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
