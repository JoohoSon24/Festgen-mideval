"""DMD-style fine-tuning for the 1-NFE or 4-NFE student (budget experiments, 2026-10-29).

Same objective as training/dmd_one_nfe/dmd_reproduce.py:
  frozen real score (fresh epsilon teacher, CFG 2) + online fake score
  + weak MSE regression to a frozen endpoint model on matched noise.
Options added: --nfe 4 (backprop through the 4-step rollout with gradient
checkpointing), --k (average the score difference over K noise/timestep draws),
--g-lr, --steps, --endpoint.

Selection uses only the 755-image development holdout (inside train_split).
The validation split is never touched here.
"""
import argparse, copy, hashlib, json, math, os, shutil, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source'))
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from torchvision.transforms.functional import to_pil_image
from cleanfid import fid, features

from model import Model
from distillation.core import coefficients, ddim_step
from teacher.config import Config
from teacher.utils import make_model
from teacher.cfg import apply_cfg

VALID_FEATURES = HERE / 'fid_cache' / 'valid_features.npy'
DEV_TRAIN = HERE / 'data' / 'development' / 'seed42_holdout5' / 'train.txt'
PARAMS = 48_883_107


def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(p, v):
    t = Path(str(p) + '.tmp'); t.write_text(json.dumps(v, indent=2) + '\n'); t.replace(p)


def load(path, nfe):
    mode = 'one_nfe' if nfe == 1 else 'few_nfe'
    m = Model.load_checkpoint(str(path), mode, device='cuda')
    assert m.count_parameters() == PARAMS
    return m.eval()


def rollout(model, x, labels, grad):
    """Deterministic DDIM rollout over model.sampling_times (1 or 4 backbone calls)."""
    times = model.sampling_times.tolist()
    for i, t in enumerate(times):
        tt = torch.full_like(labels, t)
        nt = torch.full_like(labels, times[i + 1] if i + 1 < len(times) else -1)
        a, s = coefficients(model.alpha_bar, tt); an, sn = coefficients(model.alpha_bar, nt)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if grad and len(times) > 1:
                v = checkpoint(model.backbone, x, tt, labels, use_reentrant=False)
            else:
                v = model.backbone(x, tt, labels)
        x0 = a * x - s * v.float()
        x = ddim_step(x, x0, a, s, an, sn)
    # 1-NFE: x == x0 here. Clamp like the original generator_image / sampler.
    return x.clamp(-1, 1)


class DevFID:
    def __init__(self, valid_features=VALID_FEATURES):
        v = np.load(valid_features); assert v.shape == (755, 2048)
        self.mu, self.cov = v.mean(0), np.cov(v, rowvar=False)
        self.fmap = features.build_feature_extractor('clean', device=torch.device('cuda'), use_dataparallel=False)

    def __call__(self, model, folder):
        """3,020 images, 20/class, batch 16, seed 42 + batch index, bf16 (fixed protocol)."""
        folder = Path(folder); img = folder / 'generated'; img.mkdir(parents=True, exist_ok=True)
        labels = torch.arange(151).repeat_interleave(20); calls = [0]
        h = model.backbone.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
        model.eval()
        for bi, pos in enumerate(range(0, 3020, 16)):
            y = labels[pos:pos + 16].cuda()
            x = model.sample((len(y), 3, 64, 64), device='cuda', category=y, seed=42 + bi)
            for i, im in enumerate(x):
                idx = pos + i
                to_pil_image(((im.float().cpu() + 1) / 2).clamp(0, 1)).save(img / f'{int(labels[idx]):04d}_{idx % 20:02d}.png')
        h.remove()
        nfe = len(model.sampling_times); assert calls[0] == 189 * nfe, calls
        g = fid.get_files_features(sorted(str(p) for p in img.glob('*.png')), model=self.fmap, num_workers=0,
                                   batch_size=32, device=torch.device('cuda'), mode='clean', verbose=False)
        score = float(fid.frechet_distance(g.mean(0), np.cov(g, rowvar=False), self.mu, self.cov))
        shutil.rmtree(img)  # keep disk usage low; metrics are what we need
        return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--nfe', type=int, choices=(1, 4), required=True)
    ap.add_argument('--init', required=True, help='generator start (inference ckpt)')
    ap.add_argument('--endpoint', required=True, help='frozen 4-NFE regression target')
    ap.add_argument('--real', required=True, help='scratch-trained real-score teacher checkpoint')
    ap.add_argument('--dev-train-manifest', default=str(DEV_TRAIN))
    ap.add_argument('--valid-features', default=str(VALID_FEATURES))
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--g-lr', type=float, default=2e-5)
    ap.add_argument('--fake-lr', type=float, default=2e-4)
    ap.add_argument('--reg', type=float, default=0.1)
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--cfg', type=float, default=2.0)
    ap.add_argument('--steps', type=int, default=2500)
    ap.add_argument('--evals', default='250,500,1000,1500,2000,2500')
    ap.add_argument('--patience', type=int, default=0, help='stop after N evals without a new best (0=off)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    run = Path(args.output_dir) if args.output_dir else HERE / 'runs' / args.name
    run.mkdir(parents=True, exist_ok=args.smoke)
    evals = [int(s) for s in args.evals.split(',') if s]
    status = {'name': args.name, 'args': vars(args), 'status': 'starting', 'pid': os.getpid(), 'evals': []}
    write(run / 'status.json', status)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    hashes = {k: digest(getattr(args, k)) for k in ('init', 'endpoint', 'real')}
    hashes['dev_train_manifest'] = digest(args.dev_train_manifest)
    status['input_sha256'] = hashes; write(run / 'status.json', status)

    student = load(args.init, args.nfe)
    endpoint = load(args.endpoint, 4).requires_grad_(False)
    ema = copy.deepcopy(student).eval().requires_grad_(False)
    student.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(student.backbone.parameters(), lr=args.g_lr, weight_decay=0)

    state = torch.load(args.real, map_location='cpu', weights_only=False)
    paths = Path(args.dev_train_manifest).read_text().splitlines()
    assert state['train_paths_sha256'] == hashlib.sha256('\n'.join(paths).encode()).hexdigest(), 'real teacher must use dev-train only'
    real = make_model(Config(**state['config']), 151).cuda().eval().requires_grad_(False)
    real.load_state_dict(state['ema']); del state
    fake = copy.deepcopy(real).train().requires_grad_(True)
    for m in (real, fake):
        assert sum(p.numel() for p in m.parameters()) == PARAMS
    fake_opt = torch.optim.AdamW(fake.parameters(), lr=args.fake_lr, weight_decay=0)
    alpha_bar = student.alpha_bar

    dev = None if args.smoke else DevFID(args.valid_features)
    best = {'step': 0, 'dev_fid': None}
    if dev:
        f0 = dev(ema, run / 'eval_000000'); best = {'step': 0, 'dev_fid': f0}
        status['evals'].append({'step': 0, 'dev_fid': f0}); write(run / 'status.json', status)
        print(json.dumps({'step': 0, 'dev_fid': f0}), flush=True)

    def fit_fake(images, labels):
        fake_opt.zero_grad(set_to_none=True)
        t = torch.randint(50, 951, (len(labels),), device='cuda'); a, s = coefficients(alpha_bar, t)
        n = torch.randn_like(images); noisy = a * images.detach() + s * n
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = (fake(noisy, t, labels).float() - n).square().mean()
        assert torch.isfinite(loss); loss.backward()
        torch.nn.utils.clip_grad_norm_(fake.parameters(), 1, error_if_nonfinite=True); fake_opt.step()
        return float(loss.detach())

    def gen_detached(x, labels):
        with torch.no_grad():
            return rollout(student, x, labels, grad=False)

    torch.cuda.reset_peak_memory_stats(); t0 = time.monotonic()
    for w in range(200):  # fake-score warmup on current generator samples
        labels = torch.randint(151, (16,), device='cuda'); x = torch.randn((16, 3, 64, 64), device='cuda')
        fl = fit_fake(gen_detached(x, labels), labels)
        if args.smoke and w >= 2: break
    log = (run / 'train.log').open('a'); since_best = 0
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True); tot = dmd_tot = reg_tot = 0.
        for micro in range(4):
            labels = torch.randint(151, (16,), device='cuda'); seed = 1_000_000 + step * 4 + micro
            x = torch.randn((16, 3, 64, 64), device='cuda', generator=torch.Generator(device='cuda').manual_seed(seed))
            det = gen_detached(x, labels)
            fl = fit_fake(det, labels)
            grads = []
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                for _ in range(args.k):
                    t = torch.randint(50, 951, (16,), device='cuda'); a, s = coefficients(alpha_bar, t)
                    noisy = a * det + s * torch.randn_like(x)
                    real_eps = apply_cfg(real(noisy, t, torch.full_like(labels, 151)), real(noisy, t, labels), args.cfg).float()
                    fake_eps = fake(noisy, t, labels).float()
                    rx0 = (noisy - s * real_eps) / a; fx0 = (noisy - s * fake_eps) / a
                    norm = (det - rx0).abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
                    grads.append((fx0 - rx0) / norm)
                dgrad = torch.stack(grads).mean(0)
            assert torch.isfinite(dgrad).all()
            desired = endpoint.sample((16, 3, 64, 64), device='cuda', category=labels, seed=seed).float().clone()
            images = rollout(student, x, labels, grad=True)
            dmd = .5 * (images - (images.detach() - dgrad).detach()).square().mean()
            reg = (images - desired).square().mean()
            loss = (dmd + args.reg * reg) / 4
            assert torch.isfinite(loss); loss.backward()
            tot += float(loss); dmd_tot += float(dmd) / 4; reg_tot += float(reg) / 4
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1, error_if_nonfinite=True); optimizer.step()
        with torch.no_grad():
            for e, p in zip(ema.parameters(), student.parameters(), strict=True):
                e.lerp_(p.detach(), .001)
        peak = torch.cuda.max_memory_allocated(); assert peak < 20_000_000_000
        line = {'step': step, 'loss': tot, 'dmd': dmd_tot, 'reg': reg_tot, 'fake_loss': fl, 'grad_norm': float(gn),
                'peak_vram_bytes': peak, 'elapsed_s': time.monotonic() - t0}
        log.write(json.dumps(line) + '\n'); log.flush()
        if step % 20 == 0 or args.smoke:
            print(json.dumps(line), flush=True)
            status.update(status='training', step=step, last=line); write(run / 'status.json', status)
        if args.smoke and step >= 3:
            print('SMOKE OK', json.dumps(line)); return
        if step in evals and dev:
            f = dev(ema, run / f'eval_{step:06d}')
            row = {'step': step, 'dev_fid': f, 'elapsed_s': line['elapsed_s'], 'peak_vram_bytes': peak}
            status['evals'].append(row); print(json.dumps(row), flush=True)
            if f < best['dev_fid']:
                best = row; since_best = 0
                torch.save({'state_dict': ema.state_dict(), 'step': step, 'args': vars(args), 'dev_fid': f,
                            'input_sha256': hashes, 'nfe': args.nfe}, run / 'best.ckpt.tmp')
                os.replace(run / 'best.ckpt.tmp', run / 'best.ckpt')
            else:
                since_best += 1
            status['best'] = best; write(run / 'status.json', status)
            if args.patience and since_best >= args.patience:
                status['stopped_early'] = step; break
    status.update(status='completed', best=best, total_s=time.monotonic() - t0,
                  peak_vram_bytes=torch.cuda.max_memory_allocated())
    write(run / 'status.json', status)
    print('DONE', json.dumps(best), flush=True)


if __name__ == '__main__':
    main()
