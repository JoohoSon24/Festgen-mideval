"""Portable paths and FID helpers for the included NFE-4 GAN fine-tuner."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('FASTGEN_GAN_OUTPUT', HERE / 'runs' / 'gt_gan'))
DOC = OUT / 'docs'
BASE = ROOT / 'training'
DATA = Path(os.environ.get('FASTGEN_DATA_ROOT', ROOT / 'data' / 'pokemon-generation-one-22k'))
CK = Path(os.environ.get('FASTGEN_N4_INIT_CHECKPOINT', ROOT / 'checkpoints' / 'few_nfe.ckpt'))
CK_SHA = os.environ.get('FASTGEN_N4_INIT_SHA256', '')
CACHE = HERE / 'fid_cache'

def dist(a, b):
    return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0)

def radii(a):
    pieces=[]
    for i in range(0, len(a), 128):
        d=dist(a[i:i+128], a)
        d[np.arange(i, min(i+128, len(a)))[:, None] == np.arange(len(a))[None, :]] = np.inf
        pieces.append(np.partition(d, 2, axis=1)[:, 2])
    return np.concatenate(pieces)

def cover(query, cloud, r):
    return float(np.concatenate([(dist(query[i:i+128], cloud) <= r).any(1) for i in range(0, len(query), 128)]).mean())

def pr(generated, reference):
    return {'precision_k3': cover(generated, reference, radii(reference)),
            'recall_k3': cover(reference, generated, radii(generated))}
