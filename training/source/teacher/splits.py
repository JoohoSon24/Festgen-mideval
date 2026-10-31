"""Deterministic development split drawn exclusively from official training."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from dataset import category_from_filename, read_split


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--per-class',type=int,default=5)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    root=Path(args.split_dir);out=Path(args.output_dir)
    if args.per_class<1:
        raise ValueError('per-class must be positive')
    paths=read_split(root/'train_split.txt');validation=set(read_split(root/'val_split.txt'))
    groups=defaultdict(list)
    for path in paths: groups[category_from_filename(path)].append(path)
    expected=set(json.loads((root/'category_to_id.json').read_text()))
    assert set(groups)==expected and not set(paths)&validation
    rng=random.Random(args.seed);held=set()
    for label in sorted(groups):
        candidates=sorted(groups[label]);rng.shuffle(candidates)
        if len(candidates)<=args.per_class:
            raise ValueError(f'Insufficient training images for {label}')
        held.update(candidates[:args.per_class])
    train=[path for path in paths if path not in held]
    dev=[path for path in paths if path in held]
    out.mkdir(parents=True,exist_ok=True)
    for filename,selected in [('train.txt',train),('holdout.txt',dev)]:
        content=''.join(str(p)+'\n' for p in selected)
        target=out/filename
        if target.exists() and target.read_text()!=content:
            raise FileExistsError(f'Refusing to overwrite different split: {target}')
        target.write_text(content)
    metadata={'source':'official train_split.txt only','seed':args.seed,'holdout_per_class':args.per_class,
              'train_count':len(train),'holdout_count':len(dev),'num_classes':len(groups),
              'source_train_sha256':hashlib.sha256((root/'train_split.txt').read_bytes()).hexdigest(),
              'source_validation_sha256':hashlib.sha256((root/'val_split.txt').read_bytes()).hexdigest()}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata))


if __name__=='__main__': main()
