"""Student development/official reporting through protected FastGen functions."""
from __future__ import annotations
import argparse,contextlib,json
from pathlib import Path
import torch
from torch import nn
from evaluate import compute_fid,generate_samples
from src.utils import count_parameters
from teacher.config import Config
from teacher.data import TrainImages
from teacher.diffusion import Diffusion
from teacher.utils import load_checkpoint,make_model,set_seed,write_json
from .core import sample_velocity


class Adapter(nn.Module):
    def __init__(self,backbone,alpha,times,seed,clip):
        super().__init__();self.backbone=backbone
        self.register_buffer('alpha_bar',alpha);self.register_buffer('sampling_times',times)
        self.seed=seed;self.batch_index=0;self.clip=clip
    @torch.inference_mode()
    def sample(self,shape,*,device='cuda',category=None,**kwargs):
        if tuple(shape[1:])!=(3,64,64) or category is None:raise ValueError('Invalid sample request')
        amp=torch.autocast('cuda',dtype=torch.bfloat16) if torch.device(device).type=='cuda' else contextlib.nullcontext()
        with amp:
            result=sample_velocity(self.backbone,self.alpha_bar,self.sampling_times,category,self.seed+self.batch_index,self.clip)
        self.batch_index+=1
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--reference-split',choices=['development','official'],default='development')
    p.add_argument('--reference-dir');p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();output=Path(a.output_dir);metadata=output/'metrics.json';generated=output/'generated'
    if metadata.exists() and json.loads(metadata.read_text()).get('fid') is not None:
        old=json.loads(metadata.read_text())
        if old['checkpoint']!=a.checkpoint or old['seed']!=a.seed or old['reference_split']!=a.reference_split:
            raise ValueError('Completed evaluation metadata differs')
        print(old,flush=True);return
    state=load_checkpoint(a.checkpoint,'cpu');cfg=Config(**state['config'])
    if state.get('prediction_type')!='v':raise ValueError('Student must predict v')
    set_seed(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network=make_model(cfg,151).to(device).eval();network.load_state_dict(state['ema'])
    times=state['sampling_times'].to(device)
    alpha=Diffusion(cfg.diffusion_steps,cfg.beta_start,cfg.beta_end,device).alpha_bar
    model=Adapter(network,alpha,times,a.seed,state['clip_denoised']).to(device).eval()
    batch=16
    metrics={'checkpoint':a.checkpoint,'checkpoint_step':state['step'],'weights':'ema',
             'reference_split':a.reference_split,'seed':a.seed,'batch_size':batch,'precision':'bf16',
             'sample_count':3020,'nfe_per_sample':len(times),'parameter_count':count_parameters(model,trainable_only=False),
             'sampling_times':times.tolist(),'prediction_type':'v','fixed_guidance_scale':2.0,'fid':None}
    del state
    if generated.exists() and any(generated.iterdir()):
        if not metadata.exists():raise ValueError('Existing generation without metadata')
        old=json.loads(metadata.read_text())
        for key in metrics:
            if key!='fid' and old.get(key)!=metrics[key]:raise ValueError(f'Existing generation differs: {key}')
        expected={f'{c:04d}_{i:02d}.png' for c in range(151) for i in range(20)}
        if {q.name for q in generated.glob('*.png')}!=expected:raise ValueError('Incomplete generation; preserve artifacts and restart in fresh directory')
    else:
        write_json(metadata,metrics)
        generate_samples(model,generated,torch.arange(151).repeat_interleave(20),batch,device)
    if a.reference_split=='development':
        holdout=TrainImages(cfg.data_root,cfg.split_dir,64,151,split_file=cfg.development_reference_manifest)
        train=TrainImages(cfg.data_root,cfg.split_dir,64,151,split_file=cfg.train_manifest)
        assert not set(holdout.paths)&set(train.paths)
        reference=Path(a.reference_dir or 'data/development/seed42_holdout5/reference')
        count=len(holdout)
    else:
        reference=Path(a.reference_dir or 'results/reference');count=3020
    if len(list(reference.glob('*.png')))!=count:raise ValueError('Reference set missing/incomplete')
    metrics['reference_count']=count;metrics['fid']=compute_fid(generated,reference,device,batch)
    write_json(metadata,metrics);print(metrics,flush=True)
if __name__=='__main__':main()
