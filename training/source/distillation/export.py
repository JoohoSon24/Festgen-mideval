"""Export a trained stage to the fixed FastGen checkpoint interface."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from model import Model,ModelOneNFE,ModelFewNFE
from teacher.diffusion import Diffusion
from teacher.utils import load_checkpoint,save_checkpoint


def export(checkpoint_path:str,output_path:str,mode:str):
    state=load_checkpoint(checkpoint_path,'cpu')
    expected=1 if mode=='one_nfe' else 4
    if len(state['sampling_times'])!=expected or state['prediction_type']!='v' or state['clip_denoised']:
        raise ValueError('Export requires the corresponding completed progressive stage')
    cfg=state['config']
    if (cfg['model_channels'],tuple(cfg['channel_mult']),cfg['num_res_blocks'],tuple(cfg['attention_resolutions']))!=(96,(1,2,3,4),2,(16,)):
        raise ValueError('Submission constructor architecture differs')
    model=(ModelOneNFE if expected==1 else ModelFewNFE)()
    model.backbone.load_state_dict(state['ema'])
    model.alpha_bar.copy_(Diffusion(cfg['diffusion_steps'],cfg['beta_start'],cfg['beta_end'],torch.device('cpu')).alpha_bar)
    model.sampling_times.copy_(state['sampling_times']);model.sampling_seed.fill_(state['settings']['seed'])
    assert model.count_parameters()<60_000_000
    save_checkpoint(Path(output_path),{'state_dict':model.state_dict(),'source_checkpoint':checkpoint_path,
                                     'nfe':expected,'parameter_count':model.count_parameters(),'fixed_guidance_scale':2})
    loaded=Model.load_checkpoint(output_path,mode,device='cpu')
    assert loaded.count_parameters()==model.count_parameters()
    print('Export/load PASS:',output_path,loaded.count_parameters(),expected,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['one_nfe','few_nfe'],required=True);a=p.parse_args();export(a.checkpoint,a.output,a.mode)
if __name__=='__main__':main()
