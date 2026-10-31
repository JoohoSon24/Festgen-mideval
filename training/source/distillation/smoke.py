"""Numerical, real-batch backward, export and empirical NFE validation."""
from __future__ import annotations
import gc,json,subprocess
from pathlib import Path
import torch
from model import Model
from teacher.diffusion import Diffusion
from teacher.utils import load_checkpoint,write_json
from .core import coefficients,ddim_step,merged_target,target_velocity,velocity_loss
from .train import train
from .export import export


def main():
    root=Path('experiments/distillation_smoke');root.mkdir(parents=True,exist_ok=True)
    alpha=Diffusion(1000,.0001,.02,torch.device('cpu')).alpha_bar.double()
    t=torch.tensor([999,750,300,50]);mid=torch.tensor([500,400,100,10]);end_t=torch.tensor([0,100,-1,-1])
    a,s=coefficients(alpha,t);am,sm=coefficients(alpha,mid);an,sn=coefficients(alpha,end_t)
    x=torch.randn((4,3,64,64),dtype=torch.float64)
    x01=torch.randn_like(x).clamp(-1,1);x02=torch.randn_like(x).clamp(-1,1)
    mid_x=ddim_step(x,x01,a,s,am,sm);end_x=ddim_step(mid_x,x02,am,sm,an,sn)
    target=merged_target(x,end_x,a,s,an,sn)
    reconstructed=ddim_step(x,target,a,s,an,sn)
    inversion_error=float((reconstructed-end_x).abs().max())
    assert inversion_error<1e-10
    v=target_velocity(x,target,a,s)
    assert torch.allclose(a*x-s*v,target,atol=1e-10,rtol=1e-10)
    pred=v+torch.randn_like(v)*.01
    lhs=velocity_loss(pred,v,a,s)
    rhs=(torch.maximum(a.square()/s.square(),torch.ones_like(a))*(a*x-s*pred-target).square()).mean()
    assert torch.allclose(lhs.double(),rhs,atol=1e-8,rtol=1e-5)
    stages=[];teacher='experiments/teacher_b64_fliponly_constant_100k/final.pt'
    for name in ['guidance','n16','n8','n4','n2','n1']:
        stage=root/name
        metrics=train('configs/distillation/fixed_cfg2_progressive.json',teacher,name,str(stage),3)
        state=load_checkpoint(stage/'final.pt','cpu')
        assert state['step']==3 and state['prediction_type']=='v'
        assert torch.isfinite(torch.tensor(metrics['last_loss']))
        if name=='guidance':
            saved=state['ema']['out_conv.weight'].clone();del state
            resumed=train('configs/distillation/fixed_cfg2_progressive.json',teacher,name,str(stage),3)
            loaded=load_checkpoint(stage/'final.pt','cpu')
            assert loaded['step']==3 and torch.equal(saved,loaded['ema']['out_conv.weight'])
            del loaded,saved
        else:del state
        stages.append(metrics)
        teacher=str(stage/'final.pt');gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    models=[]
    for name,mode,nfe in [('n4','few_nfe',4),('n1','one_nfe',1)]:
        path=root/f'{mode}.ckpt';export(str(root/name/'final.pt'),str(path),mode)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model=Model.load_checkpoint(str(path),mode,device=device)
        counts=[];count=[0]
        hook=model.backbone.register_forward_hook(lambda *args:count.__setitem__(0,count[0]+1))
        labels=torch.tensor([0,13],device=device)
        output=model.sample((2,3,64,64),device=device,category=labels,seed=42)
        assert count[0]==nfe and output.shape==(2,3,64,64) and torch.isfinite(output).all()
        assert output.min()>=-1 and output.max()<=1
        count[0]=0
        second=model.sample((2,3,64,64),device=device,category=labels,seed=42)
        assert count[0]==nfe and torch.equal(output,second)
        hook.remove()
        models.append({'mode':mode,'measured_nfe':nfe,'parameters':model.count_parameters(),'deterministic':True})
        del model,output,second;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    original=subprocess.check_output(['git','show','HEAD:model.py'],text=True)
    current=Path('model.py').read_text()
    assert original.split('class ModelOneNFE')[0]==current.split('class ModelOneNFE')[0]
    report={'status':'PASS','merged_endpoint_max_error':inversion_error,'velocity_and_loss_identities':'PASS',
            'real_batch_size':64,'stage_training':stages,'completed_checkpoint_resume':'PASS',
            'submission_models':models,'base_Model_unchanged':True,
            'peak_training_vram_bytes':max(r['peak_vram_bytes'] or 0 for r in stages)}
    write_json(root/'results.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
