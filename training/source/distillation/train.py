"""Constant-LR, resumable fixed-guidance/progressive distillation trainer."""
from __future__ import annotations
import argparse,contextlib,copy,hashlib,json,random,shutil,time
from dataclasses import replace
from pathlib import Path
import torch
from teacher.config import Config
from teacher.data import TrainImages
from teacher.diffusion import Diffusion
from teacher.train import epoch_loader,update_ema
from teacher.utils import load_checkpoint,make_model,save_checkpoint,save_image_grid,set_seed,write_json
from src.utils import count_parameters
from .core import coefficients,ddim_step,guided_x0,merged_target,sample_velocity,target_velocity,velocity_loss,velocity_x0


def train(config_path: str, teacher_path: str, stage: str, output: str,
          steps_override: int|None=None, resume_checkpoint: str|None=None,
          data_root: str|None=None, split_dir: str|None=None, train_manifest: str|None=None) -> dict:
    settings=json.loads(Path(config_path).read_text())
    if settings['learning_rate_schedule']!='constant' or settings['warmup_steps']!=0:
        raise ValueError('User policy requires constant LR and no warmup')
    if settings['prediction_type']!='v' or settings['loss_weighting']!='truncated_snr' or settings['amp_dtype']!='bf16':
        raise ValueError('Unsupported distillation formulation')
    set_seed(settings['seed'])
    source=load_checkpoint(teacher_path,'cpu')
    cfg=Config(**source['config'])
    if data_root is not None: cfg=replace(cfg,data_root=data_root)
    if split_dir is not None: cfg=replace(cfg,split_dir=split_dir)
    if train_manifest is not None: cfg=replace(cfg,train_manifest=train_manifest)
    if cfg.diffusion_steps!=1000 or cfg.image_size!=64 or cfg.in_channels!=3 or cfg.num_classes!=151:
        raise ValueError('Distillation currently requires the inspected 1000-step RGB64/151-class teacher')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher=make_model(cfg,151).to(device).eval().requires_grad_(False)
    teacher.load_state_dict(source['ema'])
    student=copy.deepcopy(teacher).train().requires_grad_(True)
    guided=stage=='guidance'
    if guided:
        # Epsilon and v are different heads. Copy learned features but initialize
        # the v head to zero, rather than silently interpreting epsilon as v.
        torch.nn.init.zeros_(student.out_conv.weight)
        torch.nn.init.zeros_(student.out_conv.bias)
        parent_times=torch.linspace(999,0,settings['initial_steps']).round().long()
        times=parent_times
    else:
        if source.get('prediction_type')!='v': raise ValueError('Progressive teacher must predict v')
        parent_times=source['sampling_times'].cpu()
        times=parent_times[::2]
        if len(parent_times)!=2*len(times) or stage!=f'n{len(times)}':
            raise ValueError('Only exact nested factor-two distillation is supported')
    times=times.to(device);parent_times=parent_times.to(device)
    ema=copy.deepcopy(student).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(student.parameters(),lr=settings['learning_rate'],weight_decay=0)
    dataset=TrainImages(cfg.data_root,cfg.split_dir,64,151,
                        horizontal_flip_probability=cfg.horizontal_flip_probability,
                        augmentation_degrees=cfg.augmentation_degrees,
                        augmentation_translate=cfg.augmentation_translate,
                        augmentation_scale_delta=cfg.augmentation_scale_delta,split_file=cfg.train_manifest)
    digest=hashlib.sha256('\n'.join(dataset.paths).encode()).hexdigest()
    if digest!=source['train_paths_sha256']: raise ValueError('Student/teacher train manifests differ')
    source_config=source['config'];class_names=source['class_names']
    teacher_clip=guided or source.get('clip_denoised',False)
    del source
    run=Path(output);run.mkdir(parents=True,exist_ok=True)
    steps=steps_override or (settings['guidance_updates'] if guided else settings['progressive_updates'])
    step=epoch=cursor=0;images_seen=0;elapsed_before=0.0;last_loss=0.0
    previous=run/'latest.pt'
    resume_path=previous if previous.exists() else Path(resume_checkpoint) if resume_checkpoint else None
    if resume_path is not None:
        resume=load_checkpoint(resume_path,'cpu')
        old_settings=resume['settings']
        allowed_budget_changes={'progressive_updates','output_dir','milestone_steps','evaluation_steps'}
        comparable=lambda value: {key:item for key,item in value.items() if key not in allowed_budget_changes}
        same_settings=(old_settings==settings if resume_path==previous else comparable(old_settings)==comparable(settings))
        if not same_settings or resume['teacher_checkpoint']!=teacher_path or resume['stage']!=stage:
            raise ValueError('Resume config/teacher/stage differs')
        if resume['train_paths_sha256']!=digest or (resume['num_train_steps']!=steps if resume_path==previous else resume['num_train_steps']>=steps):
            raise ValueError('Resume data/budget differs')
        if resume['step']>steps or resume['step']>resume['num_train_steps']:
            raise ValueError('Resume step exceeds training budget')
        student.load_state_dict(resume['model']);ema.load_state_dict(resume['ema'])
        optimizer.load_state_dict(resume['optimizer'])
        step,epoch,cursor=resume['step'],resume['epoch'],resume['batch_in_epoch']
        images_seen=resume['images_seen'];elapsed_before=resume['elapsed_seconds'];last_loss=resume['last_loss']
        torch.set_rng_state(resume['torch_rng']);random.setstate(resume['python_rng'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(resume['cuda_rng'])
        if resume_path!=previous:
            write_json(run/'config.json',{'settings':settings,'teacher_config':source_config,'stage':stage,
                                          'teacher_checkpoint':teacher_path,'resume_checkpoint':str(resume_path),
                                          'resume_step':step,'target_step':steps})
            snapshot=run/'source_snapshot'
            for name in ('distillation','teacher'):
                shutil.copytree(name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copyfile('model.py',snapshot/'model.py')
        del resume
    else:
        write_json(run/'config.json',{'settings':settings,'teacher_config':source_config,'stage':stage,'teacher_checkpoint':teacher_path})
        snapshot=run/'source_snapshot'
        for name in ('distillation','teacher'):
            shutil.copytree(name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile('model.py',snapshot/'model.py')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.monotonic();parameter_count=count_parameters(student)
    assert parameter_count<60_000_000
    def amp(): return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
    def state():
        return {'model':student.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
                'config':source_config,'settings':settings,'class_names':class_names,'prediction_type':'v',
                'sampling_times':times.cpu(),'clip_denoised':guided,'stage':stage,
                'teacher_checkpoint':teacher_path,'train_paths_sha256':digest,
                'step':step,'epoch':epoch,'batch_in_epoch':cursor,'images_seen':images_seen,
                'num_train_steps':steps,'last_loss':last_loss,'elapsed_seconds':elapsed_before+time.monotonic()-start,
                'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
                'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
    diffusion=Diffusion(cfg.diffusion_steps,cfg.beta_start,cfg.beta_end,device)
    alpha=diffusion.alpha_bar
    while step<steps:
        loader=epoch_loader(dataset,settings['batch_size'],settings['num_workers'],settings['seed'],epoch)
        for batch_index,(images,labels) in enumerate(loader):
            if batch_index<cursor:continue
            images=images.to(device);labels=labels.to(device)
            index=torch.randint(len(times),(len(labels),),device=device)
            t=times[index];a,s=coefficients(alpha,t)
            # Match inference's pure-noise start for the highest sampling node.
            x=a*images+s*torch.randn_like(images)
            top=t==times[0]
            x[top]=torch.randn_like(x[top])
            with torch.no_grad(),amp():
                if guided:
                    target_x0=guided_x0(teacher,x,t,labels,alpha,settings['guidance_scale'])
                else:
                    middle=parent_times[2*index+1]
                    end_idx=2*index+2
                    end_t=torch.where(end_idx<len(parent_times),parent_times[end_idx.clamp_max(len(parent_times)-1)],-1)
                    am,sm=coefficients(alpha,middle);an,sn=coefficients(alpha,end_t)
                    first=velocity_x0(teacher,x,t,labels,alpha)
                    if teacher_clip:first=first.clamp(-1,1)
                    mid=ddim_step(x,first,a,s,am,sm)
                    second=velocity_x0(teacher,mid,middle,labels,alpha)
                    if teacher_clip:second=second.clamp(-1,1)
                    end=ddim_step(mid,second,am,sm,an,sn)
                    target_x0=merged_target(x,end,a,s,an,sn)
                target=target_velocity(x,target_x0,a,s)
            optimizer.zero_grad(set_to_none=True)
            with amp():
                pred=student(x,t,labels)
                loss=velocity_loss(pred,target,a,s)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite distillation loss')
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(student.parameters(),settings['grad_clip'],error_if_nonfinite=True)
            optimizer.step();update_ema(ema,student,settings['ema_decay'])
            step+=1;cursor=batch_index+1;images_seen+=len(labels);last_loss=float(loss.detach())
            peak=torch.cuda.max_memory_allocated() if device.type=='cuda' else None
            if peak is not None and peak>=20_000_000_000:raise RuntimeError('Peak allocated VRAM exceeds 20GB')
            record={'step':step,'loss':last_loss,'lr':settings['learning_rate'],'grad_norm':float(grad_norm),
                    'peak_vram_bytes':peak,'images_seen':images_seen,'elapsed_seconds':elapsed_before+time.monotonic()-start}
            with (run/'train.log').open('a') as f:f.write(json.dumps(record)+'\n')
            if step%settings['log_interval']==0:print(json.dumps(record),flush=True)
            if step%settings['save_interval']==0:save_checkpoint(run/'latest.pt',state())
            if step in settings.get('milestone_steps',[]):save_checkpoint(run/f'step_{step:06d}.pt',state())
            if step%settings['sample_interval']==0:
                labels_preview=torch.tensor([0,0,13,13,103,103,133,133],device=device)
                with amp():samples=sample_velocity(ema,alpha,times,labels_preview,settings['seed'],guided)
                save_image_grid(samples,run/f'samples_step_{step:07d}.png')
            if step>=steps:break
        if cursor==len(loader):epoch+=1;cursor=0
    result=state();save_checkpoint(run/'final.pt',result);save_checkpoint(run/'latest.pt',result)
    metrics={'step':step,'stage':stage,'nfe':len(times),'parameter_count':parameter_count,'last_loss':last_loss,
             'peak_vram_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
             'elapsed_seconds':result['elapsed_seconds'],'teacher_checkpoint':teacher_path,'prediction_type':'v',
             'lr':settings['learning_rate'],'learning_rate_schedule':'constant','training_images':len(dataset),
             'train_paths_sha256':digest,'settings':settings}
    write_json(run/'metrics.json',metrics)
    print(json.dumps(metrics),flush=True)
    return metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--teacher-checkpoint',required=True)
    p.add_argument('--stage',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--steps',type=int)
    p.add_argument('--resume-checkpoint');p.add_argument('--data-root');p.add_argument('--split-dir');p.add_argument('--train-manifest')
    a=p.parse_args();train(a.config,a.teacher_checkpoint,a.stage,a.output_dir,a.steps,a.resume_checkpoint,a.data_root,a.split_dir,a.train_manifest)
if __name__=='__main__':main()
