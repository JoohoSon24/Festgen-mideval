"""Sequential fixed-CFG baseline with train-only teacher selection and gates."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
from teacher.utils import write_json

CONFIG='configs/distillation/fixed_cfg2_progressive.json'


def command(args:list[str],log:Path) -> None:
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as file:
        process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        assert process.stdout is not None
        for line in process.stdout:
            file.write(line);file.flush();print(line,end='',flush=True)
        rc=process.wait()
    if rc:raise subprocess.CalledProcessError(rc,args)


def student_eval(stage:Path,output:Path,split='development') -> dict:
    command([sys.executable,'-u','-m','distillation.evaluate','--checkpoint',str(stage/'final.pt'),
             '--output-dir',str(output),'--reference-split',split],output/'console.log')
    return json.loads((output/'metrics.json').read_text())


def main() -> None:
    settings=json.loads(Path(CONFIG).read_text());root=Path(settings['output_dir']);root.mkdir(parents=True,exist_ok=True)
    choice=root/'teacher_selection.json'
    if not choice.exists():
        scores=[]
        for candidate in settings['teacher_candidates']:
            path=Path(candidate)/'development_evaluation/ema_ddim_clip1_s2/metrics.json'
            score=json.loads(path.read_text())
            assert score['reference_split']=='development' and score['fid'] is not None
            scores.append({'run':candidate,'fid':score['fid'],'metrics':str(path)})
        selected=min(scores,key=lambda x:x['fid'])
        write_json(choice,{'policy':'Lowest primary-EMA DDIM50 CFG2 FID on the frozen training-only holdout',
                           'candidates':scores,'selected':selected})
    teacher_run=Path(json.loads(choice.read_text())['selected']['run'])
    teacher=teacher_run/'final.pt'
    anchor=root/'teacher32_development'
    if not (anchor/'metrics.json').exists() or json.loads((anchor/'metrics.json').read_text()).get('fid') is None:
        args=[sys.executable,'-u','-m','teacher.evaluate','--checkpoint',str(teacher),'--output-dir',str(anchor),
              '--reference-split','development','--reference-dir','data/development/seed42_holdout5/reference',
              '--weights','ema','--sampler','ddim','--clip-denoised','--steps','32','--guidance-scale','2',
              '--seed','42','--batch-size','16','--precision','bf16']
        if len(list((anchor/'generated').glob('*.png')))==3020:args.append('--score-only')
        command(args,anchor/'console.log')
    baseline=json.loads((anchor/'metrics.json').read_text())['fid']
    # Frozen, matched 32-step teacher report; never used by the quality gate.
    official_anchor=root/'teacher32_official'
    if not (official_anchor/'metrics.json').exists() or json.loads((official_anchor/'metrics.json').read_text()).get('fid') is None:
        args=[sys.executable,'-u','-m','teacher.evaluate','--checkpoint',str(teacher),'--output-dir',str(official_anchor),
              '--reference-split','official','--weights','ema','--sampler','ddim','--clip-denoised','--steps','32',
              '--guidance-scale','2','--seed','42','--batch-size','16','--precision','bf16']
        if len(list((official_anchor/'generated').glob('*.png')))==3020:args.append('--score-only')
        command(args,official_anchor/'console.log')
    reports=[]
    for name in ['guidance']+[f'n{s}' for s in settings['student_steps']]:
        stage=root/name
        if not (stage/'final.pt').exists():
            command([sys.executable,'-u','-m','distillation.train','--config',CONFIG,
                     '--teacher-checkpoint',str(teacher),'--stage',name,'--output-dir',str(stage)],stage/'console.log')
        score=student_eval(stage,stage/'development_evaluation')
        if name=='guidance' and score['fid']>baseline+settings['guidance_gate_max_fid_increase']:
            write_json(root/'blocked_quality_gate.json',{'stage':name,'teacher32_fid':baseline,'student_fid':score['fid'],
                         'reason':'Guidance-transfer quality degraded beyond the predeclared tolerance; progressive stages not launched.'})
            raise RuntimeError('Guidance transfer failed quality gate; inspect before continuing')
        reports.append(score)
        if name in ('n4','n1'):
            mode='few_nfe' if name=='n4' else 'one_nfe'
            exported=Path('checkpoints')/('few_nfe.ckpt' if name=='n4' else 'one_nfe.ckpt')
            if exported.exists():
                from teacher.utils import load_checkpoint
                provenance=load_checkpoint(exported,'cpu')
                if provenance.get('source_checkpoint')!=str(stage/'final.pt'):
                    raise FileExistsError('Refusing to replace an unrelated submission checkpoint')
                del provenance
            command([sys.executable,'-u','-m','distillation.export','--checkpoint',str(stage/'final.pt'),
                     '--output',str(exported),'--mode',mode],stage/'export_console.log')
            # Evaluate the actual protected submission interface once, frozen after development.
            official=stage/'official_evaluation';generated=official/mode
            result_path=official/'metrics.json'
            if not result_path.exists():
                if generated.exists() and any(generated.iterdir()):
                    raise FileExistsError('Partial official run preserved; inspect before repeating')
                command([sys.executable,'-u','evaluate.py','--model_checkpoint',str(exported),'--evaluate_mode',mode,
                         '--reference_dir','results/reference','--output_dir',str(official),'--batch_size','16'],official/'console.log')
                # The upstream evaluator prints FID; retain a structured report alongside its exact log.
                import re
                text=(official/'console.log').read_text()
                matches=re.findall(r'FID(?: score)?\s*:\s*([0-9.eE+-]+)',text)
                if not matches:raise ValueError('Cannot parse protected evaluator FID output')
                write_json(result_path,{'fid':float(matches[-1]),'mode':mode,'checkpoint':str(exported),
                           'nfe':4 if mode=='few_nfe' else 1,'sample_count':3020,'reference_count':3020,
                           'seed':42,'parameter_count':score['parameter_count'],'source_log':str(official/'console.log')})
            reports.append(json.loads(result_path.read_text()))
        write_json(root/'comparison.json',{'teacher32_fid':baseline,'reports':reports,'teacher_selection':str(choice)})
        teacher=stage/'final.pt'
    print('Fixed-guidance and progressive distillation completed; see comparison.json',flush=True)

if __name__=='__main__':main()
