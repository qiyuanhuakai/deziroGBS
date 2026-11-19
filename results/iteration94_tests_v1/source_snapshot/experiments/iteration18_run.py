"""Predeclared reference policies and degrees on old and new input panels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import numpy as np
from iteration14_run import save
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian import haar_unitary,covariance_from_interferometer
from gbslab.local_mps import local_circuit
from gbslab.exact import ThresholdModel
from gbslab.conditional_memory import ConditionalMemorySampler
from gbslab.meixner_probe import uniform_parameters
from gbslab.meixner_vacuum import vacuum_projection_prefixes
from iteration6_completion import extended_law


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    old=ROOT/'results/iteration17_panel'; old_inputs=json.loads((old/'inputs.json').read_text()); old_controls=json.loads((old/'controls.json').read_text())
    jobs=[dict(modes=r['modes'],ensemble=r['ensemble'],seed=r['seed'],profile=r['profile'],r=r['r'],eta=r['eta'],phase='reused_exploration',old_index=r['index']) for r in old_inputs]
    jobs.extend(dict(modes=m,ensemble=e,seed=s,profile=name,r=r,eta=eta,phase='new_seed_confirmation') for m in (8,12) for e in ('haar','local_L12') for s in (180801,180802) for name,r,eta in (('lossy',.6,.35),('bright',.9,.65),('stress',1.2,.85)))
    saved_degrees=sorted(set(range(9))|{12,16,24,32,48,64})
    save(out/'plan.json',dict(jobs=jobs,max_degree=64,saved_degrees=saved_degrees,reference_policies=['paper','physical_mean'],primary_degrees=[8,16,32,64],retained_k=[2,4],bound_target=.01,
        scope='Both reference policies and degrees fixed before execution. Old panel exploratory, new seeds confirmation. All-subset enumeration; not scalable sampling or same-quality cost advantage.'))
    files=['experiments/iteration18_run.py','src/gbslab/meixner_vacuum.py','src/gbslab/meixner_probe.py','src/gbslab/gaussian.py','src/gbslab/local_mps.py','src/gbslab/exact.py','src/gbslab/conditional_memory.py','experiments/iteration6_completion.py','tests/test_meixner_vacuum.py']
    for f in files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'metadata.json',dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},python=sys.version,numpy=np.__version__))
    rows=[]; inputs=[]; controls=[]
    for i,job in enumerate(jobs):
        item=dict(job,index=i); print('starting',i,job,flush=True)
        try:
            m=job['modes']; rng=np.random.default_rng(job['seed'])
            if 'old_index' in job:
                path=old/f"input_{job['old_index']:02d}.npz"
                with np.load(path) as data: u=data['U']; V=data['covariance']
                item['old_input_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                u=haar_unitary(m,rng) if job['ensemble']=='haar' else local_circuit(m,12,rng)[1]
                V=covariance_from_interferometer(u,job['r'],job['eta'])
            mark=perf_counter(); target=ThresholdModel(V,max_exact_modes=14).full_distribution(); item['target_seconds']=perf_counter()-mark
            np.savez_compressed(out/f'input_{i:02d}.npz',U=u,covariance=V,target=target)
            for policy in ('paper','physical_mean'):
                try:
                    b=uniform_parameters(m,job['r'],job['eta'])['b'] if policy=='paper' else None
                    result=vacuum_projection_prefixes(u,job['r'],job['eta'],64,reference_mean=b)
                    np.savez_compressed(out/f'input_{i:02d}_{policy}.npz',degrees=saved_degrees,raw=result['raw'][saved_degrees],repaired=result['repaired'][saved_degrees],
                        vacuum_coefficients=result['vacuum_coefficients'],exact_vacuum=result['exact_vacuum'],subset_radii=result['subset_radii'])
                    for d,metric in enumerate(result['rows']):
                        row=dict(job,index=i,policy=policy,status='complete',**metric,reference_mean=result['reference_mean'],physical_mean=result['physical_mean'],spectral_radius=result['spectral_radius'],
                            total_seconds=result['elapsed_seconds'],tv_repaired=float(.5*np.abs(result['repaired'][d]-target).sum()),half_l1_signed=float(.5*np.abs(result['raw'][d]-target).sum()))
                        if 'old_index' in job and policy=='paper' and d<=8:
                            previous=old if m==4 or d<=6 else ROOT/'results/iteration17_extension'
                            with np.load(previous/f"input_{job['old_index']:02d}_D{d}.npz") as data: row['old_wick_regression_error']=float(np.max(np.abs(result['raw'][d]-data['raw'])))
                        rows.append(row)
                    print('method',i,policy,result['elapsed_seconds'],'D8',rows[-57]['tv_repaired'],'D64',rows[-1]['tv_repaired'],flush=True)
                except Exception as e:
                    rows.extend(dict(job,index=i,policy=policy,degree=d,status='failed',error=repr(e)) for d in range(65)); print('failed',i,policy,repr(e),flush=True)
                save(out/'rows.json',rows)
            for k in (2,4):
                row=dict(job,index=i,k=k)
                try:
                    if 'old_index' in job:
                        previous=next(r for r in old_controls if r['index']==job['old_index'] and r['k']==k)
                        with np.load(old/f"input_{job['old_index']:02d}_retained{k}.npz") as data: q=data['distribution']
                        failure=previous['failure_mass']; records=previous['failure_records']; elapsed=previous['enumeration_seconds']; row['cost_source']='reused_iteration17'
                    else:
                        model=ThresholdModel(V,max_exact_modes=14,vacuum_cache_size=8192)
                        q,failure,records,elapsed=extended_law(ConditionalMemorySampler(model,k,selection='click_influence'),m); row['cost_source']='current_run'
                    row.update(status='complete',failure_mass=failure,failure_records=records,enumeration_seconds=elapsed,tv_extended=float(.5*(np.abs(q-target).sum()+failure)))
                    np.savez_compressed(out/f'input_{i:02d}_retained{k}.npz',distribution=q)
                except Exception as e: row.update(status='failed',error=repr(e))
                controls.append(row); save(out/'controls.json',controls)
            item['status']='complete'
        except Exception as e: item.update(status='input_failed',error=repr(e))
        inputs.append(item); save(out/'inputs.json',inputs); print('finished',i,item['status'],flush=True)
    save(out/'completion.json',dict(inputs=len(inputs),input_failures=sum(r['status']!='complete' for r in inputs),rows=len(rows),complete_rows=sum(r['status']=='complete' for r in rows),controls=len(controls),complete_controls=sum(r['status']=='complete' for r in controls)))

if __name__=='__main__': main()
