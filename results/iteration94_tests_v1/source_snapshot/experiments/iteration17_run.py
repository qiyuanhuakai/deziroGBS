"""New-seed local/Haar full-law comparison; higher-degree projection and retained baselines."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import numpy as np
from iteration14_run import save

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.meixner_high_order import projection_prefixes
from gbslab.meixner_probe import threshold_projection
from gbslab.gaussian import haar_unitary, covariance_from_interferometer
from gbslab.local_mps import local_circuit
from gbslab.exact import ThresholdModel
from gbslab.conditional_memory import ConditionalMemorySampler
from iteration6_completion import extended_law


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    jobs=[dict(modes=m,ensemble=e,seed=s,profile=name,r=r,eta=eta,max_degree=8 if m==4 else 6)
        for m in (4,8,12) for e in ('haar','local_L12') for s in (170801,170802)
        for name,r,eta in (('lossy',.6,.35),('bright',.9,.65),('stress',1.2,.85))]
    save(out/'plan.json',{'jobs':jobs,'inputs':len(jobs),'retained_k':[2,4],'selector':'click_influence',
        'max_coefficients':100000,'max_wick_states':1000000,
        'scope':'Explicit coefficient and full threshold enumeration, not implicit paper algorithm or native-speed benchmark. Signed mass, repair and retained failure symbol kept. New seeds, all choices fixed before results.'})
    files=['experiments/iteration17_run.py','src/gbslab/meixner_high_order.py','src/gbslab/meixner_probe.py',
        'src/gbslab/gaussian.py','src/gbslab/local_mps.py','src/gbslab/exact.py','src/gbslab/conditional_memory.py',
        'src/gbslab/zero_memory.py','src/gbslab/schur_memory.py','experiments/iteration6_completion.py','tests/test_meixner_high_order.py']
    for f in files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'metadata.json',{'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
        'primary_source':'https://arxiv.org/html/2611.08444v1','equations':['G.12','G.15','G.38']})
    rows=[]; controls=[]; input_rows=[]
    for index,job in enumerate(jobs):
        label=f'input_{index:02d}'; input_row=dict(job,index=index); print('starting',index,job,flush=True)
        try:
            m=job['modes']; rng=np.random.default_rng(job['seed'])
            u=haar_unitary(m,rng) if job['ensemble']=='haar' else local_circuit(m,12,rng)[1]
            V=covariance_from_interferometer(u,job['r'],job['eta'])
            mark=perf_counter(); model=ThresholdModel(V,max_exact_modes=14)
            target=model.full_distribution(); input_row['exact_reference_seconds']=perf_counter()-mark
            bits=(np.arange(1<<m)[:,None]>>np.arange(m))&1; mean=job['eta']*math.sinh(job['r'])**2
            thermal=np.prod(np.where(bits,mean/(1+mean),1/(1+mean)),axis=1)
            input_row['thermal_tv']=float(.5*np.abs(thermal-target).sum())
            np.savez_compressed(out/(label+'.npz'),U=u,covariance=V,target=target,thermal=thermal)
            try:
                result=projection_prefixes(u,job['r'],job['eta'],job['max_degree'],max_coefficients=100000,max_states=1000000)
                np.savez_compressed(out/(label+'_coefficients.npz'),indices=result['indices'],coefficients=result['coefficients'])
                for d,(metric,law) in enumerate(zip(result['rows'],result['laws'])):
                    row=dict(job,index=index,**metric,status='complete',max_imaginary_moment=result['max_imaginary_moment'],
                        tv_repaired=float(.5*np.abs(law['distribution']-target).sum()),
                        half_l1_signed=float(.5*np.abs(law['raw']-target).sum()),thermal_tv=input_row['thermal_tv'])
                    if d<=2: row['degree_two_regression_error']=float(np.max(np.abs(law['raw']-threshold_projection(u,job['r'],job['eta'],d)['raw'])))
                    rows.append(row)
                    np.savez_compressed(out/(label+f'_D{d}.npz'),raw=law['raw'],repaired=law['distribution'],subset_coefficients=law['subset_coefficients'])
                input_row.update(projection_status='complete',projection_seconds=result['elapsed_seconds'],wick_states=result['wick_states'])
            except Exception as e:
                input_row.update(projection_status='failed',projection_error=repr(e))
                rows.extend(dict(job,index=index,degree=d,status='failed',error=repr(e)) for d in range(job['max_degree']+1))
            for k in (2,4):
                row=dict(job,index=index,k=k)
                try:
                    retained_model=ThresholdModel(V,max_exact_modes=14,vacuum_cache_size=8192)
                    sampler=ConditionalMemorySampler(retained_model,k,selection='click_influence')
                    q,failure,records,elapsed=extended_law(sampler,m)
                    row.update(status='complete',failure_mass=failure,failure_records=records,enumeration_seconds=elapsed,
                        tv_extended=float(.5*(np.abs(q-target).sum()+failure)))
                    np.savez_compressed(out/(label+f'_retained{k}.npz'),distribution=q)
                except Exception as e: row.update(status='failed',error=repr(e))
                controls.append(row); save(out/'controls.json',controls)
            input_row['status']='complete'
        except Exception as e: input_row.update(status='input_failed',error=repr(e))
        input_rows.append(input_row); save(out/'inputs.json',input_rows); save(out/'rows.json',rows)
        print('finished',index,input_row['status'],input_row.get('projection_status'),input_row.get('projection_seconds'),flush=True)
    save(out/'completion.json',{'planned_inputs':len(jobs),'inputs_finished':len(input_rows),
        'input_failures':sum(r['status']!='complete' for r in input_rows),
        'projection_rows':len(rows),'projection_complete':sum(r['status']=='complete' for r in rows),
        'control_rows':len(controls),'control_complete':sum(r['status']=='complete' for r in controls)})


if __name__=='__main__': main()
