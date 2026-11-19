"""Frozen conditional sampling-law audit. Does not average displacement RNG."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import mpmath as mp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.sampling_audit import enumerate_floating_law, high_precision_law


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def compare(audit,low,high):
    with mp.workdps(140):
        convert=lambda f:mp.mpf(f.numerator)/f.denominator
        ref=list(map(mp.mpf,high['law_decimal']))
        real=list(map(convert,audit['real_cdf_law']))
        grid=[mp.mpf(n)/audit['grid_denominator'] for n in audit['grid_numerators']]
        real_tv=(sum(abs(p-q) for p,q in zip(real,ref))+convert(audit['real_failure']))/2
        grid_tv=(sum(abs(p-q) for p,q in zip(grid,ref))+mp.mpf(audit['grid_failure_numerator'])/audit['grid_denominator'])/2
        convergence=max(abs(mp.mpf(x)-y) for x,y in zip(low['law_decimal'],ref))
        return {k:mp.nstr(v,40) for k,v in {'continuous_cdf_reference_tv':real_tv,
            'grid_reference_tv':grid_tv,'reference_convergence_max':convergence,
            'state_norm_defect':abs(mp.mpf(high['state_norm_decimal'])-1)}.items()}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    if np.__version__!='2.4.6': raise RuntimeError('Pinned NumPy choice implementation required')
    panel=ROOT/'results/iteration12_layerwise_quality'
    paths=[panel/f"m{m}_L4_{profile}_s{seed}_d{d}_chi{chi}.npz"
        for m in (2,4) for profile in ('lossy','bright') for seed in (120801,120802) for d,chi in ((4,4),(6,8))]
    jobs=[{'input':str(path.relative_to(ROOT)).replace('\\','/'),'variant':v}
          for path in paths for v in ('zero','recorded','stress')]
    jobs.append({'input':'analytic_vacuum_d2','variant':'rare_alpha8'})
    save(out/'plan.json',{'jobs':jobs,'digits':[80,120],'bits':53,'max_nodes':200000,
        'stress_alpha':'(2.5+1.5j)*exp(2j*pi*arange(m)/m)',
        'scope':'Fixed binary64 MPS and alpha. Fresh independent uniforms, not fixed-seed PRNG or Gaussian displacement average.'})
    files=['src/gbslab/local_mps.py','src/gbslab/mps_probe.py','src/gbslab/sampling_audit.py',
        'tests/test_sampling_audit.py','experiments/iteration13_audit.py','pyproject.toml']
    for file in files:
        target=out/'source_snapshot'/file; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/file,target)
    save(out/'metadata.json',{'numpy':np.__version__,'mpmath':mp.__version__,'python':sys.version,
        'platform':platform.platform(),'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
        'input_sha256':{str(f.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
        'primary_sources':['https://raw.githubusercontent.com/numpy/numpy/v2.4.6/numpy/random/'+f for f in ('_generator.pyx','_common.pyx','_common.pxd','_pcg64.pyx')]})
    rows=[]
    for index,job in enumerate(jobs):
        start=perf_counter(); row=dict(job,index=index)
        try:
            if job['variant']=='rare_alpha8':
                tensors=[np.array([1,0],complex).reshape(1,2,1)]; alpha=np.array([8],complex)
            else:
                with np.load(ROOT/job['input']) as data:
                    m=data['U'].shape[0]; tensors=[data[f'A{i}'] for i in range(m)]
                    alpha={'zero':np.zeros(m,complex),'recorded':data['alpha'][0],
                        'stress':(2.5+1.5j)*np.exp(2j*np.pi*np.arange(m)/m)}[job['variant']]
            np.savez_compressed(out/f'input_{index:03d}.npz',alpha=alpha,**{f'A{i}':a for i,a in enumerate(tensors)})
            audit=enumerate_floating_law(tensors,alpha)
            low=high_precision_law(tensors,alpha,digits=80)
            high=high_precision_law(tensors,alpha,digits=120)
            row.update(compare(audit,low,high),nodes=audit['nodes'],
                quantization_tv=str(audit['quantization_tv']),quantization_tv_float=float(audit['quantization_tv']),
                real_failure=str(audit['real_failure']),grid_failure_numerator=str(audit['grid_failure_numerator']),
                max_conditional_norm_residual=audit['max_conditional_norm_residual'],status='complete')
            artifact=dict(audit)
            artifact['real_cdf_law']=list(map(str,audit['real_cdf_law']))
            for k in ('real_failure','quantization_tv','grid_denominator','grid_failure_numerator'): artifact[k]=str(audit[k])
            artifact['grid_numerators']=list(map(str,audit['grid_numerators']))
            save(out/f'law_{index:03d}.json',dict(artifact,reference80=low,reference120=high))
        except Exception as e: row.update(status='failed',error=repr(e))
        row['seconds']=perf_counter()-start; rows.append(row); save(out/'rows.json',rows)
        print(index,job['variant'],row['status'],row.get('grid_reference_tv'),flush=True)
    save(out/'completion.json',{'planned':len(jobs),'finished':len(rows),'complete':sum(r['status']=='complete' for r in rows),
        'failures':sum(r['status']!='complete' for r in rows)})


if __name__=='__main__': main()
