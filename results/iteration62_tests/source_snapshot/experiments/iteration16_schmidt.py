"""Matched local/Haar pure-state rank obstructions; not output-TV lower bounds."""
import argparse
import hashlib
import math
from pathlib import Path
import shutil
import sys
import numpy as np
from iteration14_run import save
from iteration10_reference import core_parameters

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.local_mps import local_circuit
from gbslab.gaussian import haar_unitary, covariance_from_interferometer
from gbslab.mps_probe import schmidt_parameters, geometric_top_weights


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    jobs=[dict(modes=m,seed=s,ensemble=e) for m in (32,128) for s in (150801,150802) for e in ('local_L12','haar')]
    save(out/'plan.json',{'jobs':jobs,'cuts':'m/4,m/2,3m/4','r':.9,'eta':.65,'ranks':[8,16,32,64,128,256,512],
        'scope':'Float64 evaluation of ideal best rank-overlap obstruction at three natural-order cuts. No interval certification, no existence guarantee when obstruction small, no threshold-output TV lower bound.'})
    rows=[]; core=core_parameters(.9,.65)['core_squeezing']
    for i,job in enumerate(jobs):
        m=job['modes']; rng=np.random.default_rng(job['seed'])
        u=local_circuit(m,12,rng)[1] if job['ensemble']=='local_L12' else haar_unitary(m,rng)
        covariance=covariance_from_interferometer(u,core,1.)
        np.savez_compressed(out/f'input_{i:02d}.npz',U=u,covariance=covariance)
        for cut in (m//4,m//2,3*m//4):
            q=schmidt_parameters(covariance,range(cut)); weights=geometric_top_weights(q,512)
            np.savez_compressed(out/f'spectrum_{i:02d}_cut{cut}.npz',parameters=q,top_weights=weights)
            for rank in (8,16,32,64,128,256,512):
                tail=max(0.,1-float(math.fsum(weights[:rank])))
                rows.append(dict(job,input_index=i,cut=cut,rank=rank,discarded_weight=tail,
                    pure_state_distance_lower_bound=math.sqrt(tail)))
        print(job,flush=True)
    save(out/'rows.json',rows)
    files=['experiments/iteration16_schmidt.py','src/gbslab/mps_probe.py','src/gbslab/gaussian.py','src/gbslab/local_mps.py']
    for f in files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'source_hashes.json',{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    lines=['# Iteration 16: pure-state Schmidt rank obstruction','',
        'Maximum across only three selected cuts. Float64; not an output-TV lower bound. Small bounds do not prove a satisfactory MPS exists.','',
        '| m | Seed | Ensemble | rank32 | rank64 | rank128 | rank512 |',
        '|---:|---:|---|---:|---:|---:|---:|']
    for job in jobs:
        selected=[r for r in rows if all(r[k]==v for k,v in job.items())]
        vals=[max(r['pure_state_distance_lower_bound'] for r in selected if r['rank']==rank) for rank in (32,64,128,512)]
        lines.append(f"| {job['modes']} | {job['seed']} | {job['ensemble']} | "+' | '.join(f'{x:.6f}' for x in vals)+' |')
    (out/'summary_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
