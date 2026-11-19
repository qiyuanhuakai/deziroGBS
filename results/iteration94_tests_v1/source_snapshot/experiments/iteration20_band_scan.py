"""All prescribed bandwidth errors and frontier allocation estimates."""
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
from gbslab.banded_gaussian_mps import pairing_distance,truncate_band,frontier_element_bound,pairing_covariance
from gbslab.local_mps import local_circuit


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    jobs=[dict(modes=m,seed=s,depth=12) for m in (32,128) for s in (150801,150802,200801,200802)]
    jobs.extend(dict(modes=32,seed=s,depth=2) for s in (200801,200802))
    save(out/'plan.json',dict(jobs=jobs,widths=list(range(25)),dimensions=[4,6,8,10],band_targets=[.1,.05],scope='Literal band deletion candidates; not optimal band approximation. Allocation formula is a conservative bound before reachable-state pruning.'))
    rows=[]; s=core_parameters(.9,.65)['core_squeezing']
    for i,job in enumerate(jobs):
        u=local_circuit(job['modes'],job['depth'],np.random.default_rng(job['seed']))[1]; B=-math.tanh(s)*(u@u.T)
        np.savez_compressed(out/f'input_{i:02d}.npz',U=u,pairing=B)
        for w in range(25):
            band=truncate_band(B,w)
            try:
                distance=pairing_distance(B,band); row=dict(job,index=i,width=w,status='complete',band_distance=distance,pairing_norm=float(np.linalg.norm(band,2)),
                    allocation_upper_elements={str(d):str(frontier_element_bound(job['modes'],d,w)) for d in (4,6,8,10)})
                if w in (0,1,2,3,4,8,12,24):
                    V=pairing_covariance(B); W=pairing_covariance(band); _,logdet=np.linalg.slogdet((V+W)/2)
                    row['independent_covariance_fidelity_error']=abs((1-distance**2)-math.exp(-.5*float(logdet)))
            except Exception as e: row=dict(job,index=i,width=w,status='failed',error=repr(e))
            rows.append(row)
        save(out/'rows.json',rows); print('finished',i,job,flush=True)
    files=['experiments/iteration20_band_scan.py','src/gbslab/banded_gaussian_mps.py','src/gbslab/local_mps.py','experiments/iteration10_reference.py']
    for f in files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'source_hashes.json',{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})

if __name__=='__main__': main()
