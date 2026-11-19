"""Frozen cost pilot for direct displaced threshold trajectories."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import sys,shutil
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbslab.displaced_sampling import DisplacedThresholdSampler,prior_displacements
from iteration23_support import save,digest,load_mps

def main():
    out=ROOT/'results/iteration30_pilot';out.mkdir(exist_ok=False)
    names=['experiments/iteration30_pilot.py','experiments/iteration23_support.py',*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]]
    hashes={p:digest(ROOT/p) for p in names}
    for p in names:
        dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
    target='results/iteration28_panel/case_14.npz';F=np.load(ROOT/target)['full_noise_factor'];rng=np.random.default_rng(301101)
    z=rng.normal(size=(256,F.shape[1]));u=rng.random((256,144));beta=prior_displacements(F,z)
    np.savez_compressed(out/'random_inputs.npz',standard=z,uniforms=u,beta=beta)
    rows=[];dependencies={target:digest(ROOT/target)}
    for rank,run in ((32,16),(200,19)):
        path=f'results/iteration25_panel_retry/run_{run}.state.npz';dependencies[path]=digest(ROOT/path);start=perf_counter()
        sampler=DisplacedThresholdSampler(load_mps(ROOT/path));prep=perf_counter()-start
        result=sampler.sample(beta,u);np.savez_compressed(out/f'rank_{rank}.npz',**result)
        row=dict(rank=rank,samples=256,preparation_seconds=prep,total_seconds=perf_counter()-start,mean_clicks=float(result['clicks'].sum(axis=1).mean()),maximum_normalization_residual=result['maximum_normalization_residual']);rows.append(row);print(row,flush=True)
        del sampler
    assert hashes=={p:digest(ROOT/p) for p in names}
    save(out/'summary.json',dict(rows=rows,source_hashes=hashes,dependencies=dependencies,unchanged_sources=True))
if __name__=='__main__':main()
