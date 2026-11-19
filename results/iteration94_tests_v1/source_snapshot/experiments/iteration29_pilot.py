"""Frozen pilot for displaced finite-MPS vacuum importance sampling."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbslab.displaced_vacuum import CoherentMPS,GaussianVacuumProposal,importance_summary
from gbslab.exact import ThresholdModel
from iteration23_support import save,digest,load_mps


def main():
    out=ROOT/'results/iteration29_pilot'; out.mkdir(exist_ok=False)
    sources=['experiments/iteration29_pilot.py','experiments/iteration23_support.py',*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]]
    hashes={p:digest(ROOT/p) for p in sources}
    for name in sources:
        dst=out/'source_snapshot'/name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dst)
    target='results/iteration28_panel/case_14.npz'; paths=['results/iteration25_panel_retry/run_16.state.npz','results/iteration25_panel_retry/run_19.state.npz']; dependencies={name:digest(ROOT/name) for name in [target,*paths]}
    assert dependencies[target]==json.loads((ROOT/'results/iteration28_manifest.json').read_text())['artifacts'][target]
    old=json.loads((ROOT/'results/iteration25_manifest.json').read_text())['artifacts']
    assert all(dependencies[name]==old[name] for name in paths)
    arrays=np.load(ROOT/target,allow_pickle=False); proposal=GaussianVacuumProposal(arrays['full_pairing'],arrays['full_noise_factor'])
    standard=np.random.default_rng(291101).normal(size=(256,proposal.noise_dimensions)); latent,beta=proposal.from_standard(standard)
    direct=ThresholdModel(proposal.covariance+proposal.noise_factor@proposal.noise_factor.T).vacuum_probability(tuple(range(144)))
    save(out/'plan.json',dict(source_hashes=hashes,dependencies=dependencies,samples=256,batch=128,seed=291101,reference_mixed_vacuum=proposal.mixed_vacuum,independent_reference_vacuum=direct))
    np.savez_compressed(out/'proposal.npz',standard=standard,latent=latent,beta=beta)
    rows=[]
    for path in paths:
        start=perf_counter(); print('starting',path,flush=True); raw=load_mps(ROOT/path); evaluator=CoherentMPS(raw); del raw
        prep=perf_counter()-start; logs=[]
        for first in range(0,len(beta),128): logs.append(evaluator.log_probability(beta[first:first+128])); print('batch',first,flush=True)
        logp=np.concatenate(logs); weights,logw=proposal.likelihood_ratios(logp,beta); result=importance_summary(weights,proposal.mixed_vacuum)
        row=dict(path=path,**result,normalization_seconds=prep,total_seconds=perf_counter()-start)
        np.savez_compressed(out/f'case_{len(rows)}.npz',log_probabilities=logp,weights=weights,log_weights=logw); rows.append(row); save(out/'rows.json',rows); print(json.dumps(row),flush=True)
    assert hashes=={p:digest(ROOT/p) for p in sources}; save(out/'summary.json',dict(finished=len(rows),unchanged_sources=True,dependencies=dependencies))


if __name__=='__main__': main()
