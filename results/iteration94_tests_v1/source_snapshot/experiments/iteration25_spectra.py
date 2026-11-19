"""All physical cuts of the frozen iteration24 Gaussian candidate."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian_general_spectra import pairing_representation_bounds
from iteration23_support import digest,save


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=False)
    files=sorted({'experiments/iteration25_spectra.py','experiments/iteration23_support.py',*(f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py'))}); hashes={f:digest(ROOT/f) for f in files}
    for f in files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    source='results/iteration24_chain/case_56.npz'; sha=digest(ROOT/source); frozen=json.loads((ROOT/'results/iteration24_manifest.json').read_text())['artifacts']; assert frozen[source]==sha
    save(out/'plan.json',dict(source_hashes=hashes,dependencies={source:sha},ranks=[32,64,128,200,512,1024,2048],dimensions=[4,6,8],cuts=list(range(1,144)),source_scope='Iteration24 kernel-chain Gaussian candidate, not an identified experimental state'))
    shutil.copy2(ROOT/'research/iteration25_plan.md',out/'frozen_plan.md')
    with np.load(ROOT/source,allow_pickle=False) as data: B=data['forward']
    start=perf_counter(); result=pairing_representation_bounds(B); elapsed=perf_counter()-start
    arrays=dict(pairing=B,covariance=result['spectra']['covariance'],normal=result['spectra']['normal_matrix'],anomalous=result['spectra']['anomalous_matrix'],physical_tails=result['marginal_tail_upper_estimates'],physical_probabilities=result['marginal_probabilities'])
    cuts=[]
    for c in result['spectra']['cuts']:
        arrays[f"weights{c['cut']}"]=c['top_weights']; arrays[f"parameters{c['cut']}"]=c['parameters']; cuts.append({k:v for k,v in c.items() if k!='top_weights'})
    np.savez_compressed(out/'spectra.npz',**arrays); save(out/'cuts.json',cuts); save(out/'bounds.json',result['rows'])
    assert hashes=={f:digest(ROOT/f) for f in files}
    save(out/'summary.json',dict(cuts=len(cuts),bound_rows=len(result['rows']),independent_spectrum_max_error=max(c['independent_spectrum_error'] for c in cuts),seconds=elapsed,output_sha256=digest(out/'spectra.npz'),unchanged_sources=True))
    print(json.dumps(result['rows'],indent=2))


if __name__=='__main__': main()
