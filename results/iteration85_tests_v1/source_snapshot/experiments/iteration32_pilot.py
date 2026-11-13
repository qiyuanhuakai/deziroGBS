import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import shutil,sys
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.displaced_vacuum import CoherentMPS
from gbslab.threshold_blocks import contiguous_density,subset_effects,finite_block_law,gaussian_block_law,distance_diagnostics
from iteration23_support import load_mps,digest,save
def main():
    out=ROOT/'results/iteration32_pilot';out.mkdir(exist_ok=False);names=['experiments/iteration32_pilot.py','experiments/iteration23_support.py',*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    paths=['results/iteration25_panel_retry/run_19.state.npz','results/iteration28_panel/case_14.npz','results/iteration31_panel/large_input.npz'];dependencies={p:digest(ROOT/p) for p in paths};start=perf_counter();evaluator=CoherentMPS(load_mps(ROOT/paths[0]));prep=perf_counter()-start
    rho,dimensions=contiguous_density(evaluator,18,4);density_seconds=perf_counter()-start-prep;indices=list(range(18,22))+list(range(162,166));F=np.load(ROOT/paths[1])['full_noise_factor'][indices];V=np.load(ROOT/paths[2])['covariance'][np.ix_(indices,indices)]
    effects=subset_effects(F,dimensions);law,vacua=finite_block_law(rho,dimensions,effects);target,_=gaussian_block_law(V);np.savez_compressed(out/'block.npz',density=rho,law=law,vacua=vacua,target=target,noise=F)
    summary=dict(source_hashes=hashes,dependencies=dependencies,preparation_seconds=prep,density_seconds=density_seconds,total_seconds=perf_counter()-start,distances=distance_diagnostics(target,law),unchanged_sources=hashes=={n:digest(ROOT/n) for n in names});save(out/'summary.json',summary);print({k:v for k,v in summary.items() if k not in ('source_hashes','dependencies')});assert summary['unchanged_sources']
if __name__=='__main__':main()
