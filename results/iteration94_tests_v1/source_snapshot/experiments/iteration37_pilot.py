import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,shutil,traceback
from pathlib import Path
from time import perf_counter
import numpy as np
from iteration37_support import ROOT,trajectories
from iteration22_external import digest,save
from iteration33_run import encode


def main():
    sys.set_int_max_str_digits(0);out=ROOT/'results/iteration37_pilot';out.mkdir(exist_ok=False);names=['experiments/iteration37_pilot.py','experiments/iteration37_support.py','experiments/iteration22_external.py','experiments/iteration33_run.py',*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    deps={p:digest(ROOT/p) for p in ('results/iteration35_panel/inputs.npz','results/iteration31_build/libwire_forest.dll','results/iteration31_build/build.json')};jobs=[dict(selection=s,forest=f,budget=4,attempts=1,seed=371440) for s in ('recent','pairwise','conditional_mi','click_influence') for f in ('chain','static_mi_tree')];save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=deps,scope='Training/cost pilot only; no confidence claim or accuracy ranking from one sample'));V=np.load(ROOT/'results/iteration35_panel/inputs.npz')['shifted'];U=np.random.default_rng(371440).random((1,144));np.savez_compressed(out/'uniforms.npz',uniforms=U);rows=[]
    for i,job in enumerate(jobs):
        begin=perf_counter();row=dict(sequence=i,**job);print('starting',i,job,flush=True)
        try:
            outputs,r=trajectories(V,job,U,out/f'attempts/case_{i:02d}',ROOT/'results/iteration31_build/libwire_forest.dll');np.savez_compressed(out/f'case_{i:02d}.npz',outputs=outputs);save(out/f'case_{i:02d}.audit.json',encode(r));row.update(status='completed',outcome=r['attempts'][0]['outcome'],clicks=r['attempts'][0].get('clicks'),native_calls=r['native_diagnostics']['probability_calls'],selector_seconds=r['sampler_diagnostics']['selector_seconds'])
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-begin;rows.append(row);save(out/f'case_{i:02d}.json',row);save(out/'rows.json',rows);print('finished',i,row,flush=True)
    unchanged=hashes=={n:digest(ROOT/n) for n in names};assert unchanged;save(out/'summary.json',dict(planned=8,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=unchanged,dependencies=deps))
if __name__=='__main__':main()
