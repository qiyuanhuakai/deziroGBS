"""Frozen full small threshold laws and seven 144-mode low-click comparisons."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbslab.low_clicks import gaussian_low_clicks,mps_low_clicks,low_click_tv
from gbslab.gaussian import haar_unitary
from gbslab.gaussian_schmidt_mps import build_schmidt_core
from gbslab.banded_gaussian_mps import pairing_covariance
from gbslab.exact import ThresholdModel
from gbslab.mps_probe import mps_to_dense
from iteration23_support import digest,save,load_mps,mps_fingerprint
from iteration22_global import load_family
from iteration22_external import ARCHIVE_SHA


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def dense_threshold(dense):
    result=np.zeros(1<<dense.ndim)
    for occ in np.ndindex(dense.shape): result[sum(1<<i for i,k in enumerate(occ) if k)]+=abs(dense[occ])**2
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    sources=sorted({'experiments/iteration26_run.py','experiments/iteration23_support.py','experiments/iteration22_global.py','experiments/iteration22_external.py',*(f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py'))}); hashes={f:digest(ROOT/f) for f in sources}
    for f in sources:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    small=[dict(panel='quality',modes=m,seed=s,dimension=d,rank=r) for m in (2,4,6) for s in (260801,260802) for d in (4,6) for r in (4,16)]
    large=[dict(panel='candidate',family=f'new_{r}',rank=r,modes=144) for r in (32,64,128,200)]+[dict(panel='candidate',family=f,rank=200,modes=144) for f in ('own_recurrence','author_pco','author_hafnian')]
    jobs=small+large
    for i,j in enumerate(jobs): j['sequence']=i
    dependencies={}; old25=read(ROOT/'results/iteration25_manifest.json')['artifacts']; old24=read(ROOT/'results/iteration24_manifest.json')['artifacts']; old22=read(ROOT/'results/iteration22_manifest.json')['artifacts']
    row_path='results/iteration25_panel_retry/rows.json'; dependencies[row_path]=digest(ROOT/row_path); assert dependencies[row_path]==old25[row_path]
    states={f"new_{r['rank']}":r['state'] for r in read(ROOT/row_path) if r['panel']=='candidate'}
    for r in states.values(): dependencies[r['path']]=digest(ROOT/r['path']); assert dependencies[r['path']]==r['sha256']==old25[r['path']]
    target='results/iteration24_chain/case_56.npz'; dependencies[target]=digest(ROOT/target); assert dependencies[target]==old24[target]
    assert digest(args.archive)==ARCHIVE_SHA
    save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=dependencies,archive_sha256=ARCHIVE_SHA,timing_scope='Single diagnostics, not repeated performance evidence'))
    shutil.copy2(ROOT/'research/iteration26_panel_plan.md',out/'frozen_plan.md')
    with np.load(ROOT/target,allow_pickle=False) as data: candidateB=data['forward']
    gaussian=gaussian_low_clicks(candidateB); np.savez_compressed(out/'gaussian.npz',pairing=candidateB,probabilities=gaussian['probabilities'],patterns=gaussian['patterns']); save(out/'gaussian.json',{k:v for k,v in gaussian.items() if k not in ('patterns','probabilities')})
    rows=[]
    for job in jobs:
        row=dict(job); i=job['sequence']; start=perf_counter(); print('starting',job,flush=True)
        try:
            if job['panel']=='quality':
                m,d,r=job['modes'],job['dimension'],job['rank']; U=haar_unitary(m,np.random.default_rng(job['seed'])); B=-(U*np.tanh(np.linspace(.05,.5,m)))@U.T
                tensors,info=build_schmidt_core(B,d,r,max_seconds=120,coefficient_backend='recurrence'); expected=gaussian_low_clicks(B); observed=mps_low_clicks(tensors)
                full_gaussian=ThresholdModel(pairing_covariance(B)).full_distribution(); full_mps=dense_threshold(mps_to_dense(tensors)); indices=[sum(1<<int(k) for k in pair if k>=0) for pair in observed['patterns']]
                row.update(low_click_tv(observed,expected),full_threshold_tv=.5*float(np.sum(abs(full_mps-full_gaussian))),gaussian_event_max_error=float(np.max(abs(expected['probabilities']-full_gaussian[indices]))),mps_event_max_error=float(np.max(abs(observed['probabilities']-full_mps[indices]))),construction_trace_budget=info['construction_trace_budget'])
                assert row['gaussian_event_max_error']<1e-11 and row['mps_event_max_error']<1e-11 and row['threshold_coarse_tv_lower_estimate']<=row['full_threshold_tv']+1e-11
                np.savez_compressed(out/f'case_{i:02d}.state.npz',**{f'A{k}':a for k,a in enumerate(tensors)})
                np.savez_compressed(out/f'case_{i:02d}.reference.npz',pairing=B,U=U,full_gaussian=full_gaussian,full_mps=full_mps)
            else:
                family=job['family']
                if family.startswith('new_'): tensors=load_mps(ROOT/states[family]['path'])
                else:
                    tensors,entries=load_family(family,ROOT/'results/iteration22_external',args.archive); save(out/(family+'_inputs.json'),entries)
                    if family=='own_recurrence':
                        for e in entries:
                            name='results/iteration22_external/'+e['name']; assert e['sha256']==old22[name]; dependencies[name]=e['sha256']
                row['input_array_fingerprint']=mps_fingerprint(tensors); observed=mps_low_clicks(tensors); expected=gaussian; row.update(low_click_tv(observed,expected))
            path=out/f'case_{i:02d}.npz'; np.savez_compressed(path,patterns=observed['patterns'],mps_probabilities=observed['probabilities'],gaussian_probabilities=expected['probabilities'])
            row.update(output_sha256=digest(path),mps_input_log_norm_squared=observed['input_log_norm_squared'],mps_remainder=observed['remainder_probability'],gaussian_remainder=expected['remainder_probability'],mps_remainder_roundoff_clipped=observed['remainder_roundoff_clipped'],gaussian_remainder_roundoff_clipped=expected['remainder_roundoff_clipped'],status='passed'); del tensors
        except Exception: row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-start; rows.append(row); save(out/f'case_{i:02d}.json',row); save(out/'rows.json',rows); print('finished',i,row['status'],row.get('threshold_coarse_tv_lower_estimate'),flush=True)
    assert hashes=={f:digest(ROOT/f) for f in sources}
    save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),passed=sum(r['status']=='passed' for r in rows),unchanged_sources=True,dependencies=dependencies))


if __name__=='__main__': main()
