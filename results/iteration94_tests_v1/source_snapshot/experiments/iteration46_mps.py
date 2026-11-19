"""Canonical stored layout; every MPS judged against the exact shared V."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,shutil,time,traceback
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration45_mps_trace import enumerate_trace,public_calibration
from gbslab.gaussian_schmidt_mps import build_schmidt_core
from gbslab.exact_mps_law import represented_threshold_law
from gbslab.interval_completion import physical_tv_interval
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration46_mps';out.mkdir(exist_ok=False);names=['research/iteration46_panel_plan.md',*[f'experiments/{n}.py' for n in ('iteration46_mps','iteration45_mps_trace','iteration22_external','iteration33_run','verify_iteration34')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names};deps={};sources=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    for i in range(8):
        name=f'results/iteration46_inputs/case_{i:02d}.json';deps[name]=digest(ROOT/name);sources.append(decode(json.loads((ROOT/name).read_text())));name=f'results/iteration46_inputs/case_{i:02d}.npz';deps[name]=digest(ROOT/name)
    jobs=[dict(case=i,dimension=d,rank=r) for i in range(8) for d in ((4,8) if i<4 else (2,3)) for r in (2,8)];save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=deps,numpy_version=np.__version__,layout='C-contiguous before any audit or public sampling'));rows=[]
    for i,job in enumerate(jobs):
        start=time.perf_counter();row=dict(sequence=i,**job);print('start MPS',i,flush=True)
        try:
            source=sources[job['case']];n=source['job']['modes'];B=np.load(ROOT/f"results/iteration46_inputs/case_{job['case']:02d}.npz")['B'];built,info=build_schmidt_core(B,job['dimension'],job['rank'],max_seconds=120,max_elements=1000000);tensors=[np.ascontiguousarray(a) for a in built];np.savez_compressed(out/f'case_{i:02d}.npz',**{f'tensor_{j}':a for j,a in enumerate(tensors)});stored=np.load(out/f'case_{i:02d}.npz');assert all(np.array_equal(a,stored[f'tensor_{j}']) and a.strides==stored[f'tensor_{j}'].strides for j,a in enumerate(tensors));represented=represented_threshold_law(tensors,max_states=10000);sampling=enumerate_trace(tensors,max_nodes=20000);calibration=public_calibration(tensors,sampling['trace'],461000+i);target=source['target']['target_intervals']+[(F(0),F(0))];state=represented['distribution']+[F(0)];state_tv=physical_tv_interval(state,target);sampler_tv=physical_tv_interval(sampling['word_law'],target);arithmetic=sum(abs(a-b) for a,b in zip(state,sampling['word_law']))/2;quantization=sum(abs(a-b) for a,b in zip(sampling['real_law'],sampling['word_law']))/2;serial=json.loads(json.dumps(info,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else (_ for _ in ()).throw(TypeError())))
            save(out/f'case_{i:02d}.json',encode(dict(job=job,construction=serial,stored_strides=[a.strides for a in tensors],represented=represented,sampling=sampling,calibration=calibration,state_physical_TV_interval=state_tv,sampler_physical_TV_interval=sampler_tv,arithmetic_TV=arithmetic,quantization_TV=quantization)));row.update(status='completed',state_TV=[float(x) for x in state_tv],sampler_TV=[float(x) for x in sampler_tv],arithmetic_TV=float(arithmetic),failure=float(sampling['word_law'][-1]),trace_nodes=sampling['nodes'],fock_states=represented['fock_states'])
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-start;rows.append(row);save(out/'rows.json',rows);print('end MPS',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(planned=32,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
