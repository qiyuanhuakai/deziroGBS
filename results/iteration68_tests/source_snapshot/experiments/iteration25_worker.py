"""Fresh-process physical-Schmidt construction for known or frozen candidate B."""
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
from time import perf_counter
import traceback
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian import haar_unitary
from gbslab.gaussian_schmidt_mps import build_schmidt_core
from gbslab.gaussian_general_spectra import pairing_spectra
from gbslab.mps_probe import mps_to_dense,geometric_top_weights
from gbslab.banded_gaussian_mps import pairing_log_normalization
from iteration24_reference import low_sector_mps,low_sector_gaussian,partial_state_bounds
from iteration23_support import digest,save,mps_fingerprint


def physical_box(B,d):
    singular=np.linalg.svd(B,compute_uv=False); c=math.exp(.25*math.fsum(np.log1p(-singular**2)))
    @lru_cache(None)
    def amplitude(n):
        if not any(n): return complex(c)
        if sum(n)%2: return 0j
        i=next(i for i,k in enumerate(n) if k); result=0j
        for j in range(len(B)):
            count=n[j]-int(i==j)
            if count>0:
                sub=list(n); sub[i]-=1; sub[j]-=1
                result+=B[i,j]*math.sqrt(count)*amplitude(tuple(sub))
        return result/math.sqrt(n[i])
    out=np.empty((d,)*len(B),complex)
    for n in np.ndindex(out.shape): out[n]=amplitude(n)
    return out


def main():
    job=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); output=Path(sys.argv[2]).resolve(); row=dict(job); start=perf_counter(); process=psutil.Process()
    try:
        m=job['modes']; d=job['dimension']; rank=job['rank']
        if job['panel']=='quality':
            U=haar_unitary(m,np.random.default_rng(job['seed'])); B=-(U*np.tanh(np.linspace(.05,.5,m)))@U.T
            independent=pairing_spectra(B,rank); expected=[c['top_weights'] for c in independent['cuts']]
        else:
            with np.load(ROOT/'results/iteration24_chain/case_56.npz',allow_pickle=False) as data: B=data['forward']
            with np.load(ROOT/'results/iteration25_spectra/spectra.npz',allow_pickle=False) as data: expected=[data[f'weights{i}'][:rank] for i in range(1,m)]
        t=perf_counter(); tensors,info=build_schmidt_core(B,d,rank,max_seconds=job['max_seconds'],coefficient_backend='recurrence'); elapsed=perf_counter()-t
        path=output.with_suffix('.state.npz'); np.savez_compressed(path,**{f'A{i}':a for i,a in enumerate(tensors)})
        row.update(info=info,construction_seconds=elapsed,construction_trace_budget=info['construction_trace_budget'],status='accepted' if info['construction_trace_budget']<=.1 else 'budget_missed',state=dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(path),array_fingerprint=mps_fingerprint(tensors)))
        errors=[]
        for cut,reference in zip(info['cuts'],expected):
            weights=geometric_top_weights(cut['parameters'],rank); n=max(len(weights),len(reference)); errors.append(float(np.max(np.abs(np.pad(weights,(0,n-len(weights)))-np.pad(reference,(0,n-len(reference)))))))
        row['independent_spectrum_max_error']=max(errors); assert max(errors)<1e-10
        if job['panel']=='quality':
            reference=physical_box(B,d); state=mps_to_dense(tensors); overlap=abs(np.vdot(reference,state))**2; assert overlap<1+1e-10
            actual=math.sqrt(max(0.,1-overlap)); row.update(actual_original_distance=actual,actual_bound_check=actual<=info['construction_trace_budget']+1e-10)
            assert row['actual_bound_check']; np.savez_compressed(output.with_suffix('.reference.npz'),pairing=B,gaussian_box=reference,U=U)
        else:
            low,pairs=low_sector_mps(tensors); gauss=low_sector_gaussian(B,pairing_log_normalization(B),pairs); row['low_sector']=partial_state_bounds(low,gauss)
            np.savez_compressed(output.with_suffix('.low.npz'),mps_low=low,gaussian_low=gauss,pairs=pairs)
            assert row['low_sector']['pure_trace_lower_estimate']<=info['construction_trace_budget']+1e-10
        row['validation_status']='passed'
    except Exception as e: row.update(status='time_rejected' if isinstance(e,TimeoutError) else 'resource_rejected' if isinstance(e,MemoryError) else 'failed',error=repr(e),traceback=traceback.format_exc())
    row.update(worker_seconds=perf_counter()-start,peak_working_set_bytes=getattr(process.memory_info(),'peak_wset',process.memory_info().rss)); save(output,row)


if __name__=='__main__': main()
