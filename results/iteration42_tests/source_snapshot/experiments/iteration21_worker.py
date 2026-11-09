"""Isolated Gaussian-basis construction, independent spectra, and sampling."""
import json
import math
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import psutil
from iteration10_reference import core_parameters
from iteration12_reference import gaussian_box_coefficients
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian_schmidt_mps import build_schmidt_core
from gbslab.gaussian_mps_feasibility import equal_core_spectra
from gbslab.gaussian import haar_unitary
from gbslab.local_mps import local_circuit,sample_local_threshold,LocalSamplingError
from gbslab.mps_probe import mps_to_dense,geometric_top_weights


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')


def main():
    job=json.loads(Path(sys.argv[1]).read_text()); out=Path(sys.argv[2]); row=dict(job); start=perf_counter(); process=psutil.Process()
    row['baseline_rss_bytes']=process.memory_info().rss
    try:
        m=job['modes']; rng=np.random.default_rng(job['seed']); pars=core_parameters(.9,.65)
        U=haar_unitary(m,rng) if job['ensemble']=='haar' else local_circuit(m,job['depth'],rng)[1]
        B=-math.tanh(pars['core_squeezing'])*(U@U.T)
        tensors,info,maps=build_schmidt_core(B,job['dimension'],job['rank'],max_seconds=job['max_seconds'],return_maps=True)
        row.update(info=info,construction_trace_budget=info['construction_trace_budget'],target_met=info['construction_trace_budget']<=job['target'],
            status='accepted' if info['construction_trace_budget']<=job['target'] else 'budget_missed',construction_peak_wset_bytes=process.memory_info().peak_wset)
        np.savez_compressed(out.with_suffix('.npz'),**{f'A{i}':a for i,a in enumerate(tensors)})
        np.savez_compressed(out.with_suffix('.maps.npz'),U=U,pairing=B,final_scalar=maps['final_scalar'],
            **{f'K{i}':K for i,K in enumerate(maps['kernels'])},**{f'basis{i}':b for i,b in enumerate(maps['bases'])},**{f'weights{i}':w for i,w in enumerate(maps['weights'])})
        independent=equal_core_spectra(U,pars['core_squeezing'],job['rank'])
        discrepancies=[]
        for record,cut in zip(info['cuts'],independent['cuts']):
            observed=geometric_top_weights(record['parameters'],job['rank']); expected=cut['top_weights']
            length=max(len(observed),len(expected)); discrepancies.append(float(np.max(np.abs(np.pad(observed,(0,length-len(observed)))-np.pad(expected,(0,length-len(expected)))))))
        row['max_independent_schmidt_weight_error']=max(discrepancies)
        if job['panel']=='quality':
            reference=gaussian_box_coefficients(U,pars['core_squeezing'],job['dimension'],max_elements=1000000)
            state=mps_to_dense(tensors); distance=math.sqrt(max(0.,1-abs(np.vdot(reference.ravel(),state.ravel()))**2))
            row.update(actual_state_distance=distance,actual_bound_check=distance<=info['construction_trace_budget']+1e-10)
            np.savez_compressed(out.with_suffix('.reference.npz'),gaussian_box=reference)
        count=job['samples']; z=np.random.default_rng(job['seed']+10000).standard_normal((count,m)); alpha=.5j*math.sqrt(pars['residual_variance'])*(z@U.T)
        samples,diag=sample_local_threshold(tensors,alpha,np.random.default_rng(job['seed']+20000))
        np.savez_compressed(out.with_suffix('.samples.npz'),alpha=alpha,samples=samples)
        row.update(sampling_status='complete',completed_samples=count,sampling_diagnostics=diag)
    except LocalSamplingError as e:
        row.update(sampling_status='failed',sampling_error=repr(e),completed_samples=len(e.completed_samples),failed_mode=e.mode,failed_prefix=e.prefix)
        np.savez_compressed(out.with_suffix('.samples.npz'),samples=e.completed_samples)
    except Exception as e:
        row.update(status='resource_rejected' if isinstance(e,MemoryError) else 'time_rejected' if isinstance(e,TimeoutError) else 'failed',target_met=False,error=repr(e))
    row.update(total_seconds=perf_counter()-start,process_peak_wset_bytes=process.memory_info().peak_wset); save(out,row)


if __name__=='__main__': main()
