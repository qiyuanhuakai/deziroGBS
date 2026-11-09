"""One fresh-process construction and sampling job with post-timing audit."""
import json
import math
from pathlib import Path
import sys
from time import perf_counter
import traceback
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian import haar_unitary
from gbslab.gaussian_schmidt_mps import build_schmidt_core
from gbslab.gaussian_mps_feasibility import equal_core_spectra
from gbslab.local_mps import local_circuit,sample_local_threshold,LocalSamplingError
from gbslab.mps_probe import mps_to_dense,geometric_top_weights
from iteration10_reference import core_parameters
from iteration12_reference import gaussian_box_coefficients
from iteration23_support import digest,load_mps,mps_fingerprint,save


def main():
    job=json.loads(Path(sys.argv[1]).read_text()); output=Path(sys.argv[2]); started=perf_counter()
    known=job.pop('known_states'); row=dict(job); process=psutil.Process()
    row['baseline_rss_bytes']=process.memory_info().rss
    try:
        t0=perf_counter(); m=job['modes']; pars=core_parameters(.9,.65)
        rng=np.random.default_rng(job['seed'])
        U=haar_unitary(m,rng) if job['ensemble']=='haar' else local_circuit(m,12,rng)[1]
        B=-math.tanh(pars['core_squeezing'])*(U@U.T); t1=perf_counter()
        tensors,info=build_schmidt_core(B,job['dimension'],job['rank'],max_seconds=job['max_seconds'],coefficient_backend=job['backend'])
        t2=perf_counter(); construction_peak=getattr(process.memory_info(),'peak_wset',process.memory_info().rss)
        z=np.random.default_rng(job['seed']+10000).standard_normal((job['samples'],m))
        alpha=.5j*math.sqrt(pars['residual_variance'])*(z@U.T); t3=perf_counter()
        samples,diagnostics=sample_local_threshold(tensors,alpha,np.random.default_rng(job['seed']+20000)); t4=perf_counter()
        row.update(status='accepted' if info['construction_trace_budget']<=.1 else 'budget_missed',target_met=info['construction_trace_budget']<=.1,
            info=info,construction_trace_budget=info['construction_trace_budget'],pipeline_seconds=t4-t0,
            phase_seconds=dict(input=t1-t0,construction=t2-t1,displacement=t3-t2,sampling=t4-t3),
            construction_peak_working_set_bytes=construction_peak,pipeline_peak_working_set_bytes=getattr(process.memory_info(),'peak_wset',process.memory_info().rss),
            sampling_status='complete',completed_samples=len(samples),sampling_diagnostics=diagnostics)
        # Everything below is excluded from pipeline_seconds, including state deduplication.
        fingerprint=mps_fingerprint(tensors)
        if fingerprint in known:
            record=known[fingerprint]; path=ROOT/record['path']
            assert digest(path)==record['file_sha256'] and mps_fingerprint(load_mps(path))==fingerprint
            row['state']=dict(record,reused=True)
        else:
            path=output.with_suffix('.state.npz'); np.savez_compressed(path,**{f'A{i}':a for i,a in enumerate(tensors)})
            row['state']=dict(path=path.relative_to(ROOT).as_posix(),file_sha256=digest(path),reused=False)
        row['state']['array_fingerprint']=fingerprint
        np.savez_compressed(output.with_suffix('.input.npz'),U=U,pairing=B)
        np.savez_compressed(output.with_suffix('.samples.npz'),alpha=alpha,samples=samples)
        independent=equal_core_spectra(U,pars['core_squeezing'],job['rank']); discrepancies=[]
        for record,cut in zip(info['cuts'],independent['cuts']):
            observed=geometric_top_weights(record['parameters'],job['rank']); expected=cut['top_weights']; size=max(len(observed),len(expected))
            discrepancies.append(float(np.max(np.abs(np.pad(observed,(0,size-len(observed)))-np.pad(expected,(0,size-len(expected)))))))
        row['independent_spectrum_max_error']=max(discrepancies,default=0.)
        if job['panel']=='quality':
            reference=gaussian_box_coefficients(U,pars['core_squeezing'],job['dimension'],max_elements=1_000_000)
            actual=math.sqrt(max(0.,1-abs(np.vdot(reference,mps_to_dense(tensors)))**2))
            np.savez_compressed(output.with_suffix('.reference.npz'),gaussian_box=reference)
            row.update(actual_original_distance=actual,actual_bound_check=actual<=info['construction_trace_budget']+1e-10)
            if not row['actual_bound_check']: raise ArithmeticError('Original Gaussian distance exceeds budget')
        if row['independent_spectrum_max_error']>1e-10: raise ArithmeticError('Independent spectrum mismatch')
        row['validation_status']='passed'
    except LocalSamplingError as e:
        row.update(status='sampling_failed',sampling_status='failed',completed_samples=len(e.completed_samples),failed_mode=e.mode,failed_prefix=e.prefix,error=repr(e),traceback=traceback.format_exc())
        np.savez_compressed(output.with_suffix('.partial_samples.npz'),samples=e.completed_samples)
    except Exception as e:
        row.update(status='resource_rejected' if isinstance(e,MemoryError) else 'time_rejected' if isinstance(e,TimeoutError) else 'failed',target_met=False,error=repr(e),traceback=traceback.format_exc())
    row.update(worker_seconds=perf_counter()-started,process_peak_working_set_bytes=getattr(process.memory_info(),'peak_wset',process.memory_info().rss))
    save(output,row)


if __name__=='__main__': main()
