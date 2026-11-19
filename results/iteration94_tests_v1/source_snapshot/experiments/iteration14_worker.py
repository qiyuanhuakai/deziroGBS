"""One isolated-process cost measurement; imports excluded from time, included in peak RSS."""
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.local_mps import local_circuit, build_local_core_layerwise, sample_local_threshold, LocalSamplingError
from gbslab.local_displacement import LocalPropagator, iter_local_displacements
from iteration10_reference import core_parameters


def run(job):
    process=psutil.Process(); before=process.memory_info()
    row=dict(job,baseline_rss_bytes=before.rss,baseline_peak_wset_bytes=before.peak_wset)
    start=perf_counter(); samples=[]; completed=0
    try:
        m=job['modes']; method=job['method']; count=job['samples']; batch=job['batch_size']
        gates,u=local_circuit(m,job['depth'],np.random.default_rng(job['seed']),materialize_unitary=method!='local_stream')
        propagator=LocalPropagator(m,gates) if method=='local_stream' else None
        row['circuit_seconds']=perf_counter()-start
        row['dense_unitary_bytes']=0 if u is None else u.nbytes
        row['compiled_array_bytes']=0 if propagator is None else propagator.array_bytes
        pars=core_parameters(job['r'],job['eta'])
        if job['panel']=='pipeline':
            tensors,info=build_local_core_layerwise(m,pars['core_squeezing'],job['d'],job['chi'],gates)
            row.update(construction_seconds=info['construction_seconds'],construction_trace_budget=info['construction_trace_budget'],
                stored_complex_elements=info['stored_complex_elements'])
        noise=np.random.default_rng(job['seed']+10000); measurement=np.random.default_rng(job['seed']+20000)
        variance=pars['residual_variance']; scale=.5j*math.sqrt(variance)
        def dense_batches():
            size=count if method=='dense_full' else batch
            for first in range(0,count,size):
                normals=noise.normal(size=(min(size,count-first),m))
                yield scale*(normals@u.T)
        stream=iter_local_displacements(propagator,variance,count,noise,batch_size=batch) if propagator else dense_batches()
        row['noise_seconds']=0.; row['sampling_seconds']=0.; row['alpha_abs2_sum']=0.; row['max_batch_rows']=0
        iterator=iter(stream)
        while True:
            mark=perf_counter()
            try: alpha=next(iterator)
            except StopIteration: break
            row['noise_seconds']+=perf_counter()-mark
            row['max_batch_rows']=max(row['max_batch_rows'],len(alpha))
            # Common O(batch*m) summary cost is included in end-to-end cost.
            row['alpha_abs2_sum']+=float(np.sum(np.abs(alpha)**2))
            if job['panel']=='pipeline':
                mark=perf_counter()
                try: output,diagnostic=sample_local_threshold(tensors,alpha,measurement)
                except LocalSamplingError as e:
                    samples.append(e.completed_samples); completed+=len(e.completed_samples)
                    row.update(failed_sample=completed,failed_mode=e.mode,failed_prefix=e.prefix)
                    raise
                row['sampling_seconds']+=perf_counter()-mark
                row['max_observed_conditional_norm_residual']=max(row.get('max_observed_conditional_norm_residual',0.),diagnostic['max_observed_conditional_norm_residual'])
                samples.append(output)
            completed+=len(alpha)
        row['completed_samples']=completed
        if samples:
            output=np.concatenate(samples)
            row['sample_sha256']=hashlib.sha256(output.tobytes()).hexdigest()
            row['total_clicks']=int(output.sum())
        row['status']='complete'
    except Exception as e:
        row.update(status='failed',error=repr(e),completed_samples=completed)
    row['end_to_end_seconds']=perf_counter()-start
    after=process.memory_info()
    row.update(final_rss_bytes=after.rss,peak_wset_bytes=after.peak_wset,
        peak_pagefile_bytes=after.peak_pagefile,
        memory_scope='Windows process lifetime peak working set, including imports; fresh process per case. Not allocation counts or a sampled RSS maximum.')
    return row,samples


if __name__=='__main__':
    job=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    output=Path(sys.argv[2]); row,samples=run(job)
    output.write_text(json.dumps(row,indent=2,allow_nan=False),encoding='utf-8')
    if samples: np.savez_compressed(output.with_suffix('.npz'),samples=np.concatenate(samples))
