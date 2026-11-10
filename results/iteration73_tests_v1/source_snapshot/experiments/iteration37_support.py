"""Shared, explicitly traced native trajectories for iteration 37."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.wire_forest import WireForestSampler
from gbslab.exact import ThresholdModel,NumericalProbabilityError
from gbslab.positive_completion import PositiveCompletionSamplingError
from iteration22_external import save
from iteration33_run import encode


def trace_details(sampler,prefix):
    try:
        d=sampler.conditional_details(prefix);q=float(d['click_probability']);return q,dict(prefix=''.join(map(str,prefix)),q_hex=q.hex(),retained=list(d['retained_clicks']),edges=[list(e) for e in d['forest_edges']],precision=d['precision_used'],refinement=bool(d['refinement_attempted']))
    except NumericalProbabilityError as e:return None,dict(prefix=''.join(map(str,prefix)),q_hex=None,error=str(e))


def trajectories(V,job,uniforms,directory,library):
    directory.mkdir(parents=True,exist_ok=False);m=len(V)//2;n=len(uniforms);assert uniforms.shape==(n,m);order=tuple(job.get('order',range(m)));outputs=np.full((n,m),-1,dtype=np.int16);attempts=[];calls=0
    class Sequence:
        def __init__(self,values):self.values=values;self.position=0
        def random(self):
            v=float(self.values[self.position]);self.position+=1;return v
    start=perf_counter()
    with WireForestSampler(ThresholdModel(V),job['budget'],forest=job['forest'],selection=job['selection'],order=order,cache_size=0,precision='auto',library=library) as sampler:
        def observed(prefix):
            q,r=trace_details(sampler,prefix);records.append(r)
            if q is None:raise NumericalProbabilityError(r['error'])
            return q
        sampler.conditional_click_probability=observed
        for k in range(n):
            begin=perf_counter();rng=Sequence(uniforms[k]);records=[];a=dict(attempt=k)
            try:outputs[k]=sampler.sample(1,rng)[0];a.update(outcome='completed',clicks=int(outputs[k].sum()),output=outputs[k].tolist())
            except PositiveCompletionSamplingError as e:a.update(outcome='numerical_failure',failed_prefix=''.join(map(str,e.failed_prefix)),error=str(e))
            a.update(uniforms_consumed=rng.position,conditional_attempts=len(records),seconds=perf_counter()-begin);calls+=len(records);attempts.append(a);save(directory/f'attempt_{k:03d}.json',encode(dict(**a,conditionals=records)))
            if n>1 and (k+1)%8==0:print('attempts',job.get('label',job['selection']),k+1,flush=True)
        native=sampler.kernel.cache_info();diagnostics=sampler.cache_info()
    # Selector failure can happen before the native kernel is called.
    assert native['probability_calls']<=calls
    return outputs,dict(attempts=attempts,native_diagnostics=native,sampler_diagnostics=diagnostics,conditional_attempts=calls,seconds=perf_counter()-start)
