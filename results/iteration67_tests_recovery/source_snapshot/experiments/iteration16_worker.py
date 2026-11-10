"""Matched-cap restart/checkpoint comparison with all attempt ledgers retained."""
import json
import math
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import psutil
from iteration14_run import save
from iteration10_reference import core_parameters
from iteration12_reference import gaussian_box_coefficients

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.adaptive_mps import build_adaptive_core, ConstructionBudgetUnmet
from gbslab.checkpoint_mps import build_checkpoint_core
from gbslab.local_mps import local_circuit, sample_local_threshold, LocalSamplingError
from gbslab.local_displacement import LocalPropagator, iter_local_displacements
from gbslab.mps_probe import mps_to_dense


def main():
    job=json.loads(Path(sys.argv[1]).read_text()); output=Path(sys.argv[2]); row=dict(job)
    process=psutil.Process(); row['baseline_rss_bytes']=process.memory_info().rss; start=perf_counter(); attempts=[]
    def record(info):
        attempts.append(info); save(output.with_suffix('.attempts.json'),attempts)
    try:
        gates,u=local_circuit(job['modes'],job['depth'],np.random.default_rng(job['seed']),materialize_unitary=job['panel']=='quality')
        parameters=core_parameters(.9,.65); mark=perf_counter()
        kwargs=dict(dimensions=job['dimensions'],caps=job['caps'],max_pair_elements=job['max_pair_elements'],
            max_attempt_seconds=job['max_attempt_seconds'],on_attempt=record)
        if job['method']=='restart': tensors,info=build_adaptive_core(job['modes'],parameters['core_squeezing'],gates,job['target'],**kwargs)
        else:
            tensors,info=build_checkpoint_core(job['modes'],parameters['core_squeezing'],gates,job['target'],
                allocation='equal' if job['method']=='checkpoint_equal' else 'rolling',
                max_checkpoint_elements=job['max_checkpoint_elements'],**kwargs)
        row.update(search_seconds=perf_counter()-mark,dimension=info['dimension'],construction_trace_budget=info['construction_trace_budget'],
            stored_complex_elements=info['stored_complex_elements'],max_bond_actual=info['max_bond_actual'],attempts=len(attempts),
            target_met=info['construction_trace_budget']<=job['target'],construction_peak_wset_bytes=process.memory_info().peak_wset)
        np.savez_compressed(output.with_suffix('.npz'),**{f'A{i}':a for i,a in enumerate(tensors)})
        if job['panel']=='quality':
            reference=gaussian_box_coefficients(u,parameters['core_squeezing'],row['dimension'])
            actual=mps_to_dense(tensors); distance=math.sqrt(max(0,1-abs(np.vdot(reference.ravel(),actual.ravel()))**2))
            row.update(state_trace_distance=distance,state_bound_check=distance<=row['construction_trace_budget']+1e-10)
            np.savez_compressed(output.with_suffix('.reference.npz'),gaussian_box=reference,mps_tensor=actual)
        prop=LocalPropagator(job['modes'],gates)
        alpha=next(iter_local_displacements(prop,parameters['residual_variance'],32,np.random.default_rng(job['seed']+10000),batch_size=32))
        samples,diagnostics=sample_local_threshold(tensors,alpha,np.random.default_rng(job['seed']+20000))
        np.savez_compressed(output.with_suffix('.samples.npz'),alpha=alpha,samples=samples)
        row.update(status='accepted',sampling_status='complete',completed_samples=32,sampling_diagnostics=diagnostics)
    except ConstructionBudgetUnmet as e:
        row.update(status='search_unaccepted',target_met=False,attempts=len(e.attempts),search_seconds=e.elapsed)
    except LocalSamplingError as e:
        np.savez_compressed(output.with_suffix('.samples.npz'),samples=e.completed_samples)
        row.update(status='sampling_failed',error=repr(e),completed_samples=len(e.completed_samples),failed_mode=e.mode,failed_prefix=e.prefix)
    except Exception as e: row.update(status='failed',target_met=False,error=repr(e))
    row.update(total_diagnostic_seconds=perf_counter()-start,process_peak_wset_bytes=process.memory_info().peak_wset)
    save(output,row)


if __name__=='__main__': main()
