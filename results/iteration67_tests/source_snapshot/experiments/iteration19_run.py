"""All-cut finite-box MPS existence panel and explicit small-system constructions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import numpy as np
from iteration14_run import save
from iteration10_reference import core_parameters
from iteration12_reference import gaussian_box_coefficients
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian import haar_unitary,covariance_from_interferometer
from gbslab.local_mps import local_circuit,build_local_core_layerwise
from gbslab.gaussian_mps_feasibility import representation_bounds,compress_box_once
from gbslab.mps_probe import mps_to_dense,schmidt_parameters


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    jobs=[dict(modes=m,seed=s,ensemble=e,phase='original_failure_and_haar_control') for m in (32,128) for s in (150801,150802) for e in ('local_L12','haar')]
    jobs.extend(dict(modes=m,seed=s,ensemble='local_L12',phase='new_local_confirmation') for m in (32,128) for s in (190801,190802))
    jobs.extend(dict(modes=m,seed=s,ensemble=e,phase='small_construction') for m in (4,6) for s in (190801,190802) for e in ('local_L12','haar'))
    dims=(6,8,10,12,16); ranks=(8,16,32,64,128,256,512); parameters=core_parameters(.9,.65); core=parameters['core_squeezing']
    save(out/'plan.json',dict(jobs=jobs,dimensions=dims,ranks=ranks,target=.1,physical_parameters=parameters,
        small_dimensions=[6,8],small_ranks=[8,16,32],small_box_cap=1000000,
        scope='All cuts, not only selected cuts. Existence diagnostics are float64 and do not construct large MPS. Small one-shot TT-SVD is exponential in modes. Local layerwise controls included only when gates are available.'))
    files=['experiments/iteration19_run.py','src/gbslab/gaussian_mps_feasibility.py','src/gbslab/mps_probe.py','src/gbslab/local_mps.py','src/gbslab/gaussian.py','experiments/iteration10_reference.py','experiments/iteration12_reference.py','tests/test_gaussian_mps_feasibility.py']
    for f in files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'metadata.json',dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}))
    rows=[]; inputs=[]; constructions=[]
    for i,job in enumerate(jobs):
        mark=perf_counter(); item=dict(job,index=i); print('starting',i,job,flush=True)
        try:
            m=job['modes']; rng=np.random.default_rng(job['seed'])
            if job['ensemble']=='local_L12': gates,u=local_circuit(m,12,rng)
            else: gates=None; u=haar_unitary(m,rng)
            result=representation_bounds(u,core,dims,ranks); V=covariance_from_interferometer(u,core,1.)
            np.savez_compressed(out/f'input_{i:02d}.npz',U=u,covariance=V,anomalous=result['spectra']['anomalous'],marginal_probabilities=result['marginal_probabilities'],
                marginal_tail_upper_estimates=result['marginal_tail_upper_estimates'],remainder_upper_estimates=result['remainder_upper_estimates'])
            q_errors=[]
            for c in result['spectra']['cuts']:
                cut=c['cut']; np.savez_compressed(out/f'input_{i:02d}_cut{cut:03d}.npz',**{k:v for k,v in c.items() if k!='cut'})
                if m<=6 or cut in (m//4,m//2,3*m//4):
                    expected=schmidt_parameters(V,range(cut)); q_errors.append(float(np.max(np.abs(np.sort(expected)-np.sort(c['parameters'])))))
            item['independent_symplectic_max_error']=max(q_errors)
            for row in result['rows']: rows.append(dict(job,index=i,status='complete',target_met=row['existence_trace_upper_estimate']<=.1,**row))
            item.update(status='complete',cut_count=len(result['spectra']['cuts']),seconds=perf_counter()-mark)
            if m<=6:
                for d in (6,8):
                    boxmark=perf_counter(); box=gaussian_box_coefficients(u,core,d,max_elements=1000000); box_seconds=perf_counter()-boxmark
                    np.savez_compressed(out/f'input_{i:02d}_box{d}.npz',box=box)
                    for rank in (8,16,32):
                        for method in (('direct_box_tt','local_layerwise') if gates is not None else ('direct_box_tt',)):
                            row=dict(job,index=i,dimension=d,rank=rank,method=method)
                            try:
                                buildmark=perf_counter()
                                if method=='direct_box_tt': tensors,info=compress_box_once(box,rank,max_elements=1000000)
                                else: tensors,info=build_local_core_layerwise(m,core,d,rank,gates,max_pair_elements=8000000)
                                elapsed=perf_counter()-buildmark; state=mps_to_dense(tensors); overlap=np.vdot(box.ravel(),state.ravel()); actual=math.sqrt(max(0.,1-abs(overlap)**2))
                                bound=next(r for r in result['rows'] if r['dimension']==d and r['rank']==rank)['existence_trace_upper_estimate'] if method=='direct_box_tt' else info['construction_trace_budget']
                                row.update(status='complete',state_trace_distance=actual,bound=bound,stored_complex_elements=sum(a.size for a in tensors),build_seconds=elapsed,
                                    box_seconds=box_seconds if method=='direct_box_tt' else 0.,total_seconds=elapsed+(box_seconds if method=='direct_box_tt' else 0.),info=info)
                                np.savez_compressed(out/f'input_{i:02d}_d{d}_r{rank}_{method}.npz',**{f'A{k}':a for k,a in enumerate(tensors)})
                            except Exception as e: row.update(status='failed',error=repr(e))
                            constructions.append(row); save(out/'constructions.json',constructions)
        except Exception as e:
            item.update(status='failed',error=repr(e),seconds=perf_counter()-mark)
            rows.extend(dict(job,index=i,status='failed',dimension=d,rank=rank,error=repr(e)) for d in dims for rank in ranks)
        inputs.append(item); save(out/'inputs.json',inputs); save(out/'rows.json',rows); print('finished',i,item,flush=True)
    save(out/'completion.json',dict(inputs=len(inputs),complete_inputs=sum(r['status']=='complete' for r in inputs),rows=len(rows),complete_rows=sum(r['status']=='complete' for r in rows),constructions=len(constructions),complete_constructions=sum(r['status']=='complete' for r in constructions)))

if __name__=='__main__': main()
