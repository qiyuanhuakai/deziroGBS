"""Discrete normal-displacement emulator against the continuous mixed Gaussian."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,time,traceback,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from iteration46_displaced_trace import enumerate_trace
from gbslab.normal_grid_certificate import normal_grid,displacement_grid_bound,sample_grid_threshold
from gbslab.gaussian_fock_bridge import gaussian_fidelity,root_interval
from gbslab.displaced_fock_interval import displaced_finite_law
from gbslab.displaced_fock_certificate import exp_minus_interval
from gbslab.sampling_audit import choice_intervals
from gbslab.interval_completion import physical_tv_interval
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration49_noise';out.mkdir(exist_ok=False);names=['research/iteration49_noise_plan.md',*[f'experiments/{n}.py' for n in ('iteration49_noise','iteration46_displaced_trace','iteration22_external','iteration33_run')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    jobs=[dict(pairing=lam,sigma=sigma,dimension=d,cells=c) for lam in (F(0),F(1,2),F(3,4)) for sigma in (F(1,4),F(1,2)) for d in (4,8) for c in (16,64,256)];save(out/'plan.json',encode(dict(jobs=jobs,source_hashes=hashes,dependencies={},numpy_version=np.__version__)));rows=[]
    for index,job in enumerate(jobs):
        start=time.perf_counter();row=dict(sequence=index,**{k:float(v) if isinstance(v,F) else v for k,v in job.items()})
        try:
            lam=job['pairing'];sigma=job['sigma'];d=job['dimension'];vector=np.array([0 if k%2 else float(lam**(k//2))*math.sqrt(math.factorial(k))/(2**(k//2)*math.factorial(k//2)) for k in range(d)],complex);vector/=np.linalg.norm(vector);tensors=[np.ascontiguousarray(vector.reshape(1,d,1))];np.savez_compressed(out/f'case_{index:02d}.npz',tensor_0=tensors[0]);bridge=gaussian_fidelity(tensors,[[lam]],[[F(0)]]);grid=normal_grid(sigma,job['cells']);p=np.array([float.fromhex(x) for x in grid['probabilities_hex']]);real,counts,cdf=choice_intervals(p);assert [float(x).hex() for x in cdf]==grid['cdf_hex'];weights=[F(c,1<<53) for c in counts];selector_TV=physical_tv_interval(weights,grid['weight_intervals']);mixed=[F(0)]*3;arithmetic=F(0);gridvac=[F(0),F(0)];normalization=root_interval(1-lam*lam);components=[]
            for node,(alpha,w,(wl,wh)) in enumerate(zip(grid['nodes'],weights,grid['weight_intervals'])):
                assert F(float(alpha))==alpha;target=displaced_finite_law(tensors,[complex(float(alpha))]);sampling=enumerate_trace(tensors,np.array([float(alpha)],complex));epsilon=physical_tv_interval(sampling['word_law'],target['intervals']+[(F(0),F(0))]);arithmetic+=w*epsilon[1]
                for k in range(3):mixed[k]+=w*sampling['word_law'][k]
                el,eh=exp_minus_interval((1-lam)*alpha*alpha);vac=(normalization[0]*el,normalization[1]*eh);gridvac[0]+=wl*vac[0];gridvac[1]+=wh*vac[1];components.append(dict(node=node,alpha=alpha,finite_target=target,sampling=sampling,arithmetic_TV=epsilon,pure_Gaussian_vacuum_interval=vac))
            p0=root_interval((1-lam*lam)/(1+2*(1-lam)*sigma*sigma));continuous=[p0,(1-p0[1],1-p0[0])];actual=physical_tv_interval(mixed,continuous+[(F(0),F(0))]);quantization=displacement_grid_bound(lam,grid);gridTV=(max(F(0),p0[0]-gridvac[1],gridvac[0]-p0[1]),max(abs(a-b) for a in gridvac for b in p0));upper=min(F(1),bridge['full_trace_interval'][1]+arithmetic+selector_TV[1]+quantization);assert actual[1]<=upper and gridTV[1]<=quantization;seed=493000+index;cal=sample_grid_threshold(tensors,grid,7,np.random.default_rng(seed));save(out/f'case_{index:02d}.json',encode(dict(job=job,stored_strides=tensors[0].strides,bridge=bridge,grid=grid,selector_counts=counts,selector_weights=weights,selector_TV=selector_TV,components=components,mixture_law=mixed,weighted_arithmetic_TV_upper=arithmetic,ideal_grid_vacuum_interval=tuple(gridvac),ideal_grid_vs_continuous_TV_interval=gridTV,continuous_target=continuous,discretization_TV_upper=quantization,total_TV_upper=upper,full_Gaussian_TV=actual,calibration=dict(seed=seed,samples=7,**cal))));row.update(status='completed',TV=[float(x) for x in actual],total_upper=float(upper),state_upper=float(bridge['full_trace_interval'][1]),arithmetic_upper=float(arithmetic),selector_upper=float(selector_TV[1]),discretization_upper=float(quantization),grid_TV_upper=float(gridTV[1]),tail_upper=float(grid['tail_interval'][1]),failure=float(mixed[-1]))
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-start;rows.append(row);save(out/'rows.json',rows);print('noise',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items());save(out/'summary.json',dict(planned=36,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True,source_hashes=hashes,dependencies={}))
if __name__=='__main__':main()
