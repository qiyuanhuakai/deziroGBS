"""Predeclared two-mode entangled rank-one complex displacement experiments."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,time,traceback,gzip,hashlib
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from iteration46_displaced_trace import enumerate_trace
from gbslab.gaussian_fock_bridge import gaussian_fidelity
from gbslab.extended_displaced_fock import displaced_finite_law
from gbslab.growing_normal_grid import growing_grid,scaled_exp_minus
from gbslab.rank_one_noise import integrated_finite_law,gaussian_rank_one,coupling_bound,interval_law
from gbslab.sampling_audit import choice_intervals
from gbslab.interval_completion import physical_tv_interval
from gbslab.local_mps import sample_local_threshold
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def jobs():
    families=[('vacuum',F(0),(1,0),[(1,0),(1,0)]),('common',F(1,2),(1,0),[(1,0),(1,0)]),('opposite',F(1,2),(1,0),[(1,0),(-1,0)]),('quadrature',F(1,2),(1,0),[(1,0),(0,1)]),('strong',F(3,4),(1,0),[(1,0),(1,0)])]
    out=[dict(family=name,pairing=lam,phase=phase,direction=direction,sigma=F(1,4),dimension=d,cells=N,holdout=False) for name,lam,phase,direction in families for d in (4,6) for N in (16,64)]
    return out+[dict(family='complex_holdout',pairing=F(5,8),phase=(0,1),direction=[(1,0),(F(1,2),F(1,2))],sigma=F(3,8),dimension=d,cells=N,holdout=True) for d in (4,6) for N in (16,64,256)]

def interval_tv(a,b):return (sum(max(F(0),l-y,x-h) for (l,h),(x,y) in zip(a,b))/2,sum(max(abs(l-y),abs(h-x)) for (l,h),(x,y) in zip(a,b))/2)

def main():
    out=ROOT/'results/iteration54_noise';out.mkdir(exist_ok=False);names=['research/iteration54_noise_plan.md',*[f'experiments/{n}.py' for n in ('iteration54_noise','iteration46_displaced_trace','iteration22_external','iteration33_run')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    planned=jobs();save(out/'plan.json',encode(dict(jobs=planned,source_hashes=hashes,dependencies={},numpy_version=np.__version__)));rows=[]
    for index,job in enumerate(planned):
        started=time.perf_counter();row=dict(case=index,family=job['family'],dimension=job['dimension'],cells=job['cells'])
        try:
            d=job['dimension'];lam=job['pairing'];phase=complex(*job['phase']);direction=job['direction'];sigma=job['sigma'];br=[[F(0),lam*job['phase'][0]],[lam*job['phase'][0],F(0)]];bi=[[F(0),lam*job['phase'][1]],[lam*job['phase'][1],F(0)]];a=np.zeros((1,d,d),complex);b=np.zeros((d,d,1),complex);amps=np.array([(float(lam)*phase)**k for k in range(d)]);amps/=np.linalg.norm(amps)
            for k in range(d):a[0,k,k]=amps[k];b[k,k,0]=1
            tensors=[a,b];np.savez_compressed(out/f'case_{index:02d}.npz',tensor_0=a,tensor_1=b);bridge=gaussian_fidelity(tensors,br,bi);target=gaussian_rank_one(br,bi,direction,sigma);reference=integrated_finite_law(tensors,direction,sigma);grid=growing_grid(sigma,job['cells']);prob=np.array([float.fromhex(x) for x in grid['probabilities_hex']]);_,counts,cdf=choice_intervals(prob);assert grid['cdf_hex']==[float(x).hex() for x in cdf];weights=[F(c,1<<53) for c in counts];selector=physical_tv_interval(weights,grid['weight_intervals']);mixture=[F(0)]*5;gridvac=[[F(0),F(0)] for _ in range(4)];arithmetic=F(0);components=[];preparation=time.perf_counter()-started
            for node,(x,w,(wl,wh)) in enumerate(zip(grid['nodes'],weights,grid['weight_intervals'])):
                alpha=np.array([complex(float(u*x),float(v*x)) for u,v in direction]);assert all((F(float(z.real)),F(float(z.imag)))==(u*x,v*x) for z,(u,v) in zip(alpha,direction));finite=displaced_finite_law(tensors,alpha);sampling=enumerate_trace(tensors,alpha);epsilon=physical_tv_interval(sampling['word_law'],finite['intervals']+[(F(0),F(0))]);arithmetic+=w*epsilon[1]
                for k in range(5):mixture[k]+=w*sampling['word_law'][k]
                gv=[]
                for mask,r in enumerate(target['records']):
                    e=scaled_exp_minus(r['displacement_exponent_coefficient']*x*x);p=r['pure_factor_interval'];value=(p[0]*e[0],p[1]*e[1]);gv.append(value);gridvac[mask][0]+=wl*value[0];gridvac[mask][1]+=wh*value[1]
                components.append(dict(node=node,x=x,finite_target=finite,sampling=sampling,arithmetic_TV=epsilon,pure_Gaussian_vacuum_intervals=gv))
            # Empty subset is exactly one, avoiding probability-width dependence.
            gridvac[0]=[F(1),F(1)];_,ideal_grid=interval_law(gridvac);grid_error=interval_tv(ideal_grid,target['intervals']);actual=physical_tv_interval(mixture,target['intervals']+[(F(0),F(0))]);finite_error=physical_tv_interval(mixture,reference['intervals']+[(F(0),F(0))]);reference_error=interval_tv(reference['intervals'],target['intervals']);discretization=coupling_bound(target,grid);bound=min(F(1),bridge['full_trace_interval'][1]+selector[1]+arithmetic+discretization);assert actual[1]<=bound and grid_error[1]<=discretization
            seed=540000+index;rng=np.random.default_rng(seed);selected=[];outputs=[]
            for _ in range(7):
                j=int(rng.choice(len(prob),p=prob));x=grid['nodes'][j];alpha=np.array([[complex(float(u*x),float(v*x)) for u,v in direction]]);value,_=sample_local_threshold(tensors,alpha,rng);selected.append(j);outputs.append(value[0].tolist())
            computation=time.perf_counter()-started-preparation;raw=(json.dumps(encode(components),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode();compressed=gzip.compress(raw,compresslevel=6,mtime=0);name=f'case_{index:02d}.components.json.gz';(out/name).write_bytes(compressed);storage=dict(path=name,raw_bytes=len(raw),compressed_bytes=len(compressed),raw_sha256=hashlib.sha256(raw).hexdigest(),nodes=len(components),trace_nodes=sum(r['sampling']['nodes'] for r in components));save(out/f'case_{index:02d}.json',encode(dict(job=job,stored_strides=[t.strides for t in tensors],pairing_real=br,pairing_imag=bi,bridge=bridge,target=target,finite_continuous=reference,grid=grid,selector_counts=counts,selector_weights=weights,selector_TV=selector,mixture_law=mixture,weighted_arithmetic_TV_upper=arithmetic,ideal_grid_vacuum_intervals=gridvac,ideal_grid_law=ideal_grid,ideal_grid_TV=grid_error,full_Gaussian_TV=actual,program_vs_finite_continuous_TV=finite_error,finite_continuous_vs_Gaussian_TV=reference_error,discretization_TV_upper=discretization,total_TV_upper=bound,calibration=dict(seed=seed,samples=7,selected_nodes=selected,outputs=outputs),storage=storage)));row.update(status='completed',TV=[float(v) for v in actual],finite_TV=[float(v) for v in finite_error],reference_TV=[float(v) for v in reference_error],arithmetic=float(arithmetic),total_upper=float(bound),generator_variance=float(target['generator_variance']),storage=storage,timing=dict(preparation=preparation,components_and_comparisons=computation,storage=time.perf_counter()-started-preparation-computation))
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-started;rows.append(row);save(out/'rows.json',rows);print('rank-one noise',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items());save(out/'summary.json',dict(planned=26,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True,source_hashes=hashes,dependencies={}))
if __name__=='__main__':main()
