"""Independent 2M-record cohorts for four PNR models at two fixed inputs.

Predeclared eight laws, sixteen one-sided bounds with total failure allocation
0.01, ideal iid interpretation. Fixed-event lower bounds use earlier pilots;
upper bounds use empirical full-simplex TV plus radius plus target tail.
Neither arithmetic/finite-PRNG error nor arbitrary inputs are certified.
"""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,math,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_polynomial_pnr import NativePolynomialPNRSampler
from gbslab.native_pnr_newton import NativeNewtonPNRSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
runtime=[Path(shutil.which('g++')).parent];polynomial=ROOT/'results/iteration73_polynomial_pilot_v2/polynomial.dll';finite=ROOT/'results/iteration72_newton_pilot_v1/newton.dll';N=2000000;BATCH=100000;radius=math.sqrt(math.log(16/.01)/(2*N));base_seed=0x75000000000000;stride=2**40;methods=['poly0','poly1','poly3','finite'];cases=[3,4]
save('plan.json',dict(source_sha256=H(Path(__file__)),sources={str(p.relative_to(ROOT)):H(p) for p in [ROOT/'src/gbslab/native_polynomial_pnr.py',ROOT/'src/gbslab/native_pnr_newton.py',ROOT/'native/gaussian_pnr_polynomial73_v1.cpp']},binary_hashes={str(polynomial):H(polynomial),str(finite):H(finite)},cases=cases,methods=methods,samples_per_method=N,batch=BATCH,radius=radius,total_failure_probability=.01,one_sided_bounds=16,seed_base=base_seed,seed_stride=stride,scope=__doc__,threads=16))
rows=[]
for ci,case in enumerate(cases):
    data=np.load(ROOT/f'results/iteration73_polynomial_pilot_v2/input_{case}.npz');v=data['v'];x=data['x'];mean=data['mean'];variance=data['variance'];codes=data['codes'];cutoff=int(data['cutoff']);law=data['target'];base=cutoff+1;powers=np.array([base**j for j in range(4)],np.uint64);lookup=np.full(base**4,-1,np.int32);lookup[codes]=np.arange(len(codes))
    for mi,method in enumerate(methods):
        hist=np.zeros(len(law),np.int64);photons=maximum=extreme=0;elapsed=0;compressions=iterations=fallbacks=0;max_residual=0
        if method=='finite':
            model=NativeNewtonPNRSampler(v,x,finite,mean=mean,noise_variance=variance,workers=16,runtime_dirs=runtime)
            old=np.load(ROOT/f'results/iteration71_pnr_branch_pilot_v2/counts_{case}_2_1.npz')['histogram']
        else:
            retained=int(method[-1]);model=NativePolynomialPNRSampler(v,x,polynomial,mean=mean,noise_variance=variance,retained=retained,workers=16,runtime_dirs=runtime)
            old=np.load(ROOT/f'results/iteration73_polynomial_pilot_v2/hist_{case}_{retained}.npz')['histogram']
        event=old/old.sum()>law
        with model:
            for batch in range(N//BATCH):
                seed=base_seed+(ci*100+mi*20+batch)*stride;save(f'started_{case}_{method}_{batch}.json',dict(seed=seed,samples=BATCH));result=model.sample(BATCH,seed);counts=result['counts'];stats=result['statistics'];np.savez_compressed(OUT/f'counts_{case}_{method}_{batch}.npz',counts=counts);save(f'status_{case}_{method}_{batch}.json',dict(seed=seed,returncode=0,seconds=result['wall_seconds'],statistics=stats))
                totals=counts.sum(axis=1);inside=totals<=cutoff;labels=np.full(len(counts),len(codes),np.int32);labels[inside]=lookup[(counts[inside].astype(np.uint64)@powers).astype(np.int64)];assert labels.min()>=0;hist+=np.bincount(labels,minlength=len(law));photons+=int(totals.sum());maximum=max(maximum,int(totals.max()));extreme+=int(np.count_nonzero(totals>=128));elapsed+=result['wall_seconds'];compressions+=stats['compressions'];iterations+=stats['root_iterations'];fallbacks+=stats['bisection_fallback_roots'];max_residual=max(max_residual,stats['maximum_survival_residual'])
                if batch%5==0:print(case,method,batch,'seconds',result['wall_seconds'],flush=True)
        empirical=hist/N;tv=float(abs(empirical-law).sum()/2);lower=max(0,float(empirical[event].sum()-law[event].sum())-radius);upper=min(1,tv+radius+float(law[-1]));row=dict(case=case,method=method,samples=N,empirical_tv=tv,nominal_full_tv_interval=[lower,upper],seconds=elapsed,mean_photons=photons/N,maximum_photons=maximum,events_ge128=extreme,compressions=compressions,root_iterations=iterations,fallbacks=fallbacks,maximum_survival_residual=max_residual);rows.append(row);np.savez_compressed(OUT/f'histogram_{case}_{method}.npz',histogram=hist,target=law,fixed_event=event);save(f'row_{case}_{method}.json',row);print(row,flush=True)
save('summary.json',dict(passed=True,rows=rows,radius=radius));shutil.copy2(__file__,OUT/Path(__file__).name);save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
