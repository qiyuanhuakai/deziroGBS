"""New strong-input cohorts with predeclared pair and vacuum observables.
Five observables, two methods, two sides: delta=.01/20 per side.
Maurer-Pontil theorem 4; iid and numerical-reference assumptions explicit.
"""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
mp.mp.dps=100
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_budget_number_pnr import NativeBudgetNumberPNRSampler
from gbslab.native_sphere_number_pnr import NativeSphereNumberPNRSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
models={'iid':(NativeBudgetNumberPNRSampler,ROOT/'results/iteration77_budget_pilot_v1/budget.dll'),'sphere':(NativeSphereNumberPNRSampler,ROOT/'results/iteration78_pilot_v2/sphere.dll')};runtime=[Path(shutil.which('g++')).parent];d=np.load(ROOT/'results/iteration77_budget_pilot_v1/input_4.npz');v,t=d['v'],float(d['t']);m,n=v.shape;a=mp.mpf(n)/2;z=mp.mpf(t)**2;term=(1-z)**a;p0=term;ca=cl=la=ll=mp.mpf(0);tailmass=mp.mpf(0);B=12
for j in range(1,257):
    term*=z*(a+j-1)/j;da=(a+j)/(2*a*(a+1)*(2*j-1));dl=(j-1)/(2*a*(a+1)*(2*j-1));ca+=term*da;cl+=term*dl
    if 2*j<=B:la+=term*da;ll+=term*dl
    else:tailmass+=term
ratio=z*(a+256)/257;remainder=term*ratio/(1-ratio);ell=(abs(v)**2).sum(axis=1);A=abs(v@v.T)**2;C=abs(v@v.conj().T)**2;L=np.outer(ell,ell)+C;target_pair=float(ca)*A+float(cl)*L;sphere_pair=L/(n*(n+1));iid_pair=np.outer(ell,ell)/(n*n);weights=[np.eye(m),(target_pair-float(1-p0)*sphere_pair>0).astype(float),(target_pair-float(1-p0)*iid_pair>0).astype(float)];names=['same_port_collision','pair_sphere_separator','pair_iid_separator','vacuum_first4','vacuum_first16'];target_means=[float(np.sum(W*target_pair)) for W in weights]
for size in (4,16):
    V=v[:size];o=np.block([[V.real,-V.imag],[V.imag,V.real]]);x=np.diag(np.r_[np.full(n,-t/(1+t)),np.full(n,t/(1-t))]);q=np.eye(2*size)+o@x@o.T;target_means.append(float(1/mp.sqrt(mp.det(mp.matrix(q.tolist())))))
predicted={kind:[float(np.sum(W*(float(la)*A+float(ll)*L+float(tailmass)*pair))) for W in weights] for kind,pair in [('iid',iid_pair),('sphere',sphere_pair)]}
save('plan.json',dict(scope=__doc__,samples_per_method=100000,batch=10000,budget=B,seed_base=0x90000000000000,stride=2**40,observables=names,one_sided_bounds=20,delta_per_side=.0005,theorem='https://arxiv.org/pdf/0907.3740',hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),*[p for _,p in models.values()]]}));shutil.copy2(__file__,OUT/Path(__file__).name);save('target.json',dict(target_means=target_means,predicted_hybrid_pair_means=predicted,coefficient_A=str(ca),coefficient_L=str(cl),low_A=str(la),low_L=str(ll),tail_mass=str(tailmass),omitted_mass_upper=str(remainder)));np.savez_compressed(OUT/'input.npz',v=v,t=t,weights=np.stack(weights),target_pair=target_pair);rows=[]
for panel,(kind,(cls,library)) in enumerate(models.items()):
    values=[];seconds=0;tail_records=odd=0
    with cls(v,t,library,exact_total_budget=B,workers=16,runtime_dirs=runtime) as model:
        for batch in range(10):
            seed=0x90000000000000+(panel*16+batch)*2**40;save(f'started_{kind}_{batch}.json',dict(seed=seed,records=10000));r=model.sample(10000,seed);x=r['counts'].astype(float);totals=x.sum(axis=1);den=totals*np.maximum(totals-1,0);observed=[]
            for W in weights:
                value=(x*(x@W)).sum(axis=1)-x@np.diag(W);observed.append(np.divide(value,den,out=np.zeros(len(x)),where=den>0))
            observed.extend([(x[:,:size].sum(axis=1)==0).astype(float) for size in (4,16)]);observed=np.array(observed).T;assert np.all((observed>=-1e-14)&(observed<=1+1e-14));values.append(observed);seconds+=r['wall_seconds'];tail_records+=int(r['statistics']['tail_records']);odd+=int((totals%2).sum());np.savez_compressed(OUT/f'counts_{kind}_{batch}.npz',counts=r['counts'],totals=r['total_photons'],observables=observed);save(f'row_{kind}_{batch}.json',dict(seed=seed,seconds=r['wall_seconds'],statistics=r['statistics']));print(kind,batch,r['wall_seconds'],flush=True)
    values=np.concatenate(values);mean=values.mean(axis=0);variance=values.var(axis=0,ddof=1);radius=np.sqrt(2*variance*math.log(4000)/len(values))+7*math.log(4000)/(3*(len(values)-1));lower=np.maximum(0,abs(mean-target_means)-radius-float(remainder));row=dict(kind=kind,records=len(values),mean=mean.tolist(),variance=variance.tolist(),nominal_joint99_radii=radius.tolist(),nominal_full_tv_lowers=lower.tolist(),strongest_nominal_full_tv_lower=float(lower.max()),tail_records=tail_records,odd_records=odd,seconds=seconds);rows.append(row);save(f'summary_{kind}.json',row);print(row,flush=True)
save('summary.json',dict(passed=True,rows=rows,observables=names,target_means=target_means,predicted_hybrid_pair_means=predicted));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
