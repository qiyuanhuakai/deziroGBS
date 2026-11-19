"""Independent terminal-number coefficients and a predeclared collision observable.
For f=sum_i n_i(n_i-1)/(N(N-1)) in [0,1], f=0 when N<2,
use a two-sided 99% Maurer-Pontil Theorem 4 interval (delta=.005 per side).
The nominal statistical statement assumes iid records and a numerical target.
"""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time,ctypes as C,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
mp.mp.dps=80
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_number_pnr import NativeNumberPNRSampler
from gbslab.native_budget_number_pnr import NativeBudgetNumberPNRSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
strict=ROOT/'results/iteration77_number_pilot_v2/number.dll';budget=ROOT/'results/iteration77_budget_pilot_v1/budget.dll';runtime=[Path(shutil.which('g++')).parent]
save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in (Path(__file__),strict,budget)},samples=100000,batch=10000,seed=0x8d000000000000,stride=2**40,confidence=.99,theorem='https://arxiv.org/pdf/0907.3740',exact_total_budget=12));shutil.copy2(__file__,OUT/Path(__file__).name)
DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int);ptr=lambda x:x.ctypes.data_as(DP);rng=np.random.default_rng(771777);checks=[];formulas=[]
def collision_given(v,j):
    if not j:return 0.
    a=v.shape[1]/2;ell=(abs(v)**2).sum(axis=1);diag=(v*v).sum(axis=1)
    return float((j*(a+j)*np.sum(abs(diag)**2)+2*j*(j-1)*np.sum(ell**2))/(a*(a+1)*(2*j)*(2*j-1)))
for case in (0,1):
    d=np.load(ROOT/f'results/iteration74_pure_quality_v1/input_{case}.npz');v,t,codes,target,cutoff=d['v'],d['t'],d['codes'],d['target'],int(d['cutoff']);m=len(v);base=cutoff+1;patterns=np.array([(codes//base**j)%base for j in range(m)]).T.astype(np.int64);totals=patterns.sum(axis=1)
    with NativeNumberPNRSampler(v,t,strict,runtime_dirs=runtime) as model:
        fn=model.lib.number77_probe;fn.argtypes=[C.c_void_p,C.c_double,C.c_int,C.c_int,IP,DP,DP,DP,C.c_char_p,C.c_int]
        for total in range(2,65,2):
            selected=totals==total;states=patterns[selected];probs=target[:-1][selected];assert len(states)
            f=(states*(states-1)).sum(axis=1)/(total*(total-1));reference=float(probs@f/probs.sum());formula=collision_given(v,total//2);assert abs(reference-formula)<1e-11;formulas.append(dict(case=case,total=total,reference=reference,formula=formula))
            for k in (total-2,total-1,total):
                ids=rng.integers(0,m,k,dtype=np.int32);mass=np.zeros(1);rates=np.zeros(m);diag=np.zeros(3);error=C.create_string_buffer(4096);rc=fn(model.handle,float(t[0]),total,k,ids.ctypes.data_as(IP),ptr(mass),ptr(rates),ptr(diag),error,len(error));assert rc==0,error.value
                weights=np.ones(len(states));used=np.zeros(m,np.int64)
                for i in ids:weights*=np.maximum(0,states[:,i]-used[i]);used[i]+=1
                ref=float(probs@weights);refs=np.array([float(probs@(weights*np.maximum(0,states[:,i]-used[i]))) for i in range(m)]) if total>k else np.zeros(m);scale=max(1e-290,ref,float(refs.max()));delta=max(abs(mass[0]-ref),float(abs(rates-refs).max()));assert delta<1e-8*scale,(case,total,k,delta,scale);assert abs(rates.sum()-(total-k)*mass[0])<1e-8*max(1e-290,(total-k)*mass[0]);checks.append(dict(case=case,total=total,k=k,relative_difference=delta/scale));np.savez_compressed(OUT/f'terminal_{case}_{total}_{k}.npz',ids=ids,mass=mass,rates=rates,reference=ref,reference_rates=refs)
    print('terminal coefficients and collision formula passed',case,flush=True)
save('checks.json',dict(passed=True,terminal=checks,collision_formulas=formulas))
d=np.load(ROOT/'results/iteration77_budget_pilot_v1/input_4.npz');v,t=d['v'],float(d['t']);n=v.shape[1];a=mp.mpf(n)/2;z=mp.mpf(t)**2;term=(1-z)**a;mass=term;expectation=mp.mpf(0);target_rows=[]
for j in range(1,257):
    term*=z*(a+j-1)/j;mass+=term;value=collision_given(v,j);expectation+=term*mp.mpf(value);target_rows.append(dict(pairs=j,probability=str(term),conditional_collision=value))
tail=abs(1-mass);save('target.json',dict(expectation=str(expectation),omitted_mass=str(tail),rows=target_rows));np.savez_compressed(OUT/'input.npz',v=v,t=t)
values=[];seconds=0;tail_records=0;odd=0
with NativeBudgetNumberPNRSampler(v,t,budget,exact_total_budget=12,workers=16,runtime_dirs=runtime) as model:
    for batch in range(10):
        seed=0x8d000000000000+batch*2**40;save(f'started_{batch}.json',dict(seed=seed,records=10000));r=model.sample(10000,seed);counts=r['counts'];totals=counts.sum(axis=1);x=counts.astype(np.float64);numerator=(x*(x-1)).sum(axis=1);f=np.divide(numerator,totals.astype(float)*np.maximum(totals.astype(float)-1,0),out=np.zeros(len(totals)),where=totals>=2);assert np.all((f>=0)&(f<=1));values.extend(f.tolist());seconds+=r['wall_seconds'];tail_records+=int(r['statistics']['tail_records']);odd+=int((totals%2).sum());np.savez_compressed(OUT/f'counts_{batch}.npz',counts=counts,totals=r['total_photons'],collision=f);save(f'row_{batch}.json',dict(seed=seed,seconds=r['wall_seconds'],statistics=r['statistics']));print('collision batch',batch,r['wall_seconds'],flush=True)
N=len(values);mean=float(np.mean(values));variance=float(np.var(values,ddof=1));radius=math.sqrt(2*variance*math.log(400)/N)+7*math.log(400)/(3*(N-1));lower=max(0.,abs(mean-float(expectation))-radius-float(tail));result=dict(passed=True,terminal_checks=len(checks),collision_formula_checks=len(formulas),records=N,empirical_mean=mean,target_mean=float(expectation),target_omitted_mass=str(tail),sample_variance=variance,nominal_two_sided99_radius=radius,nominal_full_tv_lower=lower,tail_records=tail_records,odd_records=odd,seconds=seconds);save('summary.json',result);print(result,flush=True);save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
