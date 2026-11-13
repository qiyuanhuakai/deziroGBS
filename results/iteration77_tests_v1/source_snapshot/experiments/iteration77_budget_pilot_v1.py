"""Build explicit total-tail policy and verify exact branch/coupling/capacity.
The ideal full-TV bound is the analytic probability of entering the tail branch;
it is not a certificate for all floating arithmetic or a fixed PRNG seed.
"""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time,subprocess
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
mp.mp.dps=80
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_budget_number_pnr import NativeBudgetNumberPNRSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
sources=[Path(__file__),ROOT/'src/gbslab/native_budget_number_pnr.py']+[ROOT/'native'/name for name in ('gaussian_pnr_number77_v3.cpp','gaussian_pnr_number77_v2.cpp','gaussian_number77_v2.hpp','gaussian_pnr_projected76_v3.cpp','gaussian_projected76_v3.hpp','gaussian_hermite75_v1.hpp','gaussian_pnr_deep75_v1.cpp','gaussian_complex_moments75_v1.hpp','gaussian_positive_norm74_v1.hpp','gaussian_direct_click.cpp','gaussian_reset_cpu.cpp')]
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';library=OUT/'budget.dll';command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(sources[2]),'-o',str(library)]
save('plan.json',dict(scope=__doc__,command=command,source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},workers=16,input_seed=771001,seed_base=0x89000000000000));
for p in sources:shutil.copy2(p,OUT/p.name)
start=time.perf_counter();build=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=build.returncode,stdout=build.stdout,stderr=build.stderr,seconds=time.perf_counter()-start));assert build.returncode==0,build.stderr;print('Bounded total-number sampler built',flush=True);runtime=[gcc.parent];rows=[];couplings=[]
def tail(n,t,budget):
    z=mp.mpf(float(t))**2;a=mp.mpf(n)/2;term=(1-z)**a;mass=term
    for j in range(1,budget//2+1):term*=z*(a+j-1)/j;mass+=term
    return 1-mass
for case in (0,1,2):
    d=np.load(ROOT/f'results/iteration74_pure_quality_v1/input_{case}.npz') if case<2 else np.load(ROOT/'results/iteration75_lowrank_v2/input.npz');v,t=d['v'],d['t'];count=10000 if case<2 else 1024;seed=0x88000000000000+case*2**40
    with NativeBudgetNumberPNRSampler(v,t,library,exact_total_budget=64,workers=16,runtime_dirs=runtime) as model:r=model.sample(count,seed)
    old=np.load(ROOT/f'results/iteration77_number_pilot_v2/counts_{case}.npz');same=np.array_equal(old['counts'],r['counts']);assert same and np.array_equal(old['totals'],r['total_photons']);row=dict(case=case,records=count,seed=seed,identical_to_strict_v2=same,statistics=r['statistics'],ideal_tv_upper=str(tail(len(t),t[0],64)));rows.append(row);save(f'old_replay_{case}.json',row);np.savez_compressed(OUT/f'old_replay_{case}.npz',counts=r['counts'],totals=r['total_photons'])
d=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');values={};seed=0x89000000000000
for budget in (0,8,16,32,64):
    with NativeBudgetNumberPNRSampler(d['v'],d['t'],library,exact_total_budget=budget,workers=16,runtime_dirs=runtime) as model:r=model.sample(20000,seed)
    totals=r['counts'].sum(axis=1);assert np.array_equal(totals,r['total_photons']) and not (totals%2).any() and r['statistics']['tail_records']==int((totals>budget).sum());values[budget]=r['counts'];np.savez_compressed(OUT/f'coupling_{budget}.npz',counts=r['counts'],totals=r['total_photons']);row=dict(budget=budget,records=20000,statistics=r['statistics'],seconds=r['wall_seconds'],ideal_tv_upper=str(tail(2,d['t'][0],budget)));couplings.append(row);save(f'coupling_{budget}.json',row);print('coupling',budget,row['seconds'],flush=True)
for budget,counts in values.items():mask=counts.sum(axis=1)<=budget;assert np.array_equal(counts[mask],values[64][mask])
expanded=[]
for case,(n,m,t,budget,count) in enumerate([(32,144,np.sqrt(1/33),20,2048),(128,384,np.sqrt(1/129),20,2048),(32,144,np.tanh(.5),16,256),(32,144,np.tanh(.5),20,128),(128,384,np.tanh(.5),12,512)]):
    rng=np.random.default_rng(771001+case);v=np.ascontiguousarray(np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0]);seed=0x89000000000000+(case+1)*2**40;np.savez_compressed(OUT/f'input_{case}.npz',v=v,t=t);save(f'started_{case}.json',dict(inputs=n,outputs=m,budget=budget,count=count,seed=seed));start=time.perf_counter()
    try:
        with NativeBudgetNumberPNRSampler(v,t,library,exact_total_budget=budget,workers=16,runtime_dirs=runtime) as model:r=model.sample(count,seed)
        totals=r['counts'].sum(axis=1);assert np.array_equal(totals,r['total_photons']) and not (totals%2).any() and r['statistics']['tail_records']==int((totals>budget).sum());np.savez_compressed(OUT/f'counts_{case}.npz',counts=r['counts'],totals=r['total_photons']);row=dict(success=True,case=case,inputs=n,outputs=m,budget=budget,records=count,seed=seed,seconds=time.perf_counter()-start,statistics=r['statistics'],ideal_tv_upper=str(tail(n,t,budget)),mean_photons=float(totals.mean()))
    except Exception as e:row=dict(success=False,case=case,inputs=n,outputs=m,budget=budget,records=count,seed=seed,seconds=time.perf_counter()-start,error=str(e))
    expanded.append(row);save(f'row_{case}.json',row);print(row,flush=True)
save('summary.json',dict(passed=all(r['success'] for r in expanded),strict_replays=rows,couplings=couplings,expanded=expanded));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
