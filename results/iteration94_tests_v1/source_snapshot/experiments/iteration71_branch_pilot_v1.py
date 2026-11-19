"""Frozen, explicit native branch experiment; all outputs created once."""
from pathlib import Path
import os,ctypes as C,hashlib,json,shutil,subprocess,time,argparse,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_double))
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'
sources=[Path(__file__),ROOT/'native/gaussian_click_branches_v1.cpp',ROOT/'native/gaussian_direct_click.cpp',ROOT/'native/gaussian_reset_cpu.cpp']
for p in sources:shutil.copy2(p,OUT/p.name)
command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(sources[1]),'-o',str(OUT/'branches.dll')]
save('plan.json',dict(command=command,source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},small_budgets=[1,2,4,8,16,32,64,128],large_budgets=[1,2,4,8,16],large_samples=128,repetitions=3,seed=711001,scope='Bounded signed Gaussian branches, no probability clipping. Budget 1 recovers previous closure, sufficient budget equals known exact threshold identity in real arithmetic. Full induced laws checked against independent 80-digit physical-space inclusion-exclusion. Large-input timings compare different approximation errors. No universal quality or novel exact-identity claim.'))
start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start));assert r.returncode==0,r.stderr
print('Native signed branches built',flush=True)
runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(OUT/'branches.dll'));DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);error=C.create_string_buffer(4096)
lib.tc70_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];lib.tc70_create.restype=C.c_void_p
lib.tc70_destroy.argtypes=[C.c_void_p];lib.tc70_destroy.restype=None
lib.tc71_prefix.argtypes=[C.c_void_p,DP,C.c_int,BP,C.c_int,DP,DP,C.c_char_p,C.c_int]
lib.tc71_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,DP,BP,DP,C.c_char_p,C.c_int]
lib.tc70_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,DP,DP,BP,DP,C.c_char_p,C.c_int]
def create(v,x):
    handle=lib.tc70_create(*v.shape,ptr(v),ptr(x),error,len(error));assert handle,error.value;return handle
def sample(h,m,count,seed,threads,budget,mean,variance,old=False):
    output=np.zeros((count,m),np.uint8);stats=np.zeros(5 if old else 9);start=time.perf_counter()
    params=[h,count,seed,threads]+([] if old else [budget])+[ptr(mean),ptr(variance),output.ctypes.data_as(BP),ptr(stats),error,len(error)]
    rc=(lib.tc70_sample if old else lib.tc71_sample)(*params);seconds=time.perf_counter()-start
    return output,stats,dict(returncode=rc,error=error.value.decode() if rc else '',seconds=seconds)
def induced(h,m,mean,budget):
    law=np.zeros(2**m);stats=np.zeros((2**m,4));start=time.perf_counter()
    for mask in range(2**m):
        bits=np.array([(mask>>j)&1 for j in range(m)],np.uint8);p=np.zeros(1)
        rc=lib.tc71_prefix(h,ptr(mean),m,bits.ctypes.data_as(BP),budget,ptr(p),ptr(stats[mask]),error,len(error))
        if rc:return law,stats,dict(returncode=rc,error=error.value.decode(),mask=mask,seconds=time.perf_counter()-start)
        law[mask]=p[0]
    return law,stats,dict(returncode=0,seconds=time.perf_counter()-start)
def target(v,x,mean):
    # Independent full physical-space vacuum overlaps, not native updates.
    mp.mp.dps=80;m,n=v.shape;V=mp.matrix([[mp.mpc(z) for z in row] for row in v]);O=mp.matrix(2*m,2*n)
    for i in range(m):
        for j in range(n):O[i,j]=V[i,j].real;O[i,n+j]=-V[i,j].imag;O[m+i,j]=V[i,j].imag;O[m+i,n+j]=V[i,j].real
    X=mp.matrix(x.tolist());mu=O*mp.matrix(mean.tolist());Q=mp.eye(2*m)+O*X*O.T;vac=[mp.mpf(1)]
    for mask in range(1,2**m):
        indices=[i for i in range(m) if mask>>i&1];indices+= [i+m for i in indices]
        A=mp.matrix([[Q[i,j] for j in indices] for i in indices]);w=mp.matrix([mu[i] for i in indices]);vac.append(mp.exp(-(w.T*mp.lu_solve(A,w))[0]/2)/mp.sqrt(mp.det(A)))
    law=[]
    for mask in range(2**m):
        zero=((1<<m)-1)^mask;terms=[];sub=mask
        while True:
            terms.append((-1)**sub.bit_count()*vac[zero|sub])
            if sub==0:break
            sub=(sub-1)&mask
        law.append(mp.fsum(terms))
    assert min(law)>-mp.mpf('1e-60') and abs(mp.fsum(law)-1)<mp.mpf('1e-60')
    return np.array([float(p) for p in law]),[mp.nstr(p,82) for p in law]
rows=[];failures=[];checks=[]
for case in range(6):
    if case<4:
        d=np.load(ROOT/f'results/iteration69_archive/iteration69_quality_pilot_20261110/input_{case}.npz');v=np.ascontiguousarray(d['v']);x=np.ascontiguousarray(d['x0'])
    else:
        m,n=8,3;r=.9 if case==4 else 1.4;eta=.65 if case==4 else .43;rng=np.random.default_rng(711100+case);v=np.ascontiguousarray(np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0]);x=np.diag(np.r_[np.full(n,.5*eta*np.expm1(2*r)),np.full(n,.5*eta*np.expm1(-2*r))])
    m,n=v.shape;mean=np.zeros(2*n)
    if case==5:mean=np.random.default_rng(711200+case).normal(size=2*n)*.2
    variance=np.zeros(2*n);expected,decimal=target(v,x,mean);save(f'target_{case}.json',dict(probabilities=decimal,precision_digits=80));np.savez_compressed(OUT/f'input_{case}.npz',v=v,x=x,mean=mean,variance=variance,target=expected)
    h=create(v,x);budgets=[1,2,4,8] if m==4 else [1,2,4,8,16,32,64,128]
    for budget in budgets:
        law,stats,status=induced(h,m,mean,budget);np.savez_compressed(OUT/f'law_{case}_{budget}.npz',law=law,stats=stats);save(f'status_{case}_{budget}.json',status)
        if status['returncode']:
            failures.append(dict(case=case,budget=budget,**status));print('FAILURE',failures[-1],flush=True);continue
        tv=float(abs(law-expected).sum()/2);row=dict(case=case,modes=m,budget=budget,tv=tv,mass=float(law.sum()),peak_components=float(stats[:,2].max()),max_absolute_weight_sum=float(stats[:,3].max()),prefix_seconds=status['seconds']);rows.append(row);save(f'row_{case}_{budget}.json',row);print(row,flush=True)
        assert abs(law.sum()-1)<1e-10 and law.min()>=0
        if budget>=2**(m-1):assert tv<2e-10,(case,budget,tv)
        if budget in (1,2,8):
            a,sa,ss=sample(h,m,40000,711001+case,16,budget,mean,variance);save(f'sample_status_{case}_{budget}.json',ss);np.savez_compressed(OUT/f'samples_{case}_{budget}.npz',clicks=a,stats=sa)
            if ss['returncode']:failures.append(dict(case=case,budget=budget,kind='sample',**ss));continue
            empirical=np.bincount(a@2**np.arange(m),minlength=2**m)/len(a);tv_emp=float(abs(empirical-law).sum()/2);assert tv_emp<np.sqrt((2**m*np.log(2)+np.log(1000))/(2*len(a)))
            checks.append(dict(case=case,budget=budget,empirical_tv=tv_emp,seconds=ss['seconds']))
    a,_,s=sample(h,m,1001,711301+case,1,1,mean,variance);b,_,t=sample(h,m,1001,711301+case,16,1,mean,variance);old,_,u=sample(h,m,1001,711301+case,1,1,mean,variance,True)
    assert s['returncode']==t['returncode']==u['returncode']==0 and np.array_equal(a,b) and np.array_equal(a,old)
    for budget in (2,8):
        a,_,s=sample(h,m,1001,711401+case,1,budget,mean,variance);b,_,t=sample(h,m,1001,711401+case,16,budget,mean,variance)
        assert s['returncode']==t['returncode']==0 and np.array_equal(a,b),(case,budget,s,t)
    lib.tc70_destroy(h)
save('small_summary.json',dict(rows=rows,checks=checks,failures=failures,schedule_replay=True,budget_one_replay=True))
large=np.load(ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110/input.npz');v=np.ascontiguousarray(large['v']);x=np.ascontiguousarray(large['x0']);mean=np.ascontiguousarray(large['mean']);variance=np.ascontiguousarray(large['variance']);h=create(v,x);large_rows=[]
for rep in range(3):
    order=[1,2,4,8,16];order=order[rep:]+order[:rep]
    for budget in order:
        output,stats,status=sample(h,144,128,711501+rep,16,budget,mean,variance);np.savez_compressed(OUT/f'large_{rep}_{budget}.npz',clicks=output,stats=stats);row=dict(rep=rep,budget=budget,**status,stats=stats.tolist());save(f'large_{rep}_{budget}.json',row);large_rows.append(row);print(row,flush=True)
lib.tc70_destroy(h)
save('summary.json',dict(passed=not failures and all(r['returncode']==0 for r in large_rows),small=rows,checks=checks,failures=failures,large=large_rows,binary_sha256=H(OUT/'branches.dll')))
assert {p.relative_to(ROOT).as_posix():H(p) for p in sources}==json.loads((OUT/'plan.json').read_text())['source_hashes']
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
