"""Build structured PNR; independent complex Wick and exact parity endpoints."""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time,subprocess,ctypes as C,functools
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_classical_tail_pnr import NativeClassicalTailPNRSampler,DP
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return np.asarray(a).ctypes.data_as(DP)
sources=[Path(__file__),ROOT/'src/gbslab/native_classical_tail_pnr.py']+[ROOT/'native'/n for n in ('gaussian_pnr_classical74_v1.cpp','gaussian_pnr_structured74_v1.cpp','gaussian_positive_norm74_v1.hpp','gaussian_pnr_polynomial73_v1.cpp','gaussian_direct_click.cpp','gaussian_reset_cpu.cpp')]
for p in sources:shutil.copy2(p,OUT/p.name)
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';library=OUT/'classical.dll';command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(sources[2]),'-o',str(library)]
save('plan.json',dict(source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},command=command,scope=__doc__,seed=742001,pure_samples=100000,noisy_samples=20000,workers=16))
start=time.perf_counter();build=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=build.returncode,stdout=build.stdout,stderr=build.stderr,seconds=time.perf_counter()-start));assert build.returncode==0,build.stderr;print('Structured PNR built',flush=True)
runtime=[gcc.parent];handles=[os.add_dll_directory(str(p)) for p in runtime];lib=C.CDLL(str(library));lib.pnrs74_norm.argtypes=[C.c_int,C.c_int,DP,DP,DP,DP,C.c_char_p,C.c_int];error=C.create_string_buffer(4096);checks=[]
def wick(alpha,matrix):
    @functools.lru_cache(None)
    def get(mask):
        if not mask:return 1+0j
        i=(mask&-mask).bit_length()-1;rest=mask&~(1<<i);value=alpha[i]*get(rest)
        for j in range(len(alpha)):
            if rest>>j&1:value+=matrix[i,j]*get(rest&~(1<<j))
        return value
    return get((1<<len(alpha))-1)
rng=np.random.default_rng(742001)
for k in (1,2,3,4):
    for rank in (1,4,50):
        for rep in range(5):
            alpha=np.ascontiguousarray((rng.normal(size=k)+1j*rng.normal(size=k))*.2);raw=(rng.normal(size=(k,k))+1j*rng.normal(size=(k,k)))*.2;m=np.ascontiguousarray((raw+raw.T)/2);factor=np.ascontiguousarray((rng.normal(size=(k,rank))+1j*rng.normal(size=(k,rank)))/np.sqrt(rank)*.3);normal=factor@factor.conj().T;matrix=np.block([[m,normal],[normal.T,m.conj()]])
            target=wick(np.r_[alpha,alpha.conj()],matrix);out=np.zeros(1);rc=lib.pnrs74_norm(k,rank,ptr(alpha),ptr(m),ptr(factor),ptr(out),error,len(error));row=dict(k=k,rank=rank,rep=rep,returncode=rc,error=error.value.decode() if rc else '',native=float(out[0]),reference=[target.real,target.imag],difference=float(abs(out[0]-target)));checks.append(row);np.savez_compressed(OUT/f'norm_{k}_{rank}_{rep}.npz',alpha=alpha,anomalous=m,factor=factor,target=target,native=out);save(f'norm_{k}_{rank}_{rep}.json',row);assert rc==0 and row['difference']<1e-10*max(1,abs(target)),row
print('60 complex Wick checks passed',flush=True)
endpoints=[]
with NativeClassicalTailPNRSampler(np.ones((1,1),complex),[1/3],library,retained=3,runtime_dirs=runtime) as model:
    lib.pnrs74_endpoint.argtypes=[C.c_void_p,DP,DP,C.c_int,C.POINTER(C.c_int),C.c_double,DP,C.c_char_p,C.c_int]
    for k in range(1,9):
        for remain in (0.,1e-15,1e-9,.5,1.):
            modes=np.zeros(k,np.int32);out=np.zeros(1);mean=np.zeros(2);rc=lib.pnrs74_endpoint(model.handle,ptr(model.squeezing),ptr(mean),k,modes.ctypes.data_as(C.POINTER(C.c_int)),remain,ptr(out),error,len(error));n=remain*(1/3)**2/(1-remain**2*(1/3)**2);m=-(1/3)/(1-remain**2*(1/3)**2);target=wick(np.zeros(2*k),np.block([[np.full((k,k),m),np.full((k,k),n)],[np.full((k,k),n),np.full((k,k),m)]]));row=dict(k=k,remaining=remain,returncode=rc,native=out[0],target=target.real,difference=float(abs(out[0]-target)));endpoints.append(row);assert rc==0 and out[0]>=0 and abs(out[0]-target)<1e-10*max(1,abs(target)),row
            if remain==0 and k%2:assert out[0]==0,row
save('positive_validation.json',dict(passed=True,norm_checks=checks,endpoints=endpoints));print('40 endpoint checks passed; odd pure vacuum norms exactly zero',flush=True)
rows=[]
for case in ('pure',0,1,2,3,4,'large'):
    if case=='pure':v=np.ones((1,1),complex);t=np.array([1/3]);mean=np.zeros(2);variance=np.zeros(2);count=100000
    else:
        path=ROOT/('results/iteration70_archive/iteration70_frugal_large_v1_20261110/input.npz' if case=='large' else f'results/iteration73_polynomial_pilot_v2/input_{case}.npz');data=np.load(path);v=data['v'];x=data['x0'] if case=='large' else data['x'];n=v.shape[1];t=-np.diag(x)[:n]/(1+np.diag(x)[:n]);reconstructed=np.diag(np.r_[-t/(1+t),t/(1-t)]);assert np.max(abs(x-reconstructed))<1e-12;mean=data['mean'];variance=data['variance'];count=32 if case=='large' else 20000
    for retained in (0,1,2,3):
        row=dict(case=case,retained=retained,count=count,seed=742101+(0 if isinstance(case,str) else 100*case));save(f'started_{case}_{retained}.json',row);start=time.perf_counter()
        try:
            with NativeClassicalTailPNRSampler(v,t,library,mean=mean,noise_variance=variance,retained=retained,workers=16,runtime_dirs=runtime) as model:r=model.sample(count,row['seed'])
            total=r['counts'].sum(axis=1);row.update(success=True,wall_seconds=time.perf_counter()-start,statistics=r['statistics'],odd_records=int(np.count_nonzero(total%2)),mean_photons=float(total.mean()));np.savez_compressed(OUT/f'counts_{case}_{retained}.npz',counts=r['counts'],v=v,tanh_squeezing=t,mean=mean,variance=variance)
        except Exception as e:row.update(success=False,wall_seconds=time.perf_counter()-start,error=str(e))
        rows.append(row);save(f'row_{case}_{retained}.json',row);print(row,flush=True)
save('summary.json',dict(completed=True,all_samples_succeeded=all(r['success'] for r in rows),positive_norm_checks_passed=True,rows=rows));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
