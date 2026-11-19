"""Remaining-number pruning: independent Fock coefficient checks for pure sampling.
Large-input polynomial-Wick checks and explicit resource failures are retained.
"""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time,subprocess,ctypes as C,functools,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_number_pnr import NativeNumberPNRSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int)
def ptr(a):return a.ctypes.data_as(DP)
sources=[Path(__file__),ROOT/'src/gbslab/native_number_pnr.py']+[ROOT/'native'/name for name in ('gaussian_pnr_number77_v2.cpp','gaussian_number77_v2.hpp','gaussian_pnr_projected76_v3.cpp','gaussian_projected76_v3.hpp','gaussian_hermite75_v1.hpp','gaussian_pnr_deep75_v1.cpp','gaussian_complex_moments75_v1.hpp','gaussian_positive_norm74_v1.hpp','gaussian_direct_click.cpp','gaussian_reset_cpu.cpp')]
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';library=OUT/'number.dll';command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(sources[2]),'-o',str(library)]
save('plan.json',dict(scope=__doc__,command=command,source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},seed=770001,workers=16));
for p in sources:shutil.copy2(p,OUT/p.name)
start=time.perf_counter();build=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=build.returncode,stdout=build.stdout,stderr=build.stderr,seconds=time.perf_counter()-start));assert build.returncode==0,build.stderr;print('Number-conditioned core compiled',flush=True)
runtime=[gcc.parent];rng=np.random.default_rng(770001);checks=[];large=[]
def probe(model,total,ids):
    ids=np.ascontiguousarray(ids,dtype=np.int32);mass=np.zeros(1);rates=np.zeros(model.modes);diag=np.zeros(3);error=C.create_string_buffer(4096);fn=model.lib.number77_probe;fn.argtypes=[C.c_void_p,C.c_double,C.c_int,C.c_int,IP,DP,DP,DP,C.c_char_p,C.c_int];rc=fn(model.handle,float(model.squeezing[0]),total,len(ids),ids.ctypes.data_as(IP),ptr(mass),ptr(rates),ptr(diag),error,len(error));assert rc==0,error.value;return mass[0],rates,diag
for case in (0,1):
    d=np.load(ROOT/f'results/iteration74_pure_quality_v1/input_{case}.npz');v,t,codes,target,cutoff=d['v'],d['t'],d['codes'],d['target'],int(d['cutoff']);m=len(v);base=cutoff+1;patterns=np.array([(codes//base**j)%base for j in range(m)]).T.astype(np.int64);totals=patterns.sum(axis=1)
    with NativeNumberPNRSampler(v,t,library,workers=16,runtime_dirs=runtime) as model:
        for total in range(0,33,2):
            selected=totals==total;states=patterns[selected];probs=target[:-1][selected]
            for k in range(min(8,total)+1):
                ids=rng.integers(0,m,k,dtype=np.int32);mass,rates,diag=probe(model,total,ids);weights=np.ones(len(states));used=np.zeros(m,np.int64)
                for i in ids:weights*=np.maximum(0,states[:,i]-used[i]);used[i]+=1
                reference=float(probs@weights);refs=np.array([float(probs@(weights*np.maximum(0,states[:,i]-used[i]))) for i in range(m)]) if total>k else np.zeros(m);scale=max(1,reference,float(refs.max()));delta=max(abs(mass-reference),float(abs(rates-refs).max()));sum_error=abs(float(rates.sum())-(total-k)*mass);row=dict(case=case,total=total,k=k,mass=mass,reference=reference,maximum_difference=delta,reference_scale=scale,detector_sum_error=sum_error,active_rank=int(diag[0]),terms=int(diag[1]));checks.append(row);np.savez_compressed(OUT/f'fock_{case}_{total}_{k}.npz',ids=ids,mass=mass,reference=reference,rates=rates,reference_rates=refs,diagnostics=diag);save(f'fock_{case}_{total}_{k}.json',row);assert delta<2e-10*scale and sum_error<2e-10*max(1,(total-k)*mass),row
    print('Fock coefficients passed',case,flush=True)
def wick_coefficient(v,t,total,ids):
    k=len(ids);V=v[ids];M=-t*(V@V.T);N=t*t*(V@V.conj().T);matrix=np.block([[M,N],[N.T,M.conj()]]).astype(np.clongdouble)
    @functools.lru_cache(None)
    def f(mask):
        result=np.zeros(k+1,np.clongdouble)
        if not mask:result[0]=1;return result
        i=(mask&-mask).bit_length()-1;rest=mask&~(1<<i)
        for j in range(2*k):
            if rest>>j&1:
                value=f(rest&~(1<<j));shift=int((i<k)!=(j<k))
                if shift:result[1:]+=matrix[i,j]*value[:-1]
                else:result+=matrix[i,j]*value
        return result
    polynomial=f((1<<(2*k))-1);shape=v.shape[1]/2+k;remaining=total-k;coefficient=0j
    for degree,value in enumerate(polynomial):
        if remaining>=degree and (remaining-degree)%2==0:
            pairs=(remaining-degree)//2;factor=math.prod((shape+j)/(j+1) for j in range(pairs))*t**(2*pairs);coefficient+=value*factor
    return coefficient*(1-t*t)**(v.shape[1]/2)
for n in (4,32,128):
    m=2*n;v=np.ascontiguousarray(np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0]);t=.2;np.savez_compressed(OUT/f'large_input_{n}.npz',v=v,t=t)
    with NativeNumberPNRSampler(v,t,library,runtime_dirs=runtime) as model:
        for total in (2,4,8,12):
            for k in (0,1,2,4):
                if k>total:continue
                ids=rng.integers(0,m,k,dtype=np.int32);mass,rates,diag=probe(model,total,ids);reference=wick_coefficient(v,t,total,ids);selected=sorted(set([0,m//2,m-1]));refs=[wick_coefficient(v,t,total,np.r_[ids,i].astype(np.int32)) for i in selected] if total>k else np.zeros(len(selected));delta=max(float(abs(mass-reference)),float(abs(rates[selected]-refs).max()));row=dict(inputs=n,total=total,k=k,maximum_difference=delta,detector_sum_error=abs(float(rates.sum())-(total-k)*mass));large.append(row);np.savez_compressed(OUT/f'large_{n}_{total}_{k}.npz',ids=ids,mass=mass,reference=reference,rates=rates,selected=selected,reference_rates=refs,diagnostics=diag);save(f'large_{n}_{total}_{k}.json',row);assert delta<2e-10 and row['detector_sum_error']<2e-10,row
    print('Large-input coefficient checks passed',n,flush=True)
save('checks.json',dict(passed=True,fock=checks,large=large));rows=[]
for case in (0,1,2):
    data=np.load(ROOT/f'results/iteration74_pure_quality_v1/input_{case}.npz') if case<2 else np.load(ROOT/'results/iteration75_lowrank_v2/input.npz');v,t=data['v'],data['t'];count=10000 if case<2 else 1024;seed=0x88000000000000+case*2**40;save(f'started_{case}.json',dict(case=case,records=count,seed=seed));start=time.perf_counter()
    try:
        with NativeNumberPNRSampler(v,t,library,workers=16,runtime_dirs=runtime) as model:r=model.sample(count,seed)
        counts=r['counts'];assert np.array_equal(counts.sum(axis=1),r['total_photons']) and not (r['total_photons']%2).any();row=dict(success=True,case=case,records=count,seed=seed,wall_seconds=time.perf_counter()-start,statistics=r['statistics'],mean_photons=float(r['total_photons'].mean()));np.savez_compressed(OUT/f'counts_{case}.npz',counts=counts,totals=r['total_photons'])
    except Exception as e:row=dict(success=False,case=case,records=count,seed=seed,wall_seconds=time.perf_counter()-start,error=str(e))
    rows.append(row);save(f'row_{case}.json',row);print(row,flush=True)
data=np.load(OUT/'large_input_32.npz')
try:
    with NativeNumberPNRSampler(data['v'],float(data['t']),library,workers=1,runtime_dirs=runtime) as model:r=model.sample(1,770099,fixed_total=10)
    capacity=dict(success=True,counts=r['counts'].tolist(),statistics=r['statistics'])
except Exception as e:capacity=dict(success=False,error=str(e))
save('capacity_fixed10.json',capacity);save('summary.json',dict(checks_passed=True,all_samples_succeeded=all(r['success'] for r in rows),rows=rows,capacity_fixed10=capacity));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
