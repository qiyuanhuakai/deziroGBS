"""Measure physical-order outputs from optically calibrated mixtures of window laws."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from scipy.optimize import least_squares
from iteration93_baseline_common_v1 import Baseline93

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
out=parser.parse_args().output.resolve()
out.mkdir(parents=True,exist_ok=False)
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,data):
    (out/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')

old=root/'results/iteration93_reference_v2'
inputs={j:[old/f'input_{j}.npz'] for j in (0,2)}
inputs[4]=[root/f'results/iteration94_actual_quality_v1/input_{j}.npz' for j in (0,1)]
hardware=root/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz'
src=root/'native/gaussian_click_window95_v2.cpp'
files=[Path(__file__),src,src.with_name('gaussian_click_window95_v1.cpp'),src.with_name('gaussian_rng_domain84_v1.hpp'),
       hardware,old/'law_0.npz',old/'law_2.npz',root/'results/iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll',
       *[p for group in inputs.values() for p in group]]
layout=root/'results/iteration83_loss_pilot_v1/input.npz'
reference_groups=[hardware.with_name(f'input_{g}.npz') for g in range(3)]
files.extend([layout,*reference_groups])
save('plan.json',dict(layout='Undo archived internal-to-detector order before any planning; validate all three physical marginals against independent round 81 references',files={p.relative_to(root).as_posix():digest(p) for p in files},
    small='4+4 window, physical order, one component. Compare prepared Frugal K2/K4 and K2/K5 in 2048-record serial batches.',
    actual='4+4, native vs optical-affinity order. 1,2,9 quadrature nodes. Match optical mean/variance solely from pair vacuum probabilities.',
    actual_quality='8192 records each. Exact count law and three preexisting eight-bit marginals; include 99% DKW hardware count interval.',
    actual_timing='8192 full 1152-byte records, 1 and 8 CPU workers; 3 repetitions after warmup; 31 one-record observations on 1 worker',
    rng='quality 0x95600000+64*case+variant; timing 0x95700000+8192*case+512*(rep+1)+16*variant+threads; latency 0x95800000+64*case+variant with disjoint offsets'))
gcc=Path(shutil.which('g++'))
command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE',
         '-I',str(root.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(src),'-o',str(out/'ensemble.dll')]
b=subprocess.run(command,capture_output=True,text=True)
save('build.json',dict(command=command,stderr=b.stderr,code=b.returncode))
b.check_returncode()
runtime=os.add_dll_directory(str(gcc.parent))
lib=C.CDLL(str(out/'ensemble.dll'))
I,P,U=C.c_int,C.c_void_p,C.c_uint64
double,integer,byte=[np.ctypeslib.ndpointer(dtype=t,flags='C_CONTIGUOUS') for t in (np.float64,np.int32,np.uint8)]
def binding(name,args,result=I):
    fun=getattr(lib,name);fun.argtypes=args;fun.restype=result
    return fun
moments=binding('window95_moments',[I,I,double,double,double,double,double,double])
target=binding('window95_true_moments',[I,I,double,double,double])
prepare=binding('window95_prepare',[I,I,I,I,double,double,double],P)
tilt=binding('window95_tilt',[P,I,C.c_double],P)
counts=binding('window95_counts',[P,I,double])
law=binding('window95_law',[P,double])
order_fn=binding('ensemble95_order',[I,I,double,integer])
make=binding('ensemble95_prepare',[I,C.POINTER(P),double],P)
set_order=binding('ensemble95_output_order',[P,integer])
alias_error=binding('ensemble95_alias_error',[P,I,P],C.c_double)
marginal=binding('ensemble95_marginal',[P,I,integer,double])
sample=binding('ensemble95_sample',[P,I,U,U,I,byte])
release=binding('window95_destroy',[P],None)
release_fast=binding('ensemble95_destroy',[P],None)

def fit(base,truth,n):
    x=np.arange(len(base))
    nodes,weights=np.polynomial.hermite.hermgauss(n)
    weights/=weights.sum();nodes*=np.sqrt(2.)
    good=base>0
    logp=np.log(base[good])
    def models(v):
        slopes=v[0]+v[1]*nodes
        a=logp[None,:]+slopes[:,None]*x[good]
        a=np.exp(a-a.max(axis=1,keepdims=True))
        return slopes,weights@(a/a.sum(axis=1,keepdims=True))
    def residual(v):
        _,p=models(v)
        return [(p@x[good]-truth[0])/np.sqrt(truth[1]),p@(x[good]-truth[0])**2/truth[1]-1]
    variance=base@(x-truth[0])**2
    guess=math.sqrt(max(0.,truth[1]-variance))/max(1.,variance)
    result=least_squares(residual,[0.,guess],bounds=([-1,0],[1,1]),xtol=1e-13,gtol=1e-13,ftol=1e-13)
    assert max(abs(np.array(residual(result.x))))<1e-8
    return models(result.x)[0],np.ascontiguousarray(weights)

def restrict(q,mu,ports):
    modes=mu.shape[1]//2
    idx=np.r_[ports,ports+modes]
    return np.ascontiguousarray(q[:,idx][:,:,idx]),np.ascontiguousarray(mu[:,idx])

def draw(plan,n,m,exp,threads,offset=0):
    bits=np.empty((n,m),np.uint8)
    assert sample(plan,n,exp,offset,threads,bits)==0
    return bits

hw=np.load(hardware);ports=hw['ports'];hist=hw['total_click_histogram']
hwcdf=hist.cumsum()/hist.sum()
hardware_radius=math.sqrt(math.log(200)/(2*hist.sum()))
rows=[];timings=[];latency=[];preparation=[]
for case,paths in inputs.items():
    physical=[dict(np.load(p)) for p in paths]
    m=len(physical[0]['u']);s=len(paths);x=np.arange(m+1)
    start=time.perf_counter();q=np.empty((s,2*m,2*m));mu=np.empty((s,2*m))
    for j,z in enumerate(physical):
        u=np.ascontiguousarray(z['u'])
        assert moments(m,len(z['t']),u.view(float),z['t'],z['alpha'].view(float),z['noise'],q[j],mu[j])==0
    optical_time=time.perf_counter()-start
    if case==4:
        start=time.perf_counter()
        detector=np.load(layout)['order']
        assert np.array_equal(np.sort(detector),np.arange(m))
        q,mu=restrict(q,mu,np.argsort(detector))
        optical_time+=time.perf_counter()-start
    start=time.perf_counter();truth=np.empty(2)
    assert target(m,s,q,mu,truth)==0
    target_time=time.perf_counter()-start
    preparation.append(dict(case=case,optical_seconds=optical_time,pair_target_seconds=target_time,mean=truth[0],variance=truth[1]))
    ideal=[]
    if case==4:
        for subset in ports:
            sub,means=restrict(q,mu,subset)
            ptr=prepare(8,8,0,s,sub,means,np.zeros(4));assert ptr
            prob=np.empty(256);assert law(ptr,prob)==0;release(ptr)
            ref_group=np.load(reference_groups[len(ideal)])
            assert np.array_equal(ref_group['ports'],subset)
            assert np.max(abs(prob-ref_group['two_common_sectors']))<1e-12
            ideal.append(prob)
    fast={};external={};variants=(1,2,9) if case==4 else (1,)
    for mode in range(2 if case==4 else 1):
        start=time.perf_counter();order=np.arange(m,dtype=np.int32)
        if mode:assert order_fn(m,s,q,order)==0
        assert np.array_equal(np.sort(order),np.arange(m))
        inverse=np.argsort(order);sub,means=restrict(q,mu,order) if mode else (q,mu)
        order_time=time.perf_counter()-start
        start=time.perf_counter();info=np.zeros(4)
        base=prepare(m,4,4,s,sub,means,info);assert base
        window_time=time.perf_counter()-start
        start=time.perf_counter();base_count=np.empty(m+1)
        assert counts(base,4,base_count)==0 and abs(base_count.sum()-1)<1e-11
        dp_time=time.perf_counter()-start
        for vi,nodes in enumerate(variants):
            variant=mode*3+vi
            start=time.perf_counter()
            slopes,w=fit(base_count,truth,nodes) if nodes>1 else (np.array([0.]),np.array([1.]))
            components=[tilt(base,4,v) for v in slopes] if nodes>1 else [base]
            assert all(components)
            ptr=make(nodes,(P*nodes)(*components),w);assert ptr and set_order(ptr,order)==0
            make_time=time.perf_counter()-start
            fast[variant]=ptr
            bound=max(alias_error(ptr,i,h) for i,h in enumerate(components))+nodes*2**-52
            assert 0<=bound<1e-10
            start=time.perf_counter();count=np.zeros(m+1)
            for h,a in zip(components,w):
                v=np.empty(m+1);assert counts(h,4,v)==0;count+=a*v
            assert abs(count.sum()-1)<1e-11
            exact=dict(count=count,order=order,weights=w,slopes=slopes)
            row=dict(case=case,variant=variant,nodes=nodes,order='optical' if mode else 'physical',
                global_alias_tv_bound=bound,mean=count@x,variance=count@(x-count@x)**2,
                setup_seconds=optical_time+order_time+window_time+make_time+(target_time+dp_time if nodes>1 else 0.),
                table_seconds=window_time,ensemble_seconds=make_time,order_seconds=order_time,
                min_mass_removed=info[0])
            if nodes>1:assert abs(row['mean']-truth[0])<1e-7 and abs(row['variance']-truth[1])<1e-6
            if case==4:
                ks=max(abs(count.cumsum()-hwcdf));row['hardware_count_ks']=ks
                row['hardware_interval']=[max(0,ks-hardware_radius-bound),min(1,ks+hardware_radius+bound)]
                marg=np.zeros((3,256))
                for h,a in zip(components,w):
                    for j,subset in enumerate(ports):
                        v=np.empty(256)
                        assert marginal(h,4,np.ascontiguousarray(inverse[subset],dtype=np.int32),v)==0
                        marg[j]+=a*v
                assert np.allclose(marg.sum(1),1,rtol=0,atol=1e-11)
                exact.update(marginal=marg,ideal=np.asarray(ideal))
                row['eight_port_optical_tv']=(abs(marg-ideal).sum(1)/2).tolist()
                hold=hw['histograms'][:,1,:];row['eight_port_hardware_empirical_tv']=(abs(marg-hold/hold.sum(1)[:,None]).sum(1)/2).tolist()
            else:
                full=np.empty(2**m);assert law(base,full)==0
                ref=np.load(old/f'law_{case}.npz')['click']
                row['full_tv']=abs(full-ref).sum()/2;exact.update(full=full,reference=ref)
            for h in components:
                if h!=base:release(h)
            exp=0x95600000+64*case+variant
            bits=draw(ptr,8192,m,exp,1)
            assert np.array_equal(draw(ptr,8192,m,exp,8),bits)
            assert np.array_equal(draw(ptr,17,m,exp,3,818),bits[818:835])
            ecdf=np.bincount(bits.sum(1).astype(int),minlength=m+1).cumsum()/len(bits)
            row['sample_count_ks']=max(abs(ecdf-count.cumsum()))
            assert row['sample_count_ks']<.04
            if case==4:
                for j,subset in enumerate(ports):
                    patterns=bits[:,subset]@(1<<np.arange(8))
                    observed=np.bincount(patterns,minlength=256)/len(bits)
                    assert abs(observed-marg[j]).sum()/2<.06
            exact['packed_bits']=np.packbits(bits,axis=1,bitorder='little')
            np.savez_compressed(out/f'quality_{case}_{variant}.npz',**exact)
            rows.append(row)
        release(base)
    if case in (0,2):
        for k in (2,4 if case==0 else 5):
            external[16+k]=Baseline93(dict(family='frugal',case=case,cutoff=k,threads=1),physical[0])
    schedules=[(v,t) for v in fast for t in ((1,8) if case==4 else (1,))]+[(v,1) for v in external]
    for rep in (-1,0,1,2):
        n=64 if rep<0 else (8192 if case==4 else 2048)
        batch=schedules[rep%len(schedules):]+schedules[:rep%len(schedules)]
        for variant,threads in batch:
            exp=0x95700000+8192*case+512*(rep+1)+16*variant+threads
            start=time.perf_counter()
            if variant in fast:bits=draw(fast[variant],n,m,exp,threads)
            else:
                raw,_=external[variant].run(n,0,exp,exp)
                bits=raw['clicks']
            elapsed=time.perf_counter()-start
            assert bits.shape==(n,m) and (variant in fast or not raw['failed'].any())
            if rep>=0:timings.append(dict(case=case,variant=variant,threads=threads,rep=rep,n=n,seconds=elapsed,
                hash=hashlib.sha256(bits.tobytes()).hexdigest()))
    if case==4:
        for variant,ptr in fast.items():
            seconds=[]
            for row in range(31):
                start=time.perf_counter();draw(ptr,1,m,0x95800000+64*case+variant,1,row)
                seconds.append(time.perf_counter()-start)
            latency.append(dict(variant=variant,seconds=seconds))
    for p in fast.values():release_fast(p)
    for f in external.values():f.close()
    save('partial.json',dict(rows=rows,timings=timings,preparation=preparation,latency=latency))
    print('completed',case,flush=True)
save('summary.json',dict(rows=rows,timings=timings,preparation=preparation,latency=latency,
     hardware_records=int(hist.sum()),hardware_radius=hardware_radius,holdout=int(hw['histograms'][0,1].sum())))
save('manifest.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()})
