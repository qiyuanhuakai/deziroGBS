"""Overlapping exact click marginals, with an optional two-component count correction."""
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
def save(name,data):
    (out/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

small=[root/f'results/iteration93_reference_v2/input_{j}.npz' for j in range(4)]
refs=[p.with_name(p.name.replace('input','law')) for p in small]
actual=[root/f'results/iteration94_actual_quality_v1/input_{j}.npz' for j in range(2)]
hardware=root/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz'
source=root/'native/gaussian_click_window95_v1.cpp'
files=[Path(__file__),source,source.with_name('gaussian_rng_domain84_v1.hpp'),*small,*refs,*actual,
       hardware,root/'results/iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll',
       root/'experiments/iteration93_baseline_common_v1.py']
configs=[(4,0),(4,2),(4,4)]
save('plan.json',dict(hashes={p.relative_to(root).as_posix():digest(p) for p in files},
    configs=configs,cases=['four pure','four displaced','eight pure','lossy two','actual combined spectra'],
    correction='equal-weight exponential tilts of the complete window law; mean and variance from all optical pairs; no hardware fit',
    strict_tilt_gate='apply if optical variance exceeds base variance by 1e-7 and two-moment solve residuals <=1e-7',
    quality_records=8192,actual_comparison='deterministic count DP vs hardware; one shared two-sided 99% hardware DKW band',
    time_records=2048,warmup=64,repetitions=3,threads=1,
    seeds='quality 0x95100000 + 32*case + 2*config + component; timing 0x95200000+1024*case+32*(rep+1)+label',
    controls='exact enumeration for small cases, full retained count laws; replay first/last sample of each 8192 record batch'))
compiler=Path(shutil.which('g++'))
cmd=[str(compiler),'-shared','-O3','-std=c++20','-DEIGEN_DONT_PARALLELIZE','-fno-fast-math','-ffp-contract=off',
     '-I',str(root.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(source),'-o',str(out/'window.dll')]
build=subprocess.run(cmd,capture_output=True,text=True)
save('build.json',dict(command=cmd,stderr=build.stderr,returncode=build.returncode))
build.check_returncode()
runtime=os.add_dll_directory(str(compiler.parent))
lib=C.CDLL(str(out/'window.dll'))
arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
bits_type=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
integer=C.c_int
handle=C.c_void_p
lib.window95_moments.argtypes=[integer,integer,arr,arr,arr,arr,arr,arr]
lib.window95_prepare.restype=handle
lib.window95_prepare.argtypes=[integer,integer,integer,integer,arr,arr,arr]
lib.window95_tilt.restype=handle
lib.window95_tilt.argtypes=[handle,integer,C.c_double]
lib.window95_counts.argtypes=[handle,integer,arr]
lib.window95_law.argtypes=[handle,arr]
lib.window95_true_moments.argtypes=[integer,integer,arr,arr,arr]
lib.window95_sample.argtypes=[handle,handle,integer,C.c_uint64,C.c_uint64,bits_type]
lib.window95_destroy.argtypes=[handle]

def sample(pair,m,n,experiment,offset=0):
    data=np.empty((n,m),np.uint8)
    assert lib.window95_sample(*pair,n,experiment,offset,data)==0
    return data

def optical(paths):
    inputs=[dict(np.load(p)) for p in paths]
    m=len(inputs[0]['u'])
    q=np.empty((len(paths),2*m,2*m));mu=np.empty((len(paths),2*m))
    for j,z in enumerate(inputs):
        u=np.ascontiguousarray(z['u']);t=z['t'];alpha=z['alpha'];l=z['noise']
        assert np.allclose(u.conj().T@u,np.eye(len(t)),atol=1e-12,rtol=0)
        assert lib.window95_moments(m,len(t),u.view(float),t,alpha.view(float),l,q[j],mu[j])==0
        idx=np.unique(np.linspace(0,2*m-1,min(16,2*m),dtype=int))
        a=np.block([[u.real,-u.imag],[u.imag,u.real]])[idx]
        core=np.diag(np.r_[t/(1-t),-t/(1+t)])+2*l@l.T
        assert np.allclose(q[j][np.ix_(idx,idx)],np.eye(len(idx))+a@core@a.T,atol=1e-12,rtol=0)
        assert np.allclose(mu[j,idx],a@(np.sqrt(2)*np.r_[alpha.real,alpha.imag]),atol=1e-12,rtol=0)
    return inputs,q,mu

def fit(counts,truth):
    x=np.arange(len(counts))
    mask=counts>0
    logp=np.log(counts[mask])
    def tilted(slope):
        logs=logp+slope*x[mask]
        w=np.exp(logs-max(logs))
        p=np.zeros_like(counts);p[mask]=w/sum(w)
        return p
    def residual(v):
        prob=(tilted(v[0])+tilted(v[1]))/2
        return np.array([(prob@x-truth[0])/max(1.,np.sqrt(truth[1])),
            (prob@(x-truth[0])**2-truth[1])/max(1.,truth[1])])
    width=max(.02,math.sqrt(max(0,truth[1]-counts@(x-truth[0])**2))/max(1.,truth[1]))
    result=least_squares(residual,[-min(2.,width),min(2.,width)],bounds=([-4.,0.],[0.,4.]),
                         ftol=1e-14,xtol=1e-14,gtol=1e-14)
    return result.x, max(abs(residual(result.x))), (tilted(result.x[0])+tilted(result.x[1]))/2

def distribution(ptr,m,h,full=False):
    a=np.empty(2**m if full else m+1)
    rc=lib.window95_law(ptr,a) if full else lib.window95_counts(ptr,h,a)
    assert rc==0 and a.min()>=0 and abs(a.sum()-1)<1e-11
    return a

hist=np.load(hardware)['total_click_histogram']
hardware_cdf=hist.cumsum()/hist.sum()
radius=math.sqrt(math.log(200)/(2*hist.sum()))
rows=[];timings=[];setups=[]
for j,paths in enumerate([[p] for p in small]+[actual]):
    begin=time.perf_counter();inputs,q,mu=optical(paths)
    input_seconds=time.perf_counter()-begin
    m=len(inputs[0]['u']); x=np.arange(m+1)
    begin=time.perf_counter();truth=np.zeros(2)
    assert lib.window95_true_moments(m,len(paths),q,mu,truth)==0
    moments_seconds=time.perf_counter()-begin
    if j<4:
        ref=np.load(refs[j])['click']
        pop=np.array([i.bit_count() for i in range(2**m)])
        assert abs(ref@pop-truth[0])<1e-11 and abs(ref@(pop-truth[0])**2-truth[1])<1e-11
    setups.append(dict(case=j,input_and_crosscheck_seconds=input_seconds,pair_moment_seconds=moments_seconds,
                       true_mean=truth[0],true_variance=truth[1]))
    jobs={};owned=[]
    for i,(width,h) in enumerate(configs):
        begin=time.perf_counter();info=np.zeros(4)
        base=lib.window95_prepare(m,width,h,len(paths),q,mu,info)
        assert base
        prep=time.perf_counter()-begin;owned.append(base)
        begin=time.perf_counter()
        count=distribution(base,m,h)
        assert abs(count@x-truth[0])<1e-8
        variants=[('base',(base,None),count,None,0.)]
        if truth[1]-count@(x-truth[0])**2>1e-7:
            slopes,residual,predicted=fit(count,truth)
            assert residual<=1e-7,(j,i,slopes,residual)
            handles=tuple(lib.window95_tilt(base,h,v) for v in slopes)
            assert all(handles);owned.extend(handles)
            mixed=sum(distribution(p,m,h) for p in handles)/2
            assert np.max(abs(mixed-predicted))<1e-10
            variants.append(('tilted',handles,mixed,slopes.tolist(),residual))
        calibration=time.perf_counter()-begin
        for k,(name,pair,law,slopes,residual) in enumerate(variants):
            key=2*i+k;jobs[key]=pair
            row=dict(case=j,width=width,history=h,variant=name,label=key,setup_seconds=prep,
                dp_fit_and_check_seconds=calibration,discarded_negative_mass=info[0],empty_contexts=int(info[1]),
                entries=int(info[2]),tiles=int(info[3]),slopes=slopes,fit_residual=residual,
                mean=float(law@x),variance=float(law@(x-law@x)**2))
            payload=dict(count=law)
            if j<4:
                full=distribution(pair[0],m,h,True)
                if pair[1]:full=(full+distribution(pair[1],m,h,True))/2
                assert np.max(abs(np.bincount(pop,weights=full,minlength=m+1)-law))<1e-12
                payload.update(click=full,reference=ref);row['full_tv']=abs(full-ref).sum()/2
            else:
                ks=max(abs(law.cumsum()-hardware_cdf))
                row.update(hardware_cdf_distance=ks,hardware_cdf_interval=[max(0.,ks-radius),min(1.,ks+radius)])
            exp=0x95100000+32*j+key
            bits=sample(pair,m,8192,exp)
            for off in (0,8191):
                assert np.array_equal(sample(pair,m,1,exp,off)[0],bits[off])
            empirical=np.bincount(bits.sum(1).astype(int),minlength=m+1).cumsum()/len(bits)
            row['empirical_vs_model_count_ks']=max(abs(empirical-law.cumsum()))
            assert row['empirical_vs_model_count_ks']<.04
            np.savez_compressed(out/f'quality_{j}_{key}.npz',bits=np.packbits(bits,axis=1,bitorder='little'),**payload)
            rows.append(row)
    frugal={}
    if j in (0,2):
        for cutoff in (2,4 if j==0 else 5):
            frugal[16+cutoff]=Baseline93(dict(family='frugal',case=j,cutoff=cutoff,threads=1),inputs[0])
    labels=[*jobs,*frugal]
    for rep in (-1,0,1,2):
        n=64 if rep<0 else 2048
        for label in labels[rep%len(labels):]+labels[:rep%len(labels)]:
            exp=0x95200000+1024*j+32*(rep+1)+label
            begin=time.perf_counter()
            if label in jobs:bits=sample(jobs[label],m,n,exp)
            else:
                raw,_=frugal[label].run(n,0,exp,exp)
                bits=raw['clicks']
            elapsed=time.perf_counter()-begin
            assert bits.shape==(n,m) and (label in jobs or not raw['failed'].any())
            if rep>=0:
                timings.append(dict(case=j,label=label,rep=rep,n=n,seconds=elapsed,
                    bits_sha256=hashlib.sha256(bits.tobytes()).hexdigest()))
    for f in frugal.values():f.close()
    for p in owned:lib.window95_destroy(p)
    save('partial.json',dict(rows=rows,timings=timings,setups=setups))
    print('case',j,'complete',flush=True)
save('summary.json',dict(rows=rows,timings=timings,setups=setups,hardware_n=int(hist.sum()),hardware_radius=radius))
save('manifest.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()})
