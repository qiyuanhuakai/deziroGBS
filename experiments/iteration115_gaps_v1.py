"""Direct next-click sampling from prefix vacuum probabilities, with an explicit click-count cap."""
import argparse
import ctypes as ct
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration115_visible_v1 import ROOT,DATA,P,I,U,J,hardware,gauss,energy_law
from iteration114_rank_v1 import optical,cov
from iteration112_cases_v1 import optics,gaussian_click,oracle,ONE


def loadlib(out):
    dll=out/('gap.dll' if os.name=='nt' else 'gap.so')
    subprocess.run(['g++','-O3','-std=c++17','-fopenmp','-shared','-fPIC',
                    str(ROOT/'native/click_gap115_v1.cpp'),'-lquadmath','-o',str(dll)],check=True)
    if os.name=='nt':loadlib.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
    lib=ct.CDLL(str(dll.resolve()))
    lib.prefix115.argtypes=[I,I,P,P,P,P]
    lib.gap115.argtypes=[I]*6+[U,J]+[P]*5
    lib.gapcheck115.argtypes=[I]*5+[P]*5
    return lib


def prepare(lib,t,ss):
    m,r=t.shape;d=2*r
    basis=np.ascontiguousarray(np.stack([np.column_stack((t.real,-t.imag)),
                                         np.column_stack((t.imag,t.real))],axis=1))
    history=np.empty((len(ss),m+1,d,d))
    logs=np.empty((len(ss),m+1))
    for j,s in enumerate(ss):
        diag=np.expm1(np.r_[-2*s,2*s])/2
        assert lib.prefix115(m,d,basis.ctypes.data,diag.ctypes.data,history[j].ctypes.data,logs[j].ctypes.data)==0
    return [basis,history,np.exp(logs.sum(0))]


def query(lib,t,ss,args,sites,end):
    v=np.array(sites,np.int32);val=np.zeros(1)
    assert lib.gapcheck115(len(t),2*t.shape[1],len(ss),len(v),end,v.ctypes.data,
                           *[a.ctypes.data for a in args],val.ctypes.data)==0
    return val[0]


def capped(law,cap):
    keys=np.arange(len(law))
    mapping=[]
    for k in keys:
        keep=0;bits=int(k)
        for j in range(cap):
            if not bits:break
            low=bits&-bits;keep|=low;bits-=low
        mapping.append(keep)
    return np.bincount(mapping,weights=law,minlength=len(law))


def run_case(lib,out,name,t,ss,stop,shots,seed):
    m,r=t.shape
    start=perf_counter();args=prepare(lib,t,ss);setup=perf_counter()-start
    # The total is two independent spectral sums. Coefficients through stop
    # suffice for an upper bound on P(more than stop threshold clicks).
    mass=np.ones(1)
    for s in ss:
        p,tail=energy_law(gauss(t,s)[-1],stop)
        mass=np.convolve(mass,p)[:stop+1]
    budget=max(0.,1-mass.sum())
    vals=dict();checks={}
    if m<=16:
        truth=gaussian_click(oracle(True),np.stack([cov(t,s) for s in ss]),ONE)
        law=capped(truth,stop)
        tail=float(sum(p for i,p in enumerate(truth) if i.bit_count()>stop))
        assert abs(abs(law-truth).sum()/2-tail)<1e-11
        checks.update(exact_click_tv=tail)
        masks=np.arange(len(law));rng=np.random.default_rng(seed)
        errors=[]
        for j in range(50):
            end=int(rng.integers(1,m+1));sites=np.sort(rng.choice(end,size=int(rng.integers(min(end,stop,5)+1)),replace=False))
            key=sum(1<<int(i) for i in sites);base=(1<<end)-1
            ref=truth[(masks&base)==key].sum()
            errors.append(abs(ref-query(lib,t,ss,args,sites,end)))
        assert max(errors)<1e-10
        vals.update(truth=truth,law=law);checks['max_prefix_probability_error']=max(errors)
    batch=128 if m<=1152 else 16
    buf=np.empty((batch,m),np.uint8);hits=np.zeros(1,np.uint64);times=[];capped_rows=0;hist=np.zeros(stop+1,int)
    full=np.zeros(1<<m,int) if m<=16 else None
    for off in range(-batch,shots,batch):
        n=min(batch,shots-off);v=buf[:n]
        start=perf_counter()
        rc=lib.gap115(n,m,2*r,len(ss),stop,8,seed,2**31 if off<0 else off,
                      *[a.ctypes.data for a in args],v.ctypes.data,hits.ctypes.data)
        assert rc==0,(name,off,rc)
        if off<0:continue
        times.append(1e6*(perf_counter()-start)/n)
        capped_rows+=int(hits[0]);hist+=np.bincount(v.sum(1).astype(int),minlength=stop+1)
        if full is not None:full+=np.bincount(v@(1<<np.arange(m)),minlength=len(full))
        if off==0:vals['sample']=v[:8].copy()
    if full is not None:
        vals['hist_click']=full;checks['empirical_click_tv']=float(abs(full/shots-vals['truth']).sum()/2)
    info=dict(name=name,ports=m,input_axes=r,spectra=len(ss),max_clicks=stop,shots=shots,seed=seed,
              setup_s=setup,bytes=sum(a.nbytes for a in args),median_us=float(np.median(times)),
              mean_us=float(np.mean(times)),times=times,full_tv_upper=budget,
              reached_cap=capped_rows,probability_vacuum=float(args[-1][-1]),**checks)
    np.savez_compressed(out/(name+'.npz'),count_hist=hist,**vals)
    (out/(name+'.json')).write_text(json.dumps(info,indent=2)+'\n',encoding='utf8')
    print(name,budget,info.get('exact_click_tv'),info['median_us'],info['mean_us'],flush=True)


def run(out):
    out.mkdir(exist_ok=True,parents=True);lib=loadlib(out)
    for r in (2,4):
        t,s=optical(13,r)
        run_case(lib,out,f'rank{r}',t,s[None],6,8192,115110000+r)
    t,s=optics(4,return_transfer=True)
    run_case(lib,out,'four_phase',t[:13],s[None],6,4096,115120006)
    raw,ss=hardware()
    weak=np.arcsinh(np.sqrt(1.5/(np.sinh(ss)**2).sum())*np.sinh(ss))
    run_case(lib,out,'weak',raw,weak,6,4096,115130006)
    t,s=optical(131072,2)
    run_case(lib,out,'wide',t,s[None],4,2048,115140004)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=2):run(p.parse_args().output)
