"""Loss-aware mixed Gaussian fixed-number sectors on the actually observed outputs."""
import argparse
import ctypes as ct
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import mpmath as mp
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from iteration114_rank_v1 import ROOT,P,I,U,J,cov,optical,physical_total
from iteration114_pairs_v1 import sector
from iteration112_cases_v1 import optics,pair_reference
from iteration106_field_v1 import actual_input

DATA=ROOT/'results'


def normalized(v):
    weights=np.asarray(v,float)
    total=weights.sum(-1,keepdims=True)
    safe=np.divide(weights,total,out=np.full_like(weights,1/weights.shape[-1]),where=total>0)
    out=np.cumsum(safe,axis=-1)
    out[...,-1]=1
    return np.ascontiguousarray(out)


def gauss(t,s):
    q,r=np.linalg.qr(t,mode='reduced')
    x=cov(r,s);eye=np.eye(len(x));k=eye-np.linalg.inv(eye+x)
    k=(k+k.T)/2
    d=len(r);j=np.vstack((np.eye(d),1j*np.eye(d)))/np.sqrt(2)
    a=j.T@k@j;c=j.T@k@j.conj()
    val,axes=eigh(c)
    assert val.min()>-2e-12
    return q,a,val.clip(0),axes,np.linalg.eigvalsh(x)


def energy_law(e,cap):
    with mp.workdps(65):
        v=list(map(mp.mpf,e))
        vacuum=mp.exp(-sum(mp.log1p(y) for y in v)/2)
        ev=[y/(1+y) for y in v];p=[vacuum]
        trace=[0]+[mp.fsum(y**k for y in ev) for k in range(1,cap+1)]
        for n in range(1,cap+1):
            p.append(mp.fsum(trace[k]*p[n-k] for k in range(1,n+1))/(2*n))
        tail=1-mp.fsum(p)
        assert min(p)>-mp.mpf('1e-15') and tail>=-mp.mpf('1e-15')
    return np.maximum(0,np.array(list(map(float,p)))),max(0.,float(tail))


def prepare(t,s,cap):
    q,a,val,axis,e=gauss(t,s)
    pair=np.ascontiguousarray((q@a)@q.T)
    mode=np.ascontiguousarray((q@axis).T)
    sq=abs(pair)**2
    p,tail=energy_law(e,cap)
    s2=sq.sum();tr=val.sum()
    table=np.zeros((cap+1,cap//2+1));attempts=0.;accept=[]
    for n in range(cap+1):
        for k in range(n//2+1):
            l=n-2*k
            d=math.factorial(n)/(2**k*math.factorial(k))
            table[n,k]=d*d*s2**k*tr**l/math.factorial(l)
        envelope=table[n].sum()
        alpha=math.factorial(n)*p[n]/(p[0]*envelope) if envelope>0 else 1.
        accept.append(alpha)
        if n and p[n]>0:attempts+=p[0]*envelope/math.factorial(n)/(1-tail)
    assert min(accept)>=0 and max(accept)<1+5e-11
    args=[normalized(p),normalized(table),normalized(sq.sum(1)),normalized(sq),
          pair,normalized(val),normalized(abs(mode)**2),mode]
    return args,dict(cap=cap,tail=tail,p=p.tolist(),accept=accept,attempts=attempts,
                     mean_visible=float(e.sum()/2),cross_trace=float(tr),pair_squared_norm=float(s2))


def hardware():
    raw,r=actual_input()
    t=np.empty_like(raw);t[:,::2]=1j*(raw[:,::2]+raw[:,1::2])/np.sqrt(2)
    t[:,1::2]=(raw[:,::2]-raw[:,1::2])/np.sqrt(2)
    return t,r.repeat(2,axis=1)


def run_case(lib,out,name,t,ss,cap,shots,seed):
    start=perf_counter()
    terms=[prepare(t,s,cap) for s in ss]
    setup=perf_counter()-start
    retained=math.prod(1-v['tail'] for _,v in terms)
    tail=-math.expm1(sum(math.log1p(-v['tail']) for _,v in terms))
    total=np.ones(1)
    for s in ss:total=np.convolve(total,physical_total(t,s))
    m,r=t.shape;idx=[0,3] if m==13 else [272,280]
    ref=pair_reference(np.stack([np.eye(4)+cov(t[idx],s) for s in ss]),8)
    batch=128
    scratch=np.empty((len(ss),batch,m),np.uint8);buf=np.empty((batch,m),np.uint8)
    hist=np.zeros(len(total),int);pair=np.zeros(81,int);bits=np.zeros(8192,int) if m==13 else None
    counters=np.zeros(1,np.uint64);tries=0;times=[];records={}
    for off in range(-batch,shots,batch):
        n=min(batch,shots-off);start=perf_counter()
        for j,(args,_) in enumerate(terms):
            rc=lib.visible115(n,m,len(args[5]),cap,8,seed,2**31 if off<0 else off,j,*[a.ctypes.data for a in args],
                             scratch[j].ctypes.data,counters.ctypes.data)
            assert rc==0,(name,rc,off)
            if off>=0:tries+=int(counters[0])
        if len(ss)==2:
            np.add(scratch[0,:n],scratch[1,:n],out=buf[:n]);v=buf[:n]
        else:v=scratch[0,:n]
        elapsed=perf_counter()-start
        if off<0:continue
        times.append(elapsed*1e6/n)
        hist+=np.bincount(v.sum(1).astype(int),minlength=len(total))
        pair+=np.bincount(np.minimum(v[:,idx[0]],8).astype(int)+9*np.minimum(v[:,idx[1]],8).astype(int),minlength=81)
        if bits is not None:bits+=np.bincount((v>0)@(1<<np.arange(m)),minlength=8192)
        if off==0:records['sample']=v[:32].copy()
    info=dict(name=name,m=m,r=r,spectra=len(ss),cap=cap,shots=shots,seed=seed,setup_s=setup,full_tv_tail=tail,
              median_us=float(np.median(times)),mean_us=float(np.mean(times)),times=times,
              observed_attempts_per_shot=tries/shots,expected_attempts_per_shot=sum(q['attempts'] for _,q in terms),
              pair_empirical_tv=float(abs(pair/shots-ref).sum()/2),total_empirical_ks=float(abs((hist/shots-total).cumsum()).max()),
              sectors=[v for _,v in terms])
    if bits is not None:records['hist_click']=bits
    np.savez_compressed(out/(name+'.npz'),total=total,pair=ref,hist_total=hist,hist_pair=pair,gains=ss,**records)
    (out/(name+'.json')).write_text(json.dumps(info,indent=2)+'\n',encoding='utf8')
    print(name,tail,info['median_us'],info['mean_us'],tries/shots,info['expected_attempts_per_shot'],flush=True)


def loadlib(out):
    dll=out/('visible.dll' if os.name=='nt' else 'visible.so')
    subprocess.run(['g++','-O3','-std=c++17','-fopenmp','-fPIC','-shared',str(ROOT/'native/visible_sector115_v1.cpp'),'-o',str(dll)],check=True)
    if os.name=='nt':loadlib.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
    lib=ct.CDLL(str(dll.resolve()));lib.visible115.argtypes=[I]*5+[U,J,I]+[P]*10
    return lib


def run(out):
    out.mkdir(exist_ok=True,parents=True);lib=loadlib(out)
    for r in (2,4):
        t,s=optical(13,r)
        run_case(lib,out,f'rank{r}',t,s[None],8,8192,115010000+r)
    original,s=optics(4,return_transfer=True);t=original[:13]
    for cap in (8,10):
        run_case(lib,out,f'four_phase_{cap}',t,s[None],cap,8192,115020000+cap)
    t,ss=hardware();scale=np.sqrt(1.5/(np.sinh(ss)**2).sum())
    weak=np.arcsinh(scale*np.sinh(ss))
    run_case(lib,out,'weak',t,weak,8,16384,115030008)
    coverage=[]
    for s in ss:
        _,_,_,_,e=gauss(t,s);p,tail=energy_law(e,360)
        coverage.append(dict(per_spectrum_mean=float(e.sum()/2),probability_at_most_8=float(p[:9].sum()),
                             emitted_at_most_8=sector(s,8)[0],outside_360=tail,p=p.tolist()))
    total=np.convolve(*[row['p'] for row in coverage])
    summary=dict(spectra=coverage,retained_per_spectrum_cap8=math.prod(v['probability_at_most_8'] for v in coverage),
                 emitted_retained_per_spectrum_cap8=math.prod(v['emitted_at_most_8'] for v in coverage),
                 total_quantiles=[int(np.searchsorted(np.cumsum(total),u)) for u in (.5,.9,.99,.999)])
    (out/'high_power.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=2):run(p.parse_args().output)
