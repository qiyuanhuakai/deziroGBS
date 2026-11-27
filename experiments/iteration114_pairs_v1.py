"""Pair-matching rejection at arbitrary source rank, including the 1,152 by 50 measured transfer."""
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
from iteration106_field_v1 import actual_input
from iteration112_cases_v1 import optics
from iteration114_rank_v1 import ROOT,BLOCK,P,I,U,J,blockgram,physical_total,cov,pair_reference,numlaw


def sector(s,cap):
    with mp.workdps(72):
        a=[mp.mpf(1)]+[mp.mpf(0)]*(cap//2)
        for val in s:
            q=[1/mp.cosh(val)]
            for k in range(1,len(a)):q.append(q[-1]*mp.tanh(val)**2*(2*k-1)/(2*k))
            a=[mp.fsum(a[i]*q[k-i] for i in range(k+1)) for k in range(len(a))]
        return float(mp.fsum(a)),np.array(list(map(float,a)))


def prepare(t,ss,cap):
    r=t.shape[1]
    val,u=eigh(np.eye(r)-t.conj().T@t)
    assert val.min()>-1e-12
    all_t=np.ascontiguousarray(np.vstack([t,np.sqrt(val.clip(0))[:,None]*u.conj().T]))
    prefix=blockgram(all_t)
    assert np.max(abs(prefix[-1]-np.eye(r)))<2e-11
    spectra=[];retained=1;attempts=0
    for s in ss:
        cdf,tail=numlaw(s,cap)
        mass,p=sector(s,cap)
        assert abs(1-mass-tail)<1e-14
        lam=np.tanh(s);s2=lam@lam
        first=np.cumsum((abs(all_t)**2)@(lam**2)/s2);first/=first[-1];first[-1]=1.
        acc=[]
        for k,prob in enumerate(p):
            d=math.prod(range(1,2*k,2))
            acc.append(math.factorial(2*k)*prob/(p[0]*d*d*s2**k))
        attempts+=sum(v/a for v,a in zip(p[1:]/mass,acc[1:]))
        spectra.append([cdf,all_t,lam,prefix,first])
        retained*=mass
    return spectra,retained,attempts


def run_case(lib,out,label,t,ss,cap,seed,shots,full_reference=None):
    start=perf_counter();spectra,retained,predicted=prepare(t,ss,cap)
    setup=perf_counter()-start
    m,r=t.shape
    total=np.ones(1)
    for s in ss:total=np.convolve(total,physical_total(t,s))
    idx=[0,3] if m==13 else [272,280]
    ref=pair_reference(np.stack([np.eye(4)+cov(t[idx],s) for s in ss]),8)
    # pair_reference is the product-of-spectra diagonal Gaussian characteristic.
    expected_mean=sum(float(np.sum((abs(t)**2)@(np.sinh(s)**2))) for s in ss)
    assert abs(total@np.arange(len(total))-expected_mean)<1e-10
    batch=128
    spectra_out=np.empty((len(ss),batch,m),np.uint8);combined=np.empty((batch,m),np.uint8)
    counts=np.zeros(2,np.uint64);times=[]
    th=np.zeros(len(total),int);ph=np.zeros(81,int);hh=np.zeros(8192,int)
    tries=0;save_data={}
    for off in range(-batch,shots,batch):
        n=min(batch,shots-off)
        start=perf_counter()
        for j,args in enumerate(spectra):
            rc=lib.pairs114(n,m,m+r,r,cap,BLOCK,8,seed,2**31 if off<0 else off,j,
                            *[a.ctypes.data for a in args],spectra_out[j].ctypes.data,counts.ctypes.data)
            assert rc==0,(label,rc)
            if off>=0:tries+=int(counts[0])
        if len(ss)==1:records=spectra_out[0,:n]
        else:
            np.add(spectra_out[0,:n],spectra_out[1,:n],out=combined[:n]);records=combined[:n]
        elapsed=perf_counter()-start
        if off<0:continue
        times.append(1e6*elapsed/n)
        if off==0:save_data['sample']=records[:16].copy()
        th+=np.bincount(records.sum(1).astype(int),minlength=len(total))
        ph+=np.bincount(np.minimum(records[:,idx[0]],8).astype(int)+9*np.minimum(records[:,idx[1]],8).astype(int),minlength=81)
        if m==13:hh+=np.bincount((records>0)@(1<<np.arange(13)),minlength=8192)
    summary=dict(label=label,ports=m,source_axes=r,spectra=len(ss),cap_per_spectrum=cap,seed=seed,
        shots=shots,setup_s=setup,median_us=float(np.median(times)),mean_us=float(np.mean(times)),times=times,
        pnr_truncation_tv_upper=1-retained,total_ks=float(abs((th/shots-total).cumsum()).max()),
        pair_empirical_tv=float(abs(ph/shots-ref).sum()/2),optical_visible_mean=expected_mean,
        attempts_per_shot=tries/shots,expected_attempts_per_shot=predicted)
    if full_reference is not None:
        summary['empirical_click_tv']=float(abs(hh/shots-full_reference).sum()/2)
    np.savez_compressed(out/(label+'.npz'),total=total,pair=ref,total_hist=th,pair_hist=ph,click_hist=hh,s=ss,**save_data)
    (out/(label+'.json')).write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print(label,summary['median_us'],summary['mean_us'],1-retained,summary['total_ks'],flush=True)
    return summary


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    dll=out/('pair.dll' if os.name=='nt' else 'pair.so')
    subprocess.run(['g++','-O3','-fopenmp','-shared','-fPIC',str(ROOT/'native/pair_emission114_v1.cpp'),'-o',str(dll)],check=True)
    runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent)) if os.name=='nt' else None
    lib=ct.CDLL(str(dll.resolve()))
    lib.pairs114.argtypes=[I]*7+[U,J,I]+[P]*7
    original,s=optics(4,return_transfer=True)
    truth=np.load(ROOT/'results/iteration112_four_peers_v1/peers.npz')['truth']
    for cap in (8,10):
        run_case(lib,out,'four_phase_'+str(cap),original[:13],s[None],cap,114810000+cap,8192,truth)
    raw,r=actual_input()
    # Independent two-mode squeezed sources converted to single-mode squeezed axes.
    t=np.empty_like(raw)
    t[:,::2]=1j*(raw[:,::2]+raw[:,1::2])/np.sqrt(2)
    t[:,1::2]=(raw[:,::2]-raw[:,1::2])/np.sqrt(2)
    s=r.repeat(2,axis=1)
    from iteration107_cliques_v1 import optical_x
    ids=[0,7,272,280,446,876,1100]
    exact=optical_x(raw[ids],r)
    np.testing.assert_allclose([cov(t[ids],row) for row in s],exact,atol=1e-12)
    mean=float((np.sinh(s)**2).sum())
    scale=np.sqrt(1.5/mean)
    weak=np.arcsinh(scale*np.sinh(s))
    summary=run_case(lib,out,'jiuzhang_transfer_weak',t,weak,8,114850008,16384)
    summary.update(original_mean_emitted=mean,weak_mean_emitted=float((np.sinh(weak)**2).sum()),
        amplitude_scale=scale,original_retained_at_cap=float(np.prod([sector(v,8)[0] for v in s])))
    (out/'jiuzhang_transfer_weak.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=2):run(p.parse_args().output)
