"""Warm dense/sparse timings, exact replay, destructive two-photon interference and random phases."""
import argparse
import itertools
import math
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from iteration106_width_v1 import case
from iteration106_sample_v1 import benchmark
from iteration108_low_bench_v1 import weak
from iteration108_compact_v1 import Compact


def expand(rows):
    x=np.zeros((len(rows['offsets'])-1,int(rows['width'])),np.uint32)
    i=np.repeat(np.arange(len(x)),np.diff(rows['offsets']))
    x[i,rows['ports']]=rows.get('counts',1)
    return x


def distribution(t,r,degree):
    # Expand product of linear optical creation operators instead of matching
    # output photons. Every coefficient includes the source Fock normalization.
    m=len(t);amplitude={};lam=np.tanh(r[0]);vac=np.prod(1-lam*lam)
    for n in range(degree+1):
        poly={(0,)*m:lam[0]**n*lam[1]**(degree-n)/(math.factorial(n)*math.factorial(degree-n))}
        for col in [0,1]*n+[2,3]*(degree-n):
            new={}
            for key,z in poly.items():
                for j in range(m):
                    idx=list(key);idx[j]+=1;idx=tuple(idx)
                    new[idx]=new.get(idx,0.)+z*t[j,col]
            poly=new
        for key,z in poly.items():amplitude[key]=amplitude.get(key,0.)+z
    return {k:vac*abs(v)**2*math.prod(math.factorial(n) for n in k) for k,v in amplitude.items()}


def controls(out):
    u=np.array([[1,1j],[1j,1]])/np.sqrt(2);r=np.array([[.54]])
    model=Compact(u,r,4,2048);x=model.draw(131072,True,108053000)
    subset=x[x.sum(1)==2];assert len(subset)>10000 and not np.any((subset[:,0]==1)&(subset[:,1]==1))
    rg=np.random.default_rng(108053100);u=np.linalg.qr(rg.normal(size=(4,4))+1j*rg.normal(size=(4,4)))[0]
    r=np.array([[.52,-.68]]);model=Compact(u,r,4,2048)
    n=1048576;high=np.zeros(7**4,np.int64);trial=np.zeros_like(high)
    ys=model.draw(32768,True,108054001)
    for off in range(0,n,8192):
        x=model.draw(8192,True,108054000,off)
        obs=x[x.sum(1)<=6];high+=np.bincount(obs@(7**np.arange(4)),minlength=len(high))
    ys=ys[ys.sum(1)<=6];trial+=np.bincount(ys@(7**np.arange(4)),minlength=len(high))
    q=np.zeros(len(high))
    for j in range(4):
        for idx,w in distribution(u,r,j).items():q[np.array(idx)@(7**np.arange(4))]=w
    allowed=np.flatnonzero(q>0);gap=abs(high/n-q);event=(trial/32768>q)&(q>0)
    # These low emitted sectors are identified exactly by total photon number
    # on a lossless four-port interferometer, even with a high-sector fallback.
    assert abs(q.sum()-sum(model.cert['spectra'][0]['mass'][:4]))<1e-12
    event_gap=float(abs(high[event].sum()/n-q[event].sum()))
    assert event_gap<.003
    np.savez_compressed(out/'independent.npz',optical=q,observed=high,trial=trial)
    return dict(hom_two_photon_records=len(subset),hom_coincidences=0,records=n,fock_states=len(allowed),frozen_event_gap=event_gap,
                low_sector_empirical_half_l1=float(gap.sum()/2))


def run(out):
    out.mkdir(exist_ok=False,parents=True);checks=controls(out)
    rows=[];prev=None
    for i,(m,k,level) in enumerate(((16,2,0),(4096,64,.12),(131072,2,.12),(131072,128,.12),(131072,128,.015))):
        if prev!=(m,k):t,initial=case(m,k)
        prev=(m,k);r=weak(initial,level) if level else initial
        start=perf_counter();model=Compact(t,r);setup=perf_counter()-start
        if i==1:np.savez_compressed(out/'weak4096.npz',transfer=t,squeezing=r)
        n=min(2048,4194304//m);timings=[]
        for pnr in (False,True):
            for sparse,batch in [(False,n),(True,n),(True,8192)]:
                ts=benchmark(lambda b,s,o:model.draw(b,pnr,s,o,sparse),batch,108051000+i*20+2*pnr+sparse,11)
                timings.append(dict(sparse=sparse,pnr=pnr,batch=batch,us=ts['us'],quartiles=ts['quartiles']))
            a=model.draw(127,pnr,108052000+i,11)
            b=model.draw(127,pnr,108052000+i,11,True,slots=1)
            assert np.array_equal(a,expand(b)) and np.array_equal(a[23:],model.draw(104,pnr,108052000+i,34))
        sparse=model.draw(32768,True,108055000+i,sparse=True)
        avg=(sparse['ports'].nbytes+sparse['offsets'].nbytes+sparse['counts'].nbytes)/32768
        rows.append(dict(m=m,k=k,mean_emitted_pairs=float((np.sinh(r)**2).sum()),setup_s=setup,cert=model.cert,
                         timing=timings,average_sparse_pnr_bytes=avg,single_pair_zero=float(model.single[0,0]),
                         single_pair_one=float(model.single[0,1]-model.single[0,0])))
        print(m,k,level,[(round(x['us'],3),x['batch']) for x in timings],flush=True);del model
    save(out,'summary.json',dict(rows=rows,controls=checks));manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=4,user_api='blas'):run(p.parse_args().output.resolve())
