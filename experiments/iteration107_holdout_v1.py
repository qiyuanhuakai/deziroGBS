"""Matched records estimate the change in joint probabilities, including panels crossing the matching."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.stats import beta
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from iteration106_sample_v1 import read
from iteration107_device_v1 import Local,DATA


def code(x,ids):
    return (x[:,ids]>0).astype(np.uint32)@(np.uint32(1)<<np.arange(len(ids),dtype=np.uint32))


def bin_interval(k,n,alpha):
    lo=0. if k==0 else beta.ppf(alpha,k,n+1-k)
    hi=1. if k==n else beta.isf(alpha,k+1,n-k)
    return float(lo),float(hi)


def bounds(base,truth,delta,changed,n,trial,positive,negative):
    # Conditional on the number of changed records, changed pairs are iid.
    # Changing one such pair changes half the L1 plug-in by at most 2/n.
    alpha=.000025
    rate=changed/n;lo,hi=bin_interval(changed,n,alpha)
    mass_error=max(hi-rate,rate-lo)
    plug=float(abs(base+delta/n-truth).sum()/2)
    upper=plug+mass_error+np.sqrt(2*changed*np.log(1/alpha))/n
    e=trial
    pl,_=bin_interval(positive,n,alpha);_,nh=bin_interval(negative,n,alpha)
    lower=max(0.,float((base[e]-truth[e]).sum())+pl-nh)
    return dict(tv_estimate=plug,lower=lower,upper=upper,changed_fraction=rate)


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    model=Local()
    stage=DATA/'iteration106_packed_v1'
    refs=[np.load(stage/f'panel_{j}.npz') for j in range(9)]
    hw=np.load(DATA/'iteration104_record_order_v1/windows.npz')['histograms'][32:].sum(0)
    full=np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf']
    timing=[]
    rng=np.random.default_rng(107010000)
    for n in (4,64,2048,16384):
        ways=[(0,0),(0,1),(1,0)]
        times=[[] for _ in ways]
        for j in range(17):
            for k in rng.permutation(3):
                p,b=ways[k]
                start=perf_counter();model.draw(n,p,107011000+int(k),j*n,packed=bool(b))
                if j>=3:times[k].append((perf_counter()-start)*1e6/n)
        for (p,b),ts in zip(ways,times):timing.append(dict(batch=n,pnr=bool(p),packed=bool(b),us=float(np.median(ts)),quartiles=np.quantile(ts,[.25,.75]).tolist(),repeats=ts))
    rows=[]
    for pnr in (False,True):
        events=[]
        for stage_no,n in enumerate((262144,4194304)):
            changes=np.zeros(len(refs),int);positive=changes.copy();negative=changes.copy()
            delta=[np.zeros_like(z['1'],dtype=np.int64) for z in refs];observed=[np.zeros_like(y) for y in delta]
            h=np.zeros(1153,np.int64);ph=np.zeros(1024,np.int64);pair=np.zeros(169,np.int64)
            for off in range(0,n,4096):
                x,raw=model.draw(4096,pnr,107012000+2*pnr+stage_no,off,paired=True)
                for j,(z,ds,hs) in enumerate(zip(refs,delta,observed)):
                    a=code(raw,z['ports']);b=code(x,z['ports'])
                    use=a!=b
                    changes[j]+=use.sum()
                    ds+=np.bincount(b[use],minlength=len(ds))-np.bincount(a[use],minlength=len(ds))
                    if stage_no:
                        hs+=np.bincount(b,minlength=len(ds))
                        aa=events[j][a[use]];bb=events[j][b[use]]
                        positive[j]+=(bb&~aa).sum();negative[j]+=(aa&~bb).sum()
                h+=np.bincount(np.count_nonzero(x,axis=1),minlength=len(h))
                if pnr:
                    ph+=np.bincount(x.sum(1).astype(np.int64),minlength=len(ph))
                    pair+=np.bincount(np.minimum(x[:,[289,861]],12)@np.array([1,13]),minlength=169)
            if not stage_no:events=[z['1']+d/n>z['target'] for z,d in zip(refs,delta)]
            else:
                panels=[]
                for j,z in enumerate(refs):
                    row=bounds(z['1'],z['target'],delta[j],int(changes[j]),n,events[j],int(positive[j]),int(negative[j]))
                    e=events[j]
                    row['raw_native_lower99']=max(0.,float(observed[j][e].sum()/n-z['target'][e].sum())-np.sqrt(np.log(1/.000025)/(2*n)))
                    row['prior_tv']=float(abs(z['1']-z['target']).sum()/2)
                    panels.append(row)
                entry=dict(pnr=pnr,records=n,panels=panels,hardware_ks=float(abs(h.cumsum()/n-hw.cumsum()/hw.sum()).max()))
                if pnr:entry['optical_total_ks']=float(abs(ph.cumsum()/n-full.cumsum()[:len(ph)]).max())
                print(pnr,entry['hardware_ks'],[(round(r['tv_estimate'],5),round(r['upper'],5)) for r in panels],flush=True)
                rows.append(entry)
                np.savez_compressed(out/f'holdout_{int(pnr)}.npz',click=h,photon=ph,pair=pair,**{f'delta_{j}':v for j,v in enumerate(delta)},**{f'observed_{j}':v for j,v in enumerate(observed)},**{f'event_{j}':v for j,v in enumerate(events)})
        a=model.draw(967,pnr,107017000+pnr,7)
        b=model.draw(900,pnr,107017000+pnr,74)
        assert np.array_equal(a[67:],b)
        if not pnr:assert np.array_equal(b,np.unpackbits(model.draw(900,False,107017000,74,True),axis=1,bitorder='little'))
    save(out,'summary.json',dict(timing=timing,runs=rows,hardware_holdout_records=int(hw.sum()),control_variate_scope='TV bounds center on the analytical base optical law plus the native paired change. Native-only lower bounds use independent recorded bits directly.'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):run(p.parse_args().output.resolve())
