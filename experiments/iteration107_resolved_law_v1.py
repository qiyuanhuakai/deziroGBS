"""Integrate the deployed per-detector photon corrections across matched and split spatial pairs."""
import argparse
import math
from pathlib import Path
import numpy as np
from scipy.stats import binom
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import manifest,save,occupancy_prior
from iteration100_projection_v1 import physical
from iteration105_jiuzhang_v1 import inputs
from iteration106_field_v1 import actual_input,factors
from iteration106_packed_v1 import compress
from iteration106_sample_v1 import DATA,read
from iteration107_projection_v1 import joint,PREV
from iteration107_tables_v1 import LIMIT,OUT as TABLES
from iteration107_patterns_v1 import OUT as PATTERNS


def response(z,patterns,g,site,caps,port=None):
    pair=z['pairs'][g];side=list(pair).index(site)
    dims=np.maximum(caps,LIMIT);n=np.arange(dims[side])
    if port is None:
        default=occupancy_prior(z['weights'][site],len(n)-1)
    else:
        prior=binom.pmf(np.arange(12)[None,:],n[:,None],z['weights'][site,port])
        default=np.c_[prior,(1-prior.sum(1)).clip(0)]
    tensor=np.broadcast_to(default[:,None,:] if side==0 else default[None,:,:],(*dims,default.shape[1])).copy()
    words=patterns['codes'];digits=((words[:,None]>>(4*np.arange(16,dtype=np.uint64)))&15).astype(int)
    ix=digits[:,8*side:8*side+8]
    labels=(ix>0)@(1<<np.arange(8)) if port is None else ix[:,port]
    factorial=np.array([math.factorial(k) for k in range(6)])
    for a,b in zip(*np.nonzero(patterns['offset']>=0)):
        start=patterns['offset'][a,b];length=patterns['sizes'][a]*patterns['sizes'][b]
        sl=slice(start,start+length);d=digits[sl]
        multi=factorial[a]*factorial[b]*np.prod(z['weights'][pair].ravel()**d,axis=1)/np.prod(factorial[d],axis=1)
        rate=patterns['move'][g,sl].astype(float)
        replacement=np.diff(patterns['cdf'][g,sl],prepend=np.uint64(0))/2**52
        fixed=multi*(1-rate)+(multi@rate)*replacement
        tensor[a,b]=np.bincount(labels[sl],weights=fixed,minlength=default.shape[1])
    count_rate=z['pnr_move'][g].reshape(LIMIT,LIMIT).T
    excess=np.diff(z['pnr_cdf'][g],prepend=np.uint64(0)).reshape(LIMIT,LIMIT).T/2**52
    after=np.einsum('ij,ijk->k',excess,tensor[:LIMIT,:LIMIT])
    tensor[:LIMIT,:LIMIT]+=count_rate[:,:,None]*(after-tensor[:LIMIT,:LIMIT])
    assert abs(tensor.sum(2)-1).max()<1e-11
    return tensor[:caps[0],:caps[1]].reshape(-1,default.shape[1])


def chernoff(e,cap):
    scale=e.max()
    if scale<=0:return 0.
    dx=np.linspace(.015,.995,150)/scale
    logm=-.5*np.log1p(-dx[:,None]*e).sum(1)
    return float(np.exp(np.min(logm-cap*np.log1p(dx))))


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    t,r=actual_input();cfg=read(PREV/'summary.json');rr=np.array(cfg['squeezing']);profile=cfg['profiles']['1']
    mo,w,_=compress(t,(np.sinh(r)**2).sum(0),1)
    mo*=np.sqrt((abs(w[:,0])**2).reshape(-1,8).sum(1))[:,None]
    z=dict(np.load(TABLES/'tables.npz'));p=dict(np.load(PATTERNS/'tables.npz'))
    loc={site:g for g,pair in enumerate(z['pairs']) for site in pair}
    full=inputs()[0];true_tail=[];base_tail=[]
    for j in range(144):
        cov=physical(full,8*j+np.arange(8))-np.eye(16)[None,:,:]
        e=np.concatenate(np.linalg.eigvalsh(cov));true_tail.append(chernoff(e,LIMIT))
        e=np.concatenate([np.linalg.eigvalsh(2*f.T@f) for f in factors(mo[[j]],rr)])
        base_tail.append(sum(v*chernoff(e*gain,LIMIT) for gain,v in zip(profile['values'],profile['weights'])))
    rows=[]
    for i,j,num in [(0,1,0),(36,107,1)]:
        a,b=loc[i],loc[j];groups=[*z['pairs'][a],*z['pairs'][b]]
        f=factors(mo[groups],rr);raw,caps,neg=joint(np.stack([2*v.T@v for v in f]),profile,2e-10)
        mat=raw.reshape(caps[0]*caps[1],-1)
        left=response(z,p,a,i,caps[:2]);right=response(z,p,b,j,caps[2:])
        obs=(right.T@(mat.T@left)).ravel()
        with np.load(PREV/f'panel_{num}.npz') as ref:truth=ref['target']
        row=dict(groups=[i,j],resolved_pnr_click_tv=float(abs(obs-truth).sum()/2),fft_tail=2e-10)
        np.savez_compressed(out/f'panel_{num}.npz',model=obs,target=truth)
        if num==1:
            left=response(z,p,a,i,caps[:2],1);right=response(z,p,b,j,caps[2:],5)
            pn=(right.T@(mat.T@left)).ravel()
            with np.load(DATA/'iteration107_projection_v1/pnr.npz') as ref:target=ref['target']
            row['two_detector_pnr_tv']=float(abs(pn-target).sum()/2)
            np.savez_compressed(out/'pnr.npz',model=pn,target=target)
        rows.append(row)
    save(out,'summary.json',dict(rows=rows,max_optical_pair_overflow=sum(sorted(true_tail)[-2:]),max_base_pair_overflow=sum(sorted(base_tail)[-2:]),cutoff=LIMIT))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(p.parse_args().output.resolve())
