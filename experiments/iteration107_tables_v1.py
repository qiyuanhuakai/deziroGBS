"""Optical probability tables for a minimum-change coupling on a matching of spatial ports."""
import argparse
import ctypes as ct
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import networkx as nx
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,Native,bind,I,J,D,realify,occupancy_prior,split_joint,save,manifest
from iteration100_projection_v1 import physical
from iteration105_jiuzhang_v1 import inputs
from iteration106_field_v1 import actual_input,factors,BASE,DATA
from iteration106_packed_v1 import compress,OUT as PREV
from iteration106_sample_v1 import read

OUT=DATA/'iteration107_tables_v1'
LIMIT=80


class Counts:
    def __init__(self,dll):
        self.runtime=Native(BASE/'spatial.dll')
        self.f=bind(ct.CDLL(str(dll)),'counts107',[I,I,D,J,D])
    def __call__(self,q,sizes):
        p=np.empty(np.prod(sizes))
        assert self.f(len(sizes),len(q),np.ascontiguousarray(q),np.array(sizes,np.int32),p)==0
        assert np.isfinite(p).all() and p.min()>-3e-14,(p.min(),p.sum())
        return p.clip(0)


def kernel(p,q):
    p=p.clip(0);q=q.clip(0);p=p/p.sum();q=q/q.sum()
    delta=q-p;lost=np.maximum(-delta,0);positive=np.maximum(delta,0)
    tv=float(positive.sum());assert abs(tv-lost.sum())<2e-11
    threshold=np.divide(lost,p,out=np.zeros_like(p),where=p>0).astype(np.float32)
    if tv:replacement=positive/tv
    else:replacement=np.r_[1.,np.zeros(len(q)-1)]
    cdf=np.rint(np.cumsum(replacement,dtype=np.longdouble)*2**52).astype(np.uint64)
    cdf=np.minimum(cdf,2**52);cdf[-1]=2**52
    realized=p*(1-threshold.astype(float))+(p@threshold)*np.diff(cdf,prepend=np.uint64(0))/2**52
    assert abs(realized.sum()-1)<1e-12
    return threshold,cdf,tv,float(abs(realized-q).sum()/2)


def matching(q,base,profile,lib):
    m=q.shape[1]//2
    def projection(x,values,weights):
        a=np.empty(m);b=np.empty((m,m))
        assert lib(m,len(x),len(values),16,np.ascontiguousarray(x),np.array(values),np.array(weights),a,b)==0
        return b-a[:,None]*a
    # Compare vacuum correlations under three detector efficiencies. Single-site
    # marginal errors affect every partner and are removed before edge ranking.
    weights=np.zeros((m,m))
    for eta in (.25,.5,1.):
        weights+=abs(projection(eta*q,[1.],[1.])-projection(eta*base,profile['values'],profile['weights']))
    G=nx.from_numpy_array(weights);G.remove_edges_from(nx.selfloop_edges(G))
    pairs=sorted(tuple(sorted(e)) for e in nx.max_weight_matching(G,maxcardinality=True))
    assert len(pairs)*2==m and len(set(sum((list(p) for p in pairs),[])))==m
    return pairs,weights


def run(out):
    out.mkdir(parents=True,exist_ok=False);dll=out/'counts.dll'
    eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'
    subprocess.run([shutil.which('g++'),'-std=c++20','-O3','-march=x86-64-v3','-ffp-contract=off','-shared','-I',str(eigen),str(ROOT/'native/local_counts107_v1.cpp'),'-o',str(dll)],check=True)
    count=Counts(dll);z,_,_=inputs();t,r=actual_input()
    mo,w,_=compress(t,(np.sinh(r)**2).sum(0),1)
    weight=(abs(w[:,0])**2).reshape(-1,8);power=weight.sum(1)
    w=weight/power[:,None];mo=mo*np.sqrt(power[:,None])
    config=read(PREV/'summary.json');rr=np.array(config['squeezing']);prof=config['profiles']['1']
    base=np.stack([2*f.T@f for f in factors(mo,rr)])
    # Each compressed spatial mode uses the optical transfer projected onto
    # its dominant output direction. Use the full sixteen physical rows below.
    from iteration106_weak_v1 import covariance
    q=np.stack([covariance(np.stack([mo[:,2*j:2*j+2] for j in range(len(row))]),row,np.arange(len(mo)))-np.eye(2*len(mo)) for row in r])
    census=bind(ct.CDLL(str(PREV/'pairs.dll')),'census106',[I,I,I,I,D,D,D,D,D])
    pairs,edges=matching(q,base,prof,census);pairs=np.array(pairs,np.int32)
    oracle=bind(ct.CDLL(str(DATA/'iteration105_source_groups_v1/extended.dll')),'oracle96',[I,I,D,D,D,D])
    high=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    rows=[];priors=[occupancy_prior(weight,LIMIT-1) for weight in w]
    v=z['v'][np.argsort(z['order'])];bs=[realify(v[8*g:8*g+8]) for g in range(144)]
    gram=[b.T@b for b in bs]
    click_move=[];click_cdf=[];pnr_move=[];pnr_cdf=[];counts=[];references=[]
    for j,(a,b) in enumerate(pairs):
        ix=np.array([a,b,144+a,144+b]);xb=base[:,ix][:,:,ix]
        raw=sum(vv*count(np.eye(4)[None,:,:]+g*xb,[LIMIT,LIMIT]) for g,vv in zip(prof['values'],prof['weights']))
        # First coordinate runs fastest, matching an eight-bit low byte.
        target=split_joint(count.runtime,gram[a],gram[b],z['x'],LIMIT-1,2*LIMIT-1)[:,:LIMIT].T.copy().ravel()
        assert target.min()>-1e-12 and abs(target.sum()-1)<2e-10 and abs(raw.sum()-1)<2e-10,(a,b,raw.sum(),target.sum())
        target=target.clip(0);counts.append(target)
        pm=priors[b].T@raw.reshape(LIMIT,LIMIT)@priors[a];pc=pm.ravel()
        approx=(priors[b].T@target.reshape(LIMIT,LIMIT)@priors[a]).ravel()
        ids=np.r_[np.arange(8)+8*a,np.arange(8)+8*b];cov=physical(z,ids)
        truth=np.empty(65536);assert oracle(16,2,cov,np.zeros((2,32)),truth,np.empty(2))==0
        error=0.
        if j in (0,23,48):
            check=np.empty_like(truth);assert high(16,2,cov,np.zeros((2,32)),check,np.empty(2))==0
            error=float(abs(check-truth).sum()/2);assert error<1e-9
        wc,dc,ctv,ce=kernel(pc,truth);wn,dn,ntv,ne=kernel(raw,target)
        click_move.append(wc);click_cdf.append(dc);pnr_move.append(wn);pnr_cdf.append(dn)
        if j in (0,23,48) or (a,b) in [(0,1),(34,35),(36,107)]:references.append((j,truth,pc,approx))
        rows.append(dict(pair=[int(a),int(b)],click_tv_before=ctv,click_table_tv_after=ce,pnr_count_tv_before=ntv,pnr_count_table_tv_after=ne,
                         split_click_tv_after=float(abs(approx-truth).sum()/2),base_tail=max(0.,float(1-raw.sum())),target_truncated_mass=float(target.sum()),quad_recheck=error))
        if j%12==0:print(j,rows[-1],flush=True)
    np.savez_compressed(out/'tables.npz',pairs=pairs,weights=w,click_move=click_move,click_cdf=click_cdf,pnr_move=pnr_move,pnr_cdf=pnr_cdf)
    np.savez_compressed(out/'counts.npz',target_counts=counts,matching_weights=edges)
    np.savez_compressed(out/'selected.npz',**{f'{j}_{name}':arr for j,v,b,c in references for name,arr in [('full',v),('base',b),('pnr',c)]})
    save(out,'summary.json',dict(limit=LIMIT,rows=rows,selected_by='Maximum-weight perfect matching of vacuum covariance errors at detector efficiencies .25, .5 and 1.'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):run(p.parse_args().output.resolve())
