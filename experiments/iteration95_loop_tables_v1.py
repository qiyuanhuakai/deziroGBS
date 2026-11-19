"""Optical count and pattern tables for groups of up to eight detectors."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import itertools
import math

import numpy as np
from scipy.sparse import csr_matrix

WORD=2**32

def integer_row(p):
    p=np.maximum(np.asarray(p,dtype=float),0)
    if not p.sum()>0:
        raise ValueError("empty conditional distribution")
    c=np.rint(WORD*(p/p.sum()).cumsum()).clip(0,WORD).astype(np.uint64)
    c[-1]=WORD
    return c

def discrete(c):
    return np.diff(np.r_[0.,c.astype(float)])/WORD

@lru_cache(None)
def interpolation(width,degree):
    def compositions(n,k):
        if k==1:
            yield (n,)
        else:
            for a in range(n+1):
                for rest in compositions(n-a,k-1):yield (a,*rest)
    points=[p for n in range(degree+1) for p in compositions(n,width)]
    index={p:i for i,p in enumerate(points)}
    matrices=[]
    codes=[]
    for t in range(degree+1):
        patterns=list(compositions(t,width))
        row=[];col=[];data=[]
        for i,k in enumerate(patterns):
            divisor=math.prod(math.factorial(a) for a in k)
            for v in itertools.product(*(range(a+1) for a in k)):
                row.append(i);col.append(index[v])
                data.append((-1)**(t-sum(v))*math.prod(math.comb(a,b) for a,b in zip(k,v))/divisor)
        matrices.append(csr_matrix((data,(row,col)),shape=(len(patterns),len(points))))
        codes.append(np.array([sum(a<<(4*j) for j,a in enumerate(k)) for k in patterns],np.uint32))
    return np.array(points,dtype=float),matrices,codes

def make_optics(lib,q,mu,groups,degree,cap):
    m=q.shape[1]//2
    stride=cap+2
    freq=1<<(stride-1).bit_length()
    def subset(ports):
        ix=np.r_[ports,ports+m]
        return np.ascontiguousarray(q[:,ix][:,:,ix]),np.ascontiguousarray(mu[:,ix])
    def vacuum(ports):
        qs,u=subset(ports)
        sign,log=np.linalg.slogdet(qs)
        assert np.all(sign>0)
        correction=np.einsum("si,si->",u,np.linalg.solve(qs,u[:,:,None])[:,:,0]) if u.any() else 0.
        return math.exp(-.5*(log.sum()+correction))
    def one_group(ports):
        width=len(ports)
        split=np.zeros((16,stride,stride),np.uint64)
        min_coeff=0.
        max_imag=0.
        groups_true={}
        def visit(ports,node):
            nonlocal min_coeff,max_imag
            size=len(ports)
            qs,u=subset(ports)
            left=size//2
            raw=np.empty((freq,stride),complex)
            assert lib.radial(size,left,len(q),stride-1,freq,qs,u,raw.view(float))==0
            coef=np.fft.fft(raw,axis=0)/freq
            max_imag=max(max_imag,float(abs(coef.imag).max()))
            pmf=raw[0].real
            assert abs(pmf.imag).max()<1e-10 and pmf.min()>-1e-11
            for n in range(stride):
                v=coef[:n+1,n].real
                min_coeff=min(min_coeff,float(v.min()))
                assert abs(v.sum()-pmf[n])<5e-11 and v.min()>-2e-11
                if pmf[n]>1e-18:
                    split[node,n,:n+1]=integer_row(v)
                else:
                    split[node,n,:n+1]=integer_row([math.comb(n,j)*.5**n for j in range(n+1)])
            groups_true[node]=pmf.copy()
            if size>1:
                if left>1:visit(ports[:left],2*node)
                if size-left>1:visit(ports[left:],2*node+1)
            return pmf.copy()
        root=visit(ports,1)
        qs,u=subset(ports)
        h=lib.window(width,width,0,len(q),qs,u,np.zeros(4))
        bits=np.empty(1<<width)
        assert h and lib.window_law(h,bits)==0
        lib.window_free(h)
        assert bits.min()>-1e-14 and abs(bits.sum()-1)<1e-12
        pop=np.array([k.bit_count() for k in range(len(bits))])
        click_count=np.bincount(pop,weights=bits,minlength=9)
        mask=np.zeros((9,256),np.uint64)
        for k in range(width+1):
            v=bits*(pop==k)
            mask[k,:len(bits)]=integer_row(v if v.sum()>1e-23 else pop==k)
            mask[k,len(bits):]=WORD
        points,transform,pattern=interpolation(width,degree)
        pgf=np.empty((len(points),degree+1))
        assert lib.low(width,len(q),degree,len(points),qs,u,points,pgf)==0
        cdfs=[];law=[];mass_error=0.
        for t,matrix in enumerate(transform):
            prob=matrix@pgf[:,t]
            min_coeff=min(min_coeff,float(prob.min()))
            mass_error=max(mass_error,abs(prob.sum()-root[t]))
            assert prob.min()>-3e-10 and abs(prob.sum()-root[t])<5e-9
            law.append(prob)
            cdfs.append(integer_row(prob if root[t]>1e-17 else np.ones_like(prob)))
        return dict(split=split,root=root,click=bits,click_count=click_count,mask=mask,low=cdfs,pattern=pattern,
                    true_low=law,min_coeff=min_coeff,max_imag=max_imag,mass_error=mass_error)
    # Cache shared interpolation transforms before starting worker threads.
    for size in {len(g) for g in groups}:interpolation(size,degree)
    with ThreadPoolExecutor(max_workers=12) as pool:items=list(pool.map(one_group,groups))
    single=1-np.array([item["root"][0] for item in items])
    joint=np.diag(single)
    for i,ports in enumerate(groups):
        for j in range(i):
            z=vacuum(np.r_[groups[j],ports])
            joint[i,j]=joint[j,i]=single[i]+single[j]-1+z
    return items,single,joint

def mask_given_photons(width,stride,split,degree,cdfs,codes):
    def build(w,node):
        if w==1:
            a=np.zeros((stride,2));a[0,0]=1;a[1:,1]=1
            return a
        left=w//2
        a=build(left,2*node);b=build(w-left,2*node+1)
        law=np.zeros((stride,1<<w))
        for n in range(stride):
            p=discrete(split[node,n,:n+1])
            law[n]=np.einsum("t,ti,tj->ji",p,a[:n+1],b[n::-1]).ravel()
        return law
    law=build(width,1)
    for t in range(degree+1):
        pattern=codes[t]
        mask=sum((((pattern>>(4*j))&15)>0).astype(int)<<j for j in range(width))
        law[t]=np.bincount(mask,weights=discrete(cdfs[t]),minlength=1<<width)
    assert abs(law.sum(1)-1).max()<1e-12
    return law

def fock_probabilities(counts,root,split,degree,cdfs,patterns):
    sums=counts.sum(1).astype(int)
    probability=root[sums].copy()
    def visit(states,ports,node):
        if len(ports)==1 or not len(states):return
        half=len(ports)//2
        total=counts[states][:,ports].sum(1).astype(int)
        left=counts[states][:,ports[:half]].sum(1).astype(int)
        hi=split[node,total,left].astype(float)
        lo=np.where(left>0,split[node,total,np.maximum(left-1,0)].astype(float),0.)
        probability[states]*=(hi-lo)/WORD
        visit(states,ports[:half],2*node);visit(states,ports[half:],2*node+1)
    visit(np.flatnonzero(sums>degree),np.arange(counts.shape[1]),1)
    packed=sum(counts[:,j].astype(np.uint32)<<(4*j) for j in range(counts.shape[1]))
    for t in range(degree+1):
        sel=np.flatnonzero(sums==t)
        order=np.argsort(patterns[t])
        where=np.searchsorted(patterns[t][order],packed[sel])
        probability[sel]*=discrete(cdfs[t])[order[where]]
    return probability
