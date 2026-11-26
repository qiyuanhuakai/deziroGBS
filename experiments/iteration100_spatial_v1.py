"""Gaussian photon totals on a binary tree and conditional threshold projections at its leaves."""
import argparse
import ctypes as ct
import itertools
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
import networkx as nx
import numpy as np
from threadpoolctl import threadpool_limits
from iteration96_group_common_v1 import ROOT,H,save,manifest,bind,I,D,T,J,B,W,U
from iteration96_group_benchmark_v1 import ORDER,HIST
SHORT=np.ctypeslib.ndpointer(dtype=np.uint16,flags='C_CONTIGUOUS')
S=1<<53

def realify(v):
    return np.block([[v.real,-v.imag],[v.imag,v.real]])

def capacity(eigen,eps=2e-16):
    top=max(float(eigen.max()),1e-12)
    if top<1e-10:return 8
    r=1+np.geomspace(.02,.96,40)/top
    logm=-.5*np.log1p((1-r[:,None])*eigen).sum(1)
    return max(8,int(np.ceil(np.min((logm-math.log(eps))/np.log(r)))))

def series(lib,values,p0,cap):
    v=np.ascontiguousarray(values);res=np.empty((len(v),cap+1))
    assert lib.series(len(v),v.shape[1],cap,p0,v,res)==0
    return res

def cdf_rows(rows):
    cumulative=np.maximum.accumulate(rows.cumsum(axis=-1),axis=-1)
    if np.any(cumulative[...,-1]<=0):raise ValueError('empty conditional probability table')
    c=np.rint((cumulative/cumulative[...,-1:])*S).clip(0,S).astype(np.uint64)
    return c

def realized(c):
    return np.diff(c,axis=-1,prepend=np.zeros((*c.shape[:-1],1),np.uint64)).astype(float)/S

class Native:
    def __init__(self,dll):
        self.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));self.lib=ct.CDLL(str(dll))
        self.series=bind(self.lib,'spatial100_series',[I,I,I,ct.c_double,D,D])
        self.complex=bind(self.lib,'spatial100_complex',[I,I,I,D,D,D])
        self.draw=bind(self.lib,'spatial100_draw',[I,I,J,J,J,W,W,J,W,D,D,I,I,U,U,I,SHORT,B])

def compact(gram):
    e,v=np.linalg.eigh(gram)
    keep=e>max(e[-1]*3e-15,1e-19)
    return np.sqrt(e[keep,None])*v[:,keep].T

def local_kernel(x):
    e,v=np.linalg.eigh(x)
    assert e[0]>-.5-5e-12
    return (v*(e/(1+e)))@v.T,-.5*np.log1p(e).sum(),e

def split_joint(lib,left,right,spectra,left_cap,right_cap):
    a=compact(left);b=compact(right);h=np.r_[a,b];d=len(a)
    ny=1<<right_cap.bit_length()
    y=np.exp(-2j*np.pi*np.arange(ny//2+1)/ny)
    # Avoid an artificial imaginary residual at the Nyquist endpoint.
    y[-1]=-1
    values=[];pref=np.ones(len(y),complex)
    for x in spectra:
        k,logp,_=local_kernel(h@x@h.T)
        ev,rot=np.linalg.eigh(k[d:,d:]);cross=k[:d,d:]@rot
        pref*=np.exp(logp-.5*np.log1p(-y[:,None]*ev).sum(1))
        roots=[]
        for j,t in enumerate(y):
            m=k[:d,:d]+(cross*(t/(1-t*ev)))@cross.T
            roots.append(np.linalg.eigvalsh(m.real) if j==0 or j==len(y)-1 else np.linalg.eigvals(m))
        values.append(np.array(roots))
    eig=np.ascontiguousarray(np.column_stack(values));co=np.empty((len(y),left_cap+1),complex)
    assert np.abs(eig).max()<1+1e-11
    assert lib.complex(len(y),eig.shape[1],left_cap,eig.view(float),pref.view(float),co.view(float))==0
    joint=np.fft.irfft(co,n=ny,axis=0).T[:,:right_cap+1].copy()
    return joint

def occupancy_prior(p,cap):
    w=len(p);size=1<<w;pm=np.zeros((cap+1,size));pm[0,0]=1
    words=np.arange(size)
    for n in range(1,cap+1):
        for j,v in enumerate(p):
            has=words[(words&(1<<j))!=0]
            pm[n,has]+=v*(pm[n-1,has]+pm[n-1,has^(1<<j)])
    return pm

def leaf_table(lib,local,cap):
    w=local.shape[1]//2;size=1<<w
    blocks=[local_kernel(x) for x in local];p0=np.exp(sum(b[1] for b in blocks))
    joint=np.zeros((cap+1,size));joint[0,0]=p0
    for s in range(1,w+1):
        ids=np.array(list(itertools.combinations(range(w),s)),int);qids=np.c_[ids,ids+w]
        eig=np.column_stack([np.linalg.eigvalsh(k[qids[:,:,None],qids[:,None,:]]) for k,_,_ in blocks])
        masks=(1<<ids).sum(1);joint[:,masks]=series(lib,eig,p0,cap).T
    for j in range(w):
        blocks1=joint.reshape(cap+1,-1,2,1<<j)
        blocks1[:,:,1]-=blocks1[:,:,0]
    # Impossible masks can acquire small floating-point residues in inclusion-exclusion.
    pop=np.array([v.bit_count() for v in range(size)])
    invalid=(pop[None,:]>np.arange(cap+1)[:,None])|((np.arange(cap+1)[:,None]>0)&(pop==0))
    residue=float(abs(joint[invalid]).sum()+abs(joint[joint<0]).sum())
    joint[invalid]=0;joint=np.maximum(joint,0)
    mean=np.diagonal(local,axis1=1,axis2=2).sum(0).reshape(2,w).sum(0)/2
    p=mean/mean.sum()
    prior=occupancy_prior(p,cap)
    mass=joint.sum(1)
    cond=np.divide(joint,mass[:,None],out=prior.copy(),where=mass[:,None]>1e-200)
    return cdf_rows(cond),prior,p,residue

def topology(grams,spectra,ordered):
    count=len(grams)
    children=[[-1,-1] for _ in range(count)];active=list(range(count))
    cov=np.zeros((2*count-1,2*count-1))
    gr=np.array(grams)
    for x in spectra:
        v=(x@gr@x).reshape(count,-1)
        cov[:count,:count]+=.5*(v@gr.reshape(count,-1).T)
    cov=(cov+cov.T)/2
    while len(active)>1:
        if ordered:pairs=list(zip(active[::2],active[1::2]))
        else:
            graph=nx.Graph()
            for i,j in itertools.combinations(active,2):
                corr=cov[i,j]/np.sqrt(max(1e-30,cov[i,i]*cov[j,j]))
                graph.add_edge(i,j,weight=round(corr*1e12))
            pairs=sorted(tuple(sorted(x)) for x in nx.max_weight_matching(graph,maxcardinality=True))
        used=set();following=[]
        for i,j in pairs:
            k=len(children);children.append([i,j]);grams.append(grams[i]+grams[j])
            cov[k,:k]=cov[i,:k]+cov[j,:k];cov[:k,k]=cov[k,:k];cov[k,k]=cov[i,i]+cov[j,j]+2*cov[i,j]
            used.update([i,j]);following.append(k)
        active=[k for k in active if k not in used]+following
    return np.array(children,np.int32)

class Tree:
    def __init__(self,lib,arrays):
        self.lib=lib
        for name,val in arrays.items():setattr(self,name,val)
        self.leaves,self.width=self.ports.shape
    def sample(self,n,route,seed,offset,workers):
        x=np.empty((n,self.leaves*self.width),np.uint16);flags=np.empty(n,np.uint8)
        rc=self.lib.draw(self.leaves,self.width,self.children,self.caps,self.starts,self.root,self.splits,
            self.leafstarts,self.masks,self.prior,self.ports,n,route,seed,offset,workers,x,flags)
        if rc or (flags&2).any():raise RuntimeError(('spatial draw failed',rc,int(flags.max())))
        return x,flags

def prepare(lib,v,spectra,width,ordered=False):
    leaves=len(v)//width;assert len(v)==leaves*width
    transfer=[realify(v[i*width:(i+1)*width]) for i in range(leaves)]
    grams=[t.T@t for t in transfer]
    children=topology(grams,spectra,ordered)
    def radial(g):
        t=compact(g);eig=np.concatenate([np.linalg.eigvalsh(t@x@t.T) for x in spectra])
        cap=capacity(eig)
        law=series(lib,(eig/(1+eig))[None,:],np.exp(-.5*np.log1p(eig).sum()),cap)[0]
        assert law.min()>-1e-13 and abs(law.sum()-1)<3e-12
        return cap,np.maximum(law,0)
    radial_laws=[radial(g) for g in grams]
    caps=np.array([c for c,_ in radial_laws],np.int32)
    def oneleaf(i):
        t=transfer[i];return leaf_table(lib,np.array([t@x@t.T for x in spectra]),int(caps[i]))
    def node(i):
        a,b=children[i];cap=int(caps[i])
        joint=split_joint(lib,grams[a],grams[b],spectra,int(caps[a]),int(caps[b]))
        err=float(abs(joint[joint<0]).sum());joint=np.maximum(joint,0)
        m=np.zeros(cap+1);rows=[]
        for n in range(cap+1):
            first=max(0,n-caps[b]);last=min(n,caps[a])
            row=np.zeros(n+1)
            if last>=first:
                ix=np.arange(first,last+1);row[ix]=joint[ix,n-ix]
            m[n]=row.sum()
            if m[n]>0:row/=m[n]
            else:row[n//2]=1
            rows.append(cdf_rows(row))
        err+=float(abs(m-radial_laws[i][1]).sum())
        err+=float(abs(joint.sum(0)-radial_laws[b][1]).sum()+abs(joint.sum(1)-radial_laws[a][1]).sum())
        return np.concatenate(rows),err
    with ThreadPoolExecutor(max_workers=4) as pool:
        local=list(pool.map(oneleaf,range(leaves)))
        branches=list(pool.map(node,range(leaves,len(grams))))
    leafstarts=np.r_[0,np.cumsum(caps[:leaves-1]+1)].astype(np.int32)
    starts=np.zeros(len(grams),np.int32)
    if branches:starts[leaves:]=np.r_[0,np.cumsum([len(x[0]) for x in branches[:-1]],dtype=int)]
    first=radial_laws[-1][1].copy();first[-1]+=1-first.sum();first=np.maximum(first,0);first/=first.sum()
    arrays=dict(caps=caps,children=children,starts=starts,root=cdf_rows(first),
        splits=np.concatenate([x[0] for x in branches]) if branches else np.zeros(1,np.uint64),
        leafstarts=leafstarts,masks=np.concatenate([x[0] for x in local]),prior=np.concatenate([x[1] for x in local]),
        ports=np.array([x[2] for x in local]))
    diag=dict(max_node_consistency=max([x[1] for x in branches],default=0.),sum_node_consistency=sum(x[1] for x in branches),
              max_leaf_residue=max(x[3] for x in local),root_cap=int(caps[-1]),max_leaf_cap=int(caps[:leaves].max()))
    return arrays,diag

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    out=ap.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    src=ROOT/'native/spatial_tree100_v1.cpp';files=[Path(__file__),src,ROOT/'native/gaussian_rng_domain84_v1.hpp',ORDER]
    save(out,'plan.json',dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},width=8,
        optical_fit='physical signal blocks and maximum weight matching on off-diagonal optical photon covariance; no hardware or heldout data',
        conditional='true optical bivariate total law at every sibling split; true leaf mask law given total; multinomial PNR conditional on that mask'))
    dll=out/'spatial.dll';cmd=[shutil.which('g++'),'-std=c++20','-O3','-shared','-fopenmp','-ffp-contract=off',str(src),'-o',str(dll)]
    run=subprocess.run(cmd,capture_output=True,text=True);save(out,'build.json',dict(command=cmd,code=run.returncode,stderr=run.stderr));run.check_returncode()
    z=np.load(ORDER);v=z['v'][np.argsort(z['order'])];lib=Native(dll)
    t=perf_counter();arrays,diag=prepare(lib,v,z['x'],8);diag['prep_s']=perf_counter()-t
    assert diag['max_node_consistency']<1e-10 and diag['max_leaf_residue']<1e-10,diag
    np.savez_compressed(out/'prepared.npz',**arrays)
    ratios=np.linalg.svd(v.reshape(144,8,50),compute_uv=False)
    diag['branch_second_over_first']=np.percentile(ratios[:,1]/ratios[:,0],[0,50,100]).tolist()
    save(out,'summary.json',diag);manifest(out);print(diag,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
