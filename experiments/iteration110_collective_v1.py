"""Collective Gaussian interference with loss and arbitrary-width uniform detector banks."""
import argparse
import ctypes as ct
import math
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,save,manifest,realify
from iteration103_gpu_v1 import build as build_gpu,I,U,CU,P
from iteration106_field_v1 import DATA
from iteration106_sample_v1 import benchmark,panel_hist
from iteration107_fast_v1 import oracle,gaussian_click,ONE
from iteration109_sample_v1 import tv


def unitary(d):
    rng=np.random.default_rng(110100000+d)
    u,rr=np.linalg.qr(rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)))
    return u*np.exp(1j*np.angle(np.diag(rr)))


def covariance(u,r,eta):
    f=realify(np.sqrt(eta)[:,None]*u)
    return (f*(np.expm1(np.r_[-2*r,2*r])/2))@f.T


def tail(r,cut):
    l=np.tanh(r)**2
    if not l.max():return 0.
    def bound(s):return -(cut+1)*np.log(s)+.5*(np.log1p(-l)-np.log1p(-l*s*s)).sum()
    fit=minimize_scalar(bound,bounds=(1,1/np.sqrt(l.max())),method='bounded')
    return float(np.exp(min(0.,fit.fun)))


def sparse(law,states,budget):
    if np.min(law)<0 or not np.isfinite(law).all() or law.sum()<=0:raise ValueError('Invalid collective law')
    order=np.argsort(law,kind='stable')
    keep=order[np.cumsum(law[order])>budget]
    keep=np.unique(np.r_[0,keep]);weights=law[keep].copy()
    dropped=math.fsum(law)-math.fsum(weights)
    # Assign unresolved sectors and sparsified mass to the vacuum record.
    weights[0]+=1-weights.sum()
    if weights[0]<0:raise ArithmeticError('Invalid vacuum correction')
    norm=np.cumsum(weights.astype(np.longdouble));cdf=np.rint(norm*2**52).astype(np.uint64)
    cdf=np.minimum(cdf,np.uint64(2**52));cdf[-1]=2**52
    rounded=np.diff(cdf,prepend=np.uint64(0)).astype(float)/2**52
    return (np.ascontiguousarray(states[keep]),cdf),dict(entries=len(keep),dropped=max(0.,dropped),rounding=tv(weights,rounded))


def prepare(lib,u,r,eta,width,cut):
    r=np.asarray(r,float);eta=np.asarray(eta,float);u=np.asarray(u,complex);width=np.asarray(width,np.int32);d=len(r)
    if not 1<=d<=8 or u.shape!=(d,d) or width.shape!=(d,) or eta.shape!=(d,) or np.any(width<1) or np.any((eta<0)|(eta>1)) or not np.isfinite(r).all() or not np.isfinite(eta).all() or not np.isfinite(u).all() or np.max(abs(u.conj().T@u-np.eye(d)))>1e-10:raise ValueError('Invalid optical banks')
    if not 0<=cut<=512 or math.comb(cut+d,d)>8000000 or width.sum()>=2**30:raise ValueError('Requested core table exceeds native storage')
    size=math.comb(cut+d,d);states=np.empty((size,d),np.uint16)
    arrays=np.empty((3,size));b=np.ascontiguousarray(-(u*np.tanh(r))@u.T)
    fn=lib.collective110;fn.argtypes=[I,I,I,P,ct.c_double]+[P]*6+[I]
    rc=fn(d,cut,size,b.ctypes.data,float(np.prod(1/np.cosh(r))),width.ctypes.data,eta.ctypes.data,
          states.ctypes.data,*[a.ctypes.data for a in arrays],12)
    if rc:raise ValueError('Core table rejected')
    p0=float(np.prod(1/np.cosh(r)));l=np.tanh(r)**2
    exact=[p0]
    for n in range(1,cut//2+1):exact.append(sum(np.sum(l**j)*exact[n-j] for j in range(1,n+1))/(2*n))
    totals=states.sum(1).astype(int);sector=np.bincount(totals,weights=arrays[0],minlength=cut+1)
    err=float(abs(sector[::2]-exact).sum()+sector[1::2].sum());assert err<2e-12
    assert abs(arrays.sum(1)-sum(exact)).max()<2e-12
    plans=[];stats=[]
    for law in (arrays[2],arrays[1]):
        table,info=sparse(law,states,2e-13);plans.append(table);stats.append(info)
    return plans,dict(core=d,cut=cut,states=size,emission_tail_upper=tail(r,cut),sector_l1_residual=err,tables=stats)


class Collective:
    def __init__(self,plans,width,cells,dll):
        self.cells=cells;self.d=len(width);self.span=sum(map(int,width));self.width=np.array(width,int)
        if cells<1 or self.span*cells>=2**31:raise ValueError('Invalid optical layout')
        self.end=torch.as_tensor(np.cumsum(width,dtype=np.int32),device='cuda')
        self.plans=[]
        for values,cdf in plans:
            self.plans.append((len(cdf),torch.as_tensor(cdf,device='cuda'),torch.as_tensor(values,device='cuda'),bool(values.max()>255)))
        self.lib=ct.CDLL(str(dll));self.fn=self.lib.collective_emit110
        self.fn.argtypes=[I]*7+[U,CU]+[P]*5
        self.buf={}
    def draw(self,n,pnr,seed,offset=0):
        if not 0<n or n*self.cells>=2**31-256 or not 0<=offset<=2**32-n or not 0<=seed<2**64:raise ValueError('Invalid RNG domain')
        size,cdf,values,wide=self.plans[int(pnr)];m=self.span*self.cells
        if wide not in self.buf or len(self.buf[wide])<n:self.buf[wide]=torch.empty((n,m),device='cuda',dtype=torch.uint16 if wide else torch.uint8)
        v=self.buf[wide][:n];rc=self.fn(n,self.cells,self.d,self.span,size,int(pnr),int(wide),seed,offset,self.end.data_ptr(),cdf.data_ptr(),values.data_ptr(),v.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if rc:raise RuntimeError(rc)
        return v.cpu().numpy()


def banks(x,width,ids):
    ix=np.repeat(np.arange(len(width)),width)[ids];scale=1/np.sqrt(np.asarray(width)[ix])
    take=np.r_[ix,ix+len(width)];return x[np.ix_(take,take)]*np.outer(np.tile(scale,2),np.tile(scale,2))


def partial_transposes(x):
    d=len(x)//2;omega=np.kron(np.array([[0,1],[-1,0]]),np.eye(d));v=2*x+np.eye(2*d);nu=[]
    for mask in range(1,1<<(d-1)):
        sign=np.r_[np.ones(d),[(-1 if mask>>i&1 else 1) for i in range(d)]]
        nu.append(float(np.min(abs(np.linalg.eigvals(1j*omega@(v*np.outer(sign,sign)))))))
    return nu


def run(out):
    out.mkdir(parents=True,exist_ok=False);build_gpu(out,'collective_readout110_v1.cu')
    subprocess.run([shutil.which('g++'),'-O3','-ffp-contract=off','-std=c++20','-march=x86-64-v3','-fopenmp','-shared',str(ROOT/'native/collective_laws110_v1.cpp'),'-o',str(out/'laws.dll')],check=True)
    rt=os.add_dll_directory(str(Path(shutil.which('g++')).parent)) if os.name=='nt' else None
    lib=ct.CDLL(str(out/'laws.dll'));rows=[];arrays={};u=unitary(4);r=np.array([.76,.64,.53,.45]);eta=np.array([.91,.87,.94,.89]);cut=68
    x=covariance(u,r,eta);nu=partial_transposes(x);assert max(nu)<1
    layouts=[([4,4,4,4],[1,72,8192]),([1,1,1,1],[1,32768]),([32768,32768,32768,32769],[1])]
    for phase,(width,counts) in enumerate(layouts):
        st=perf_counter();plans,meta=prepare(lib,u,r,eta,width,cut);prep=perf_counter()-st
        for cell in counts:
            m=sum(width)*cell;model=Collective(plans,width,cell,out/'gpu.dll')
            batch=min(2048,max(1,4194304//m));timings=[]
            for pn in (False,True):
                tm=benchmark(lambda n,s,o:model.draw(n,pn,s,o),batch,110120000+1000*phase+2*cell+pn,11);timings.append(tm)
                if cell==1:
                    a=model.draw(17,pn,110160000+2*phase+pn)
                    assert np.array_equal(a,np.vstack((model.draw(5,pn,110160000+2*phase+pn),model.draw(12,pn,110160000+2*phase+pn,5))))
            upper=cell*(meta['emission_tail_upper']+max(v['dropped']+v['rounding'] for v in meta['tables']))
            rows.append(dict(width=width,cells=cell,ports=m,mean_photons=float(cell*x.trace()/2),timing=timings,
                             ideal_budget_tv_upper=min(1.,upper),setup_s=prep,core=meta))
            if phase==0 and cell==1:
                target=gaussian_click(oracle(True),banks(x,width,np.arange(m))[None],ONE)
                # Given K, every word of those group cardinalities is equiprobable.
                words=np.arange(1<<m);start=0;key=np.zeros(len(words),int);multiplicity=np.ones(len(words))
                for i,L in enumerate(width):
                    ks=np.array([int((w>>start)&((1<<L)-1)).bit_count() for w in words])
                    key+=ks*5**i;multiplicity*=[math.comb(L,int(n)) for n in ks];start+=L
                vs,cdf=plans[0];mass=np.bincount(vs@(5**np.arange(4)),weights=np.diff(cdf,prepend=np.uint64(0)).astype(float)/2**52,minlength=625)
                law=mass[key]/multiplicity;gap=tv(law,target);assert gap<1e-9
                arrays.update(full_model=law,full_optical=target,core_x=x,unitary=u,squeezing=r,efficiency=eta)
                for pn in (False,True):
                    h=np.zeros(65536,np.int64);shots=262144
                    for off in range(0,shots,2048):h+=panel_hist(model.draw(2048,pn,110170000+pn,off),np.arange(m))
                    arrays[f'full_hist{int(pn)}']=h
                rows[-1]['full_16_tv']=gap
    np.savez_compressed(out/'optical.npz',**arrays);save(out,'summary.json',dict(rows=rows,partial_transpose_minima=nu));manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(ap.parse_args().output.resolve())
