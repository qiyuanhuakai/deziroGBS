"""Coherent pair matching on a passive optical dilation with a full joint-distribution TV bound."""
import argparse
import ctypes as ct
import math
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,manifest
from iteration103_gpu_v1 import I,U,CU,P
from iteration106_sample_v1 import DATA

OUT=DATA/'iteration108_matching_v1'


def cum(w):
    p=np.cumsum(w,dtype=np.longdouble)
    if p[-1]>0:p/=p[-1]
    else:p[:]=1
    p=p.astype(float);p[-1]=1.;return p


def certificate(lam,cap,tries):
    results=[]
    for l in lam:
        x=np.array(l,dtype=np.longdouble)**2
        h=np.zeros(cap+1,np.longdouble);h[0]=1
        for w in x:
            for j in range(1,cap+1):h[j]+=w*h[j-1]
        mass=h*np.exp(np.log1p(-x).sum())
        acc=np.ones(cap+1,np.longdouble)
        if x.sum()>0:
            for n in range(1,cap+1):
                odd=math.prod(range(1,2*n,2))
                acc[n]=math.factorial(n)*h[n]/(odd*x.sum()**n)
        limit=max(0.,float(1-mass.sum()));fail=float(mass@((1-acc.clip(0,1))**tries))
        results.append(dict(mass=mass.astype(float).tolist(),mean_acceptance=acc.astype(float).tolist(),
                            emission_tail=limit,rejection_budget=fail))
    bound=-np.expm1(sum(np.log1p(-min(1.,z['emission_tail']+z['rejection_budget'])) if z['emission_tail']+z['rejection_budget']<1 else -np.inf for z in results))
    return dict(full_joint_tv_bound=float(bound),spectra=results,cap=cap,max_tries=tries)


class Matching:
    def __init__(self,t,r,cap=4,tries=2048,workers=12,dll=OUT/'matching.dll'):
        t=np.asarray(t,dtype=complex);r=np.atleast_2d(r).astype(float)
        if t.ndim!=2 or min(t.shape)<1 or not np.isfinite(t).all() or not np.isfinite(r).all() or r.shape[1]*2!=t.shape[1] or not 1<=len(r)<=65536:
            raise ValueError('Expected a finite passive paired source input')
        if not 1<=cap<=6 or not 1<=tries<2**31 or not 1<=workers<2**16:raise ValueError('Invalid matching budget')
        self.lam=np.ascontiguousarray(np.tanh(r));p=self.lam**2
        if p.max()>=1-1e-6:raise ValueError('Squeezing exceeds the supported geometric-count range')
        self.s,self.k=r.shape;self.m=len(t);self.cap=cap;self.tries=tries;self.workers=workers
        loss=np.eye(2*self.k)-t.conj().T@t
        e,v=np.linalg.eigh(loss)
        if min(e)<-1e-10:raise ValueError('Transfer exceeds passivity')
        self.u=np.ascontiguousarray(np.vstack((t,np.sqrt(e.clip(0))[:,None]*v.conj().T)))
        self.rows=len(self.u)
        self.isometry=float(np.max(abs(self.u.conj().T@self.u-np.eye(2*self.k))))
        if self.isometry>2e-10:raise ValueError('Invalid loss dilation')
        w=abs(self.u)**2;lev=w.sum(1);self.cert=certificate(self.lam,cap,tries)
        col=np.stack([cum(w[:,j]) for j in range(2*self.k)])
        hazard=np.c_[np.zeros(self.s),np.cumsum(-np.log1p(-p),axis=1)]
        first=np.stack([cum((w[:,::2]+w[:,1::2])@x) for x in p])
        self.args=[self.u,self.lam,hazard,first,lev,cum(lev),col]
        if os.name=='nt':self.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
        self.lib=ct.CDLL(str(dll));self.fn=self.lib.matching108_draw
        self.fn.argtypes=[I]*10+[U,CU]+[P]*9
    def draw(self,n,pnr,seed,offset=0,stats=False):
        if not 1<=n<2**31 or not 0<=seed<2**64 or not 0<=offset<=2**32-n:raise ValueError('Invalid record domain')
        for wide in (0,1):
            x=np.empty((n,self.m),np.uint32 if wide else np.uint8);counts=np.zeros((self.s,self.cap+2,3),np.uint64)
            rc=self.fn(self.m,self.k,self.s,self.rows,n,self.cap,self.tries,int(pnr),wide,self.workers,seed,offset,
                       *(v.ctypes.data for v in self.args),x.ctypes.data,counts.ctypes.data)
            if rc==0:return (x,counts) if stats else x
            if rc!=1:raise ArithmeticError(f'Inconsistent conditional pair law ({rc})')
        raise ArithmeticError('Photon count exceeds the wide output limit')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','draw'));p.add_argument('output',type=Path)
    p.add_argument('--input',type=Path);p.add_argument('--cap',type=int,default=4);p.add_argument('--tries',type=int,default=2048)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--seed',type=int);p.add_argument('--shots',type=int,default=256);p.add_argument('--offset',type=int,default=0)
    p.add_argument('--pnr',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if a.action=='build':
            a.output.mkdir(exist_ok=False,parents=True)
            subprocess.run([shutil.which('g++'),'-std=c++20','-O3','-march=x86-64-v3','-ffp-contract=off','-fopenmp','-shared',str(ROOT/'native/pair_matching108_v1.cpp'),'-o',str(a.output/'matching.dll')],check=True);manifest(a.output)
        else:
            if a.input is None or a.seed is None:p.error('Specify the input and an experiment seed')
            with np.load(a.input) as z:model=Matching(z['transfer'],z['squeezing'],a.cap,a.tries,a.workers)
            with a.output.open('xb') as f:np.save(f,model.draw(a.shots,a.pnr,a.seed,a.offset))
