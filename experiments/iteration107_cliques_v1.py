"""Sparse transformations of a correlated optical field to restore local Gaussian quantum laws."""
import argparse
import ctypes as ct
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration103_gpu_v1 import P,CU,I,U,build
from iteration106_field_v1 import Field,optical,brightness,factors,DATA
from iteration100_spatial_v1 import manifest,save,D,bind
from iteration107_tables_v1 import Counts,kernel,OUT as TABLES

OUT=DATA/'iteration107_cliques_v1'


def optical_x(t,r):
    a,b=t[:,::2],t[:,1::2];out=[]
    for row in r:
        n=np.sinh(row)**2;v=np.sinh(row)*np.cosh(row)
        normal=(a*n)@a.conj().T+(b*n)@b.conj().T
        an=(a*v)@b.T;an=an+an.T
        out.append(np.block([[(normal+an).real,(an-normal).imag],[(an+normal).imag,(normal-an).real]]))
    return np.array(out)


def tail_caps(e,eps):
    # Per-port eigenvalues across all independent spectra, including negative
    # normally ordered quantum quadratures. Chernoff only needs a finite MGF.
    maxeig=e.max(1);caps=np.ones(len(e),np.int32);logs=[]
    for f in np.linspace(.18,.97,24):
        dx=f/np.maximum(maxeig,1e-16)
        logm=-.5*np.log1p(-dx[:,None]*e).sum(1)
        logs.append((logm,np.log1p(dx)))
    caps=np.maximum(2,np.ceil(np.min([(v-np.log(eps))/z for v,z in logs],axis=0)).astype(np.int32))
    return caps


def prepare(t,r,eps=1e-11,max_atoms=6):
    t=np.ascontiguousarray(t,complex);r=np.atleast_2d(r).astype(float)
    m=t.shape[0]
    if t.ndim!=2 or min(r.shape)<1 or t.shape[1]!=2*r.shape[1] or not np.isfinite(t).all() or not np.isfinite(r).all():raise ValueError('Expected paired transfer columns and finite squeezing')
    if np.linalg.eigvalsh(t.conj().T@t)[-1]>1+1e-9:raise ValueError('Transfer must be passive')
    if len(r)>4:raise ValueError('Count recurrence supports up to four spectra per input')
    native=Counts(TABLES/'counts.dll');fs=factors(t,r);eig=np.concatenate([np.linalg.eigvalsh(2*f@f.T) for f in fs])
    prof=([p for p in brightness(optical(t,r),eig,max_atoms) if p['feasible']][-1] if eig.max()>0 else dict(d=1,values=[1.],weights=[1.]))
    n=np.sinh(r)**2;an=np.sinh(r)*np.cosh(r);a,b=t[:,::2],t[:,1::2]
    power=(abs(a)**2+abs(b)**2)@n.T;v=abs(2*(a*b)@an.T)
    true_e=np.c_[power+v,power-v]
    vbase=abs(2*(a*b)@(np.sign(r)*n).T)
    base_e=np.c_[power+vbase,power-vbase]*max(prof['values'])
    caps=np.maximum(tail_caps(true_e,eps/m),tail_caps(base_e,eps/m))
    blocks=[list(range(i,min(i+4,m))) for i in range(0,m,4)]
    mu=power.sum(1)
    bright=max(blocks,key=lambda ix:mu[ix].sum())
    # Keep the brightest four-port region together. Other adjacent pairs give
    # bounded lookup cost as width grows. If a local tensor exceeds budget,
    # split the block while preserving its rigorously chosen per-port cutoffs.
    blocks=[bright]+[ix[k:k+2] for ix in blocks if ix!=bright for k in range(0,len(ix),2)]
    groups=[]
    def split(ids):
        cells=1
        for j in ids:cells*=int(caps[j])
        if cells<=1900000:groups.append(ids)
        elif len(ids)>1:
            k=len(ids)//2;split(ids[:k]);split(ids[k:])
        else:raise ValueError('Single-port photon range exceeds available table')
    for ids in blocks:split(ids)
    plan=[];cn=[];pn=[];cd=[];pd=[];co=po=0;coverage=[];masses=[]
    for ids in groups:
        d=len(ids);ix=np.array(ids);sl=np.r_[ix,m+ix];size=caps[ix]
        q=np.eye(2*d)[None,:,:]+optical_x(t[ix],r)
        h=np.stack([2*f[:,sl].T@f[:,sl] for f in fs]);eye=np.eye(2*d)
        target=native(q,size)
        base=sum(v*native(eye[None,:,:]+g*h,size) for v,g in zip(prof['weights'],prof['values']))
        assert abs(base.sum()-1)<3e-10 and abs(target.sum()-1)<3e-10
        stride=np.cumprod(np.r_[1,size[:-1]]);states=np.arange(len(base))
        mask=sum((((states//s)%c)>0).astype(int)<<j for j,(s,c) in enumerate(zip(stride,size)))
        mb=np.bincount(mask,weights=base,minlength=1<<d);mt=np.bincount(mask,weights=target,minlength=1<<d)
        rc,dc,tvc,ec=kernel(mb,mt);rp,dp,tvp,ep=kernel(base,target)
        cn.append(rc);cd.append(dc);pn.append(rp);pd.append(dp)
        plan.append(ids+[0]*(4-d)+size.tolist()+[1]*(4-d)+[d,co,po,len(base)])
        po+=len(base);co+=len(mb)
        if max(po,co)>=2**31:raise ValueError('Table offset exceeds native range')
        coverage.append((tvc,tvp,ec,ep));masses.append((base.sum(),target.sum()))
    return dict(plan=np.array(plan,np.int32),click_move=np.concatenate(cn),click_cdf=np.concatenate(cd),pnr_move=np.concatenate(pn),pnr_cdf=np.concatenate(pd),values=np.array(prof['values']),weights=np.array(prof['weights']),coverage=np.array(coverage),masses=np.array(masses),caps=caps,tail_budget=np.array(eps))


class Generic:
    def __init__(self,t,r,config,dll=OUT/'gpu.dll'):
        self.base=Field(t,r,dict(values=config['values'],weights=config['weights']))
        self.lib=ct.CDLL(str(dll));self.fn=self.lib.clique107_launch;self.fn.argtypes=[I,I,I,I,U,CU,P,P,P,P,P]
        self.plan=torch.as_tensor(config['plan'],device='cuda')
        self.table=[(torch.as_tensor(config[k+'_move'],device='cuda'),torch.as_tensor(config[k+'_cdf'],device='cuda')) for k in ('click','pnr')]
    def draw(self,n,pnr,seed,offset=0,paired=False):
        b=self.base
        if n<1 or n*max(b.s*b.k,b.m)>=2**31 or not 0<=offset<=2**32-n or not 0<=seed<2**64:raise ValueError('Invalid draw')
        if b.capacity<n:
            b.z=torch.empty(b.s*n*2*b.k,dtype=b.dtype,device='cuda');b.f=torch.empty(b.s*n*2*b.m,dtype=b.dtype,device='cuda')
            b.x=torch.empty((n,b.m),dtype=torch.int32,device='cuda');b.capacity=n
        z=b.z[:b.s*n*2*b.k].view(b.s,n,2*b.k);f=b.f[:b.s*n*2*b.m].view(b.s,n,2*b.m)
        stream=torch.cuda.current_stream().cuda_stream
        rc=b.noise(b.k,b.s,n,seed,offset,0,z.data_ptr(),stream)
        if rc:raise RuntimeError(rc)
        torch.bmm(z,b.factor,out=f)
        rc=b.detect(b.m,b.s,n,int(pnr),seed,offset,0,f.data_ptr(),b.atoms,b.values.data_ptr(),b.cdf.data_ptr(),b.x.data_ptr(),b.error.data_ptr(),stream)
        if rc:raise RuntimeError(rc)
        before=b.x[:n].cpu().numpy() if paired else None
        move,cdf=self.table[int(pnr)]
        rc=self.fn(len(self.plan),b.m,n,int(pnr),seed,offset,self.plan.data_ptr(),move.data_ptr(),cdf.data_ptr(),b.x.data_ptr(),stream)
        if rc:raise RuntimeError(rc)
        out=b.x[:n].cpu().numpy()
        if b.error.item():raise ArithmeticError('Invalid optical rate')
        return (out,before) if paired else out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','prepare','draw'));p.add_argument('output',type=Path)
    p.add_argument('--input',type=Path);p.add_argument('--tables',type=Path);p.add_argument('--seed',type=int);p.add_argument('--shots',type=int,default=2048);p.add_argument('--pnr',action='store_true')
    a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):
        if a.action=='build':
            a.output.mkdir(parents=True,exist_ok=False);build(a.output,'clique_transport107_v1.cu');manifest(a.output)
        else:
            if a.input is None:p.error('Provide an optical input')
            with np.load(a.input) as z:t,r=z['transfer'],np.atleast_2d(z['squeezing'])
            if a.action=='prepare':
                with a.output.open('xb') as f:np.savez_compressed(f,**prepare(t,r))
            else:
                if a.tables is None or a.seed is None:p.error('Provide tables and an experiment seed')
                with np.load(a.tables) as z:config=dict(z)
                model=Generic(t,r,config)
                with a.output.open('xb') as f:np.save(f,model.draw(a.shots,a.pnr,a.seed))
