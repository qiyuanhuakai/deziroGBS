"""Exact pair-entangled Gaussian family with a dense, shared random displacement on the idlers."""
import argparse
import ctypes as ct
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from iteration100_baselines_v1 import frugal,purification,pair_reference
from iteration92_pe_common_v1 import pe_sample
from iteration103_gpu_v1 import I,U,CU,P,build
from iteration106_field_v1 import DATA
from iteration106_sample_v1 import benchmark,panel_hist
from iteration107_fast_v1 import oracle,gaussian_click,ONE
from iteration109_sample_v1 import tv

def optics(m):
    k=(m+1)//2;rng=np.random.default_rng(109400000+m)
    l=.20+.31*rng.random(k);factor=rng.normal(size=(8,2,k))*.13
    if m%2:l[-1]=0
    return l,factor

def covariance(m,l,f):
    k=len(l);z=np.zeros((2,m,2,m))
    field=np.zeros((len(f),2,m))
    for j in range(k):
        a=2*j;b=min(m-1,a+1);mean=l[j]/(1-l[j])
        for q in (0,1):
            z[q,a,q,a]=z[q,b,q,b]=mean
            if a!=b:z[q,a,q,b]=z[q,b,q,a]=(1-2*q)*np.sqrt(mean*(mean+1))
            field[:,q,b]=f[:,q,j]
    field=field.reshape(len(f),2*m)
    return z.reshape(2*m,2*m)+2*field.T@field

def exact_click(m,l,f):
    size=1<<m;mask=np.arange(size);r=len(f)
    sums=np.broadcast_to(np.eye(r),(size,r,r)).copy();base=np.ones(size)
    for j,w in enumerate(l):
        a=2*j;b=min(m-1,a+1);v=1-w
        ca=(mask>>a)&1;cb=(mask>>b)&1
        base*=np.where(ca|cb,v,1)
        rate=cb*np.where(ca,1,v)
        sums+=2*rate[:,None,None]*(f[:,:,j]@f[:,:,j].T)
    vac=base*np.exp(-np.linalg.slogdet(sums)[1]/2);words=vac[::-1].astype(np.longdouble)
    for bit in range(m):
        half=1<<bit;view=words.reshape(-1,2*half);view[:,half:]-=view[:,:half]
    return np.asarray(words,float)

class Twins:
    def __init__(self,m,l,f,dll):
        self.m=m;self.l=l;self.k=len(l);self.rank=len(f)
        if not (0<m<2**30 and self.k==(m+1)//2 and len(f)%2==0 and f.shape==(len(f),2,self.k)
                and np.isfinite(f).all() and np.all((l>=0)&(l<1)) and (m%2==0 or l[-1]==0)):raise ValueError('Invalid twin-beam layout')
        self.normal_lib=ct.CDLL(str(DATA/'iteration107_fast_v1/gpu.dll'))
        self.normal=self.normal_lib.normals107_launch;self.normal.argtypes=[I,I,I,U,CU,I,P,P,I,P,P,P]
        self.lib=ct.CDLL(str(dll));self.emit=self.lib.twin109;self.emit.argtypes=[I]*4+[U,CU]+[P]*5
        self.factor=torch.as_tensor(f.reshape(len(f),2*self.k).copy(),device='cuda',dtype=torch.float32)
        self.lam=torch.as_tensor(l,device='cuda');self.unit=torch.ones(1,device='cuda',dtype=torch.float64)
        self.error=torch.zeros(1,device='cuda',dtype=torch.int32);self.capacity=0
        if self.k>=65536*((65535-373)//4) or self.rank//2>=65536*((65535-221)//8):raise ValueError('RNG address overflow')
    def draw(self,n,pnr,seed,offset=0):
        if not 0<n or n*max(self.rank,2*self.k)>=2**31-256 or not 0<=offset<=2**32-n or not 0<=seed<2**64:raise ValueError('Invalid batch')
        if n>self.capacity:
            self.z=torch.empty((n,self.rank),device='cuda');self.f=torch.empty((n,self.k*2),device='cuda')
            self.gain=torch.empty(n,device='cuda',dtype=torch.float64);self.x=torch.empty((n,self.m),device='cuda',dtype=torch.uint8);self.capacity=n
        st=torch.cuda.current_stream().cuda_stream;z=self.z[:n];f=self.f[:n];out=self.x[:n]
        rc=self.normal(self.rank//2,1,n,seed,offset,1,self.unit.data_ptr(),self.unit.data_ptr(),0,z.data_ptr(),self.gain.data_ptr(),st)
        if rc:raise RuntimeError(rc)
        torch.mm(z,self.factor,out=f)
        for wide in (0,1):
            if wide:out=torch.empty_like(out,dtype=torch.int32)
            rc=self.emit(self.m,n,int(pnr),wide,seed,offset,self.lam.data_ptr(),f.data_ptr(),out.data_ptr(),self.error.data_ptr(),st)
            if rc:raise RuntimeError(rc)
            err=self.error.item()
            if not err:return out.cpu().numpy()
            self.error.zero_()
            if wide or err!=1:raise ArithmeticError('Displaced Fock readout guard reached')

def run(out):
    out.mkdir(exist_ok=False,parents=True);build(out,'displaced_twins109_v1.cu')
    torch.backends.cuda.matmul.allow_tf32=False
    quad=oracle(True);summ=[];refs={};comparisons={};small=16
    for m in (small,1152,8192,131073):
        l,f=optics(m);model=Twins(m,l,f,out/'gpu.dll');batch=min(2048,4194304//m)
        mu=float(2*np.sum(l/(1-l))+np.sum(f*f))
        x2=covariance(2,l[:1],f[:,:,:1]);omega=np.kron(np.array([[0,1],[-1,0]]),np.eye(2))
        pt=np.diag([1,1,1,-1]);nu=float(abs(np.linalg.eigvals(1j*omega@pt@(np.eye(4)+2*x2)@pt)).min())
        assert nu<1
        row=dict(modes=m,mean_photons=mu,first_pair_partial_transpose_symplectic_min=nu)
        for i,pnr in enumerate((False,True)):
            row['pnr' if pnr else 'click']=benchmark(lambda n,s,o:model.draw(n,pnr,s,o),batch,109410000+m+i,9)
        summ.append(row)
        if m!=small:continue
        x=covariance(m,l,f);q=x+np.eye(2*m);p=exact_click(m,l,f);target=gaussian_click(quad,x[None],ONE)
        gap=tv(p,target);assert gap<1e-9 and p.min()>-1e-13;refs.update(model=p,target=target)
        lib=ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'))
        for k in (2,4):
            fr,times=frugal(lib,q[None],k,109420000+100*k);refs[f'frugal{k}']=fr
            comparisons[f'frugal{k}']=dict(model_tv=tv(fr,target),us=float(np.median(times)))
        comparisons['local']=dict(model_tv=gap)
        for pnr in (False,True):
            xs=model.draw(64,pnr,109425000+pnr)
            assert np.array_equal(xs,np.vstack([model.draw(13,pnr,109425000+pnr),model.draw(51,pnr,109425000+pnr,13)]))
            shots=262144;h=np.zeros(1<<m,np.int64)
            for off in range(0,shots,2048):h+=panel_hist(model.draw(2048,pnr,109426000+pnr,off),np.arange(m))
            refs[f'hist{int(pnr)}']=h
            comparisons['local']['threshold_tv' if pnr else 'direct_tv']=tv(h/shots,target)
        # Compare two full physical counts in the exact same reduced state.
        ptarg=pair_reference((x2+np.eye(4))[None],12);state,u,err=purification(x2+np.eye(4))
        tm=[];batch2=128;hist=np.zeros((2,169),np.int64);ns=8192
        reduced=Twins(2,l[:1],f[:,:,:1],out/'gpu.dll')
        gt=benchmark(lambda n,s,o:reduced.draw(n,True,s,o),batch2,109427000,9)
        for off in range(0,ns,batch2):
            t0=perf_counter()
            y,stats=pe_sample(state,u,batch2,seed=109428000+off//batch2,device='cpu',impl='v2',window=4,max_photons=96,return_stats=True)
            y=y[:,:2].copy();tm.append((perf_counter()-t0)*1e6/batch2);assert not np.any(stats.censored)
            a=reduced.draw(batch2,True,109429000,off)
            for hh,z in zip(hist,(a,y)):hh+=np.bincount(z.clip(0,12)@np.array([1,13]),minlength=169)
        refs.update(pair_target=ptarg,pair_hist=hist)
        comparisons['pnr']=dict(shots=ns,ports=[0,1],purification_error=err,ours_us=gt['us'],pe_us=float(np.median(tm[1:])),
                                ours_empirical_tv=tv(hist[0]/ns,ptarg),pe_empirical_tv=tv(hist[1]/ns,ptarg))
    np.savez_compressed(out/'laws.npz',**refs)
    save(out,'summary.json',dict(scaling=summ,comparison=comparisons));manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(ap.parse_args().output.resolve())
