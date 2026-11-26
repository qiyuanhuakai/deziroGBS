"""Preserve optical mean and aggregate on-port factorial moments while compressing the measured fan-out."""
import argparse
import ctypes as ct
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,I,U,D,bind,Native,save,manifest
from iteration100_projection_v1 import physical
from iteration103_gpu_v1 import P,CU,build
from iteration105_jiuzhang_v1 import inputs
from iteration106_field_v1 import Field,optical,factors,brightness,actual_input,OUT as FIELD,DATA,BASE
from iteration106_reference_v1 import OUT as REF
from iteration106_sample_v1 import benchmark,read

OUT=DATA/'iteration106_packed_v1'


def compress(t,n,rank,width=8):
    groups=len(t)//width
    u,s,v=np.linalg.svd(t.reshape(groups,width,-1),full_matrices=False)
    modes=(s[:,:rank,None]*v[:,:rank]).reshape(groups*rank,-1)
    weight=u[:,:,:rank].copy()
    reconstruction=(weight@modes.reshape(groups,rank,-1)).reshape(t.shape)
    # The shared detector gain restores the optical mean at each physical port.
    original=abs(t)**2@n.repeat(2);model=abs(reconstruction)**2@n.repeat(2)
    ratio=np.sqrt(np.divide(original,model,out=np.ones_like(model),where=model>0))
    if np.any((model==0)&(original>0)):raise ValueError('A compressed row lost its entire mean intensity')
    weight*=ratio.reshape(groups,width,1);reconstruction*=ratio[:,None]
    return modes,np.ascontiguousarray(weight.reshape(-1,rank)),reconstruction


def spectrum_fit(t,r):
    power=abs(t[:,::2])**2+abs(t[:,1::2])**2
    uv=2*t[:,::2]*t[:,1::2]
    n=np.sinh(r)**2
    ideal=optical(t,r);mu=ideal.sum()/2
    target_f2=mu**2+ideal@ideal/2
    ms=n@power.T
    diag=(ms.sum(0)**2+(ms**2+abs((np.cosh(r)*np.sinh(r))@uv.T)**2).sum(0)).sum()
    def at(k):
        nn=n.copy();nn[0]+=k*n[1];nn[1]-=k*n[1]
        rr=np.arcsinh(np.sqrt(nn));f=factors(t,rr)
        eig=np.concatenate([np.linalg.eigvalsh(2*a@a.T) for a in f])
        g2=target_f2/(mu**2+eig@eig/2)
        local=nn@power.T
        est=g2*(local.sum(0)**2+(local**2+abs(nn@uv.T)**2).sum(0)).sum()
        return est-diag,rr,eig
    k=brentq(lambda x:at(x)[0],0,1,xtol=1e-13)
    gap,rr,eig=at(k);assert abs(gap)<1e-10
    return k,rr,eig,ideal


class Packed:
    def __init__(self,modes,weight,r,profile,width=8,dll=OUT/'gpu.dll'):
        self.rank=weight.shape[1];self.m=len(weight);self.width=width;self.groups=self.m//width
        self.s,self.k=np.shape(r)
        if self.rank not in (1,2) or self.m%width or self.m>65536 or self.s*self.k>65536:raise ValueError('Unsupported port layout')
        self.factor=torch.as_tensor(factors(modes,r),device='cuda',dtype=torch.float32)
        self.transfer=torch.as_tensor(weight,device='cuda')
        self.val=torch.as_tensor(profile['values'],dtype=torch.float64,device='cuda')
        self.cdf=torch.as_tensor(np.cumsum(profile['weights']),dtype=torch.float64,device='cuda')
        self.lib=ct.CDLL(str(dll))
        self.noise=self.lib.normal_packed106_launch;self.noise.argtypes=[I,I,I,U,CU,P,P]
        self.readout=self.lib.packed_field106_launch;self.readout.argtypes=[I,I,I,I,I,I,U,CU,P,P,I,P,P,P,P,P]
        self.error=torch.zeros(1,device='cuda',dtype=torch.int32)
        self.capacity=0
    def draw(self,n,pnr,seed,offset=0):
        if n<1 or n*max(self.m,self.s*self.k)>=2**31 or not 0<=seed<2**64 or not 0<=offset<=2**32-n:raise ValueError('Invalid random domain')
        cols=self.groups*self.rank
        if n>self.capacity:
            self.z=torch.empty((self.s*n*2*self.k,),device='cuda');self.f=torch.empty((self.s*n*2*cols,),device='cuda')
            self.out=torch.empty((n,self.m),dtype=torch.uint16,device='cuda');self.capacity=n
        z=self.z[:self.s*n*2*self.k].view(self.s,n,2*self.k);f=self.f[:self.s*n*2*cols].view(self.s,n,2*cols)
        stream=torch.cuda.current_stream().cuda_stream
        rc=self.noise(self.k,self.s,n,seed,offset,z.data_ptr(),stream)
        if rc:raise RuntimeError(rc)
        torch.bmm(z,self.factor,out=f)
        rc=self.readout(self.groups,self.width,self.rank,self.s,n,int(pnr),seed,offset,f.data_ptr(),self.transfer.data_ptr(),
                len(self.val),self.val.data_ptr(),self.cdf.data_ptr(),self.out.data_ptr(),self.error.data_ptr(),stream)
        if rc:raise RuntimeError(rc)
        y=self.out[:n].cpu().numpy()
        if self.error.item():raise ArithmeticError('Optical rate or uint16 overflow')
        return y


def model_case(rank):
    t,r=actual_input();conf=read(OUT/'summary.json')
    rr=np.array(conf['squeezing'])
    mo,weight,eff=compress(t,(np.sinh(r)**2).sum(0),rank)
    return Packed(mo,weight,rr,conf['profiles'][str(rank)]),eff,rr,conf['profiles'][str(rank)]


def field_panel(fs,profile,ids,oracle):
    m=fs.shape[-1]//2;sub=np.r_[ids,m+ids];x=np.stack([2*f[:,sub].T@f[:,sub] for f in fs]);p=np.zeros(1<<len(ids))
    for weight,v in zip(profile['weights'],profile['values']):
        part=np.empty_like(p)
        assert oracle(len(ids),len(fs),np.eye(2*len(ids))[None,:,:]+v*x,np.zeros((len(fs),2*len(ids))),part,np.empty(2))==0
        p+=weight*part
    return p


def run(out):
    out.mkdir(exist_ok=False,parents=True)
    build(out,'packed_field106_v1.cu')
    eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'
    subprocess.run([shutil.which('g++'),'-O3','-std=c++20','-march=x86-64-v3','-ffp-contract=off','-fopenmp','-shared','-I',str(eigen),str(ROOT/'native/pair_census106_v1.cpp'),'-o',str(out/'pairs.dll')],check=True)
    native=Native(BASE/'spatial.dll')
    oracle=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    fun=bind(ct.CDLL(str(out/'pairs.dll')),'census106',[I,I,I,I,D,D,D,D,D])
    def census(x,prof):
        m=x.shape[1]//2;single=np.empty(m);joint=np.empty((m,m))
        assert fun(m,len(x),len(prof['values']),16,x,np.array(prof['values']),np.array(prof['weights']),single,joint)==0
        return single,joint
    t,r=actual_input();z=inputs()[0];m=len(t);n=(np.sinh(r)**2).sum(0)
    k,rr,e0,eideal=spectrum_fit(t,r)
    one=dict(values=[1.],weights=[1.])
    exact=census(physical(z,np.arange(m))-np.eye(2*m),one)
    cases=[];profiles={};stages=[]
    total_ref=np.load(REF/'total.npz')['target']
    for rank in (-1,0,1,2):
        if rank<1:
            tr=t
        else:
            mo,weight,tr=compress(t,n,rank)
        strength=r if rank==-1 else rr
        fs=factors(tr,strength)
        eig=np.concatenate([np.linalg.eigvalsh(2*f@f.T) for f in fs])
        prof=([p for p in brightness(eideal,eig) if p['feasible']][-1] if rank!=-1 else read(FIELD/'summary.json')['selected'])
        profiles[rank]=prof
        si,ij=census(np.stack([2*f.T@f for f in fs]),prof)
        a=si-exact[0];v=ij-exact[1]
        tv=(abs(v)+abs(a[:,None]-v)+abs(a[None,:]-v)+abs(v-a[:,None]-a[None,:]))/2
        np.fill_diagonal(tv,0)
        indices=np.tril_indices(m,-1);po=tv[indices]
        worst=np.unravel_index(np.argmax(tv),tv.shape)
        far=tv.copy()
        for g in range(m//8):far[g*8:g*8+8,g*8:g*8+8]=0
        far_i,far_j=np.unravel_index(np.argmax(far),far.shape)
        picked=list(map(int,worst))
        while len(picked)<16:
            score=tv[:,picked].sum(1);score[picked]=-1;picked.append(int(score.argmax()))
        w=np.exp(-2j*np.pi*np.arange(513)/1024)
        total=np.fft.irfft(sum(a*np.exp(-.5*np.log1p((1-w[:,None])*b*eig).sum(1)) for a,b in zip(prof['weights'],prof['values'])))
        dz=.5/(max(prof['values'])*eig.max())
        tail=sum(a*np.exp(-.5*np.log1p(-dz*b*eig).sum()-1024*np.log1p(dz)) for a,b in zip(prof['weights'],prof['values']))
        np.savez_compressed(out/f'census_{rank}.npz',two_bit_tv=po,single_gap=a,total=total)
        cases.append(dict(rank=rank,pairs=len(po),worst=list(map(int,worst)),max_tv=float(po.max()),mean_tv=float(po.mean()),p99_tv=float(np.quantile(po,.99)),
                          worst_separate=[int(far_i),int(far_j)],greedy=picked,total_tv=float(abs(total-total_ref).sum()/2),total_alias=float(tail)))
        stages.append((fs,prof))
    # Freeze stress panels from the *baseline* census, then evaluate all variants there.
    ref=cases[0];aa,bb=ref['worst_separate']
    panels=[np.load(REF/f'jiuzhang_{j}.npz')['ports'] for j in range(4)]
    panels+=[np.array(ref['greedy']),np.r_[np.arange(8)+8*(aa//8),np.arange(8)+8*(bb//8)]]
    rng=np.random.default_rng(106100000)
    panels.extend(np.sort(rng.choice(m,16,replace=False)) for _ in range(3))
    for j,ids in enumerate(panels):
        dim=2*len(ids);q=physical(z,ids);target=np.empty(1<<len(ids))
        assert oracle(len(ids),2,q,np.zeros((2,dim)),target,np.empty(2))==0
        values={}
        for row,(fs,prof) in zip(cases,stages):
            p=field_panel(fs,prof,ids,oracle)
            row.setdefault('click_tvs',[]).append(float(abs(p-target).sum()/2));values[str(row['rank'])]=p
        np.savez_compressed(out/f'panel_{j}.npz',target=target,ports=ids,**values)
    print([(row['rank'],row['max_tv'],row['total_tv'],row['click_tvs'][:2]) for row in cases],flush=True)
    save(out,'summary.json',dict(rows=cases,profiles=profiles,redistributed_weak_fraction=k,squeezing=rr.tolist(),panels=[p.tolist() for p in panels]))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','draw'));p.add_argument('output',type=Path)
    p.add_argument('--shots',type=int,default=2048);p.add_argument('--rank',type=int,choices=(1,2),default=1);p.add_argument('--seed',type=int);p.add_argument('--pnr',action='store_true')
    a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):
        if a.action=='build':run(a.output.resolve())
        else:
            if a.seed is None:p.error('Provide an experiment seed')
            with a.output.open('xb') as f:np.save(f,model_case(a.rank)[0].draw(a.shots,a.pnr,a.seed))
