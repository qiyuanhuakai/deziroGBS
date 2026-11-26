"""Shared phase-conjugate Gaussian fields with brightness moments fixed by the optical input."""
import argparse
import ctypes as ct
import math
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import I,U,D,bind,Native,save,manifest,realify
from iteration103_gpu_v1 import P,CU,build
from iteration103_sample_v1 import BASE,DATA
from iteration105_jiuzhang_v1 import inputs,xpair

OUT=DATA/'iteration106_field_v1'


def optical(t,r):
    b=realify(np.linalg.qr(t,mode='reduced')[1])
    eig=[]
    for row in r:
        x=np.zeros((2*t.shape[1],2*t.shape[1]))
        for j,a in enumerate(row):
            ids=np.r_[2*j,2*j+1,t.shape[1]+2*j,t.shape[1]+2*j+1]
            x[np.ix_(ids,ids)]=xpair(a)
        eig.extend(np.linalg.eigvalsh(b@x@b.T))
    return np.array(eig)


def factors(t,r):
    mats=[]
    for row in np.atleast_2d(r):
        scale=np.abs(np.sinh(row))/np.sqrt(2)
        a=t[:,::2];b=t[:,1::2]*np.sign(row)
        f=np.empty_like(t);f[:,::2]=(a+b)*scale
        f[:,1::2]=1j*(a-b)*scale
        mats.append(np.concatenate((f.real.T,f.imag.T),axis=1))
    return np.ascontiguousarray(mats)


def factorial_moments(eigs,degree):
    v=[mp.mpf(float(x)) for x in eigs]
    cumul=[mp.mpf(0)]+[mp.factorial(k-1)*mp.fsum(x**k for x in v)/2 for k in range(1,degree+1)]
    m=[mp.mpf(1)]
    for k in range(1,degree+1):
        m.append(mp.fsum(math.comb(k-1,j-1)*cumul[j]*m[k-j] for j in range(1,k+1)))
    return m


def brightness(ideal,base,max_atoms=6):
    # A positive quadrature with d nodes matches the first 2d-1 moments.
    with mp.workdps(75):
        target=factorial_moments(ideal,2*max_atoms-1)
        proposal=factorial_moments(base,2*max_atoms-1)
        moments=[p/q for p,q in zip(target,proposal)]
        rows=[]
        for d in range(1,max_atoms+1):
            h=mp.matrix([[moments[i+j] for j in range(d)] for i in range(d)])
            shift=mp.matrix([[moments[i+j+1] for j in range(d)] for i in range(d)])
            low=mp.eigsy(h,eigvals_only=True)[0]
            if low<=0:
                rows.append(dict(d=d,feasible=False,min_hankel=float(low)))
                break
            inv=mp.cholesky(h)**-1
            values,vectors=mp.eigsy(inv*shift*inv.T)
            weights=[vectors[0,j]**2 for j in range(d)]
            if min(values)<0:
                rows.append(dict(d=d,feasible=False,min_support=float(min(values))))
                break
            err=max(abs(mp.fsum(w*a**k for w,a in zip(weights,values))-moments[k]) for k in range(2*d))
            assert err<mp.mpf('1e-48')
            rows.append(dict(d=d,feasible=True,values=list(map(float,values)),weights=list(map(float,weights)),relative_residual=float(err)))
        return rows


def actual_input():
    z,_,_=inputs()
    return z['T'][np.argsort(z['order'])],z['squeezing'].T


class Field:
    def __init__(self,t,r,profile,wide=False,dll=OUT/'gpu.dll'):
        self.m=len(t);self.s,self.k=np.shape(r)
        if t.shape!=(self.m,2*self.k) or min(self.s,self.k,self.m)<1:raise ValueError('Invalid optical transfer dimensions')
        self.dtype=torch.float64 if wide else torch.float32
        self.wide=wide;self.lib=ct.CDLL(str(dll))
        self.noise=self.lib.noise106_launch
        self.noise.argtypes=[I,I,I,U,CU,I,P,P]
        self.detect=self.lib.detect106_launch
        self.detect.argtypes=[I,I,I,I,U,CU,I,P,I,P,P,P,P,P]
        self.factor=torch.as_tensor(factors(t,r),device='cuda',dtype=self.dtype)
        self.error=torch.zeros(1,dtype=torch.int32,device='cuda')
        self.set_profile(profile)
        self.capacity=0
    def set_profile(self,profile):
        self.atoms=len(profile['values'])
        self.values=torch.as_tensor(profile['values'],dtype=torch.float64,device='cuda')
        cdf=np.cumsum(profile['weights']);cdf[-1]=1
        self.cdf=torch.as_tensor(cdf,dtype=torch.float64,device='cuda')
    def draw(self,n,pnr,seed,offset=0):
        if n<1 or n*max(self.s*self.k,self.m)>=2**31 or offset<0 or offset+n>2**32 or not 0<=seed<2**64:raise ValueError('Invalid record domain')
        # Contiguous spectrum-major buffers make every CUDA row stride exactly n.
        if self.capacity<n:
            self.z=torch.empty((self.s*n*2*self.k,),dtype=self.dtype,device='cuda')
            self.f=torch.empty((self.s*n*2*self.m,),dtype=self.dtype,device='cuda')
            self.x=torch.empty((n,self.m),dtype=torch.int32,device='cuda');self.capacity=n
        z=self.z[:self.s*n*2*self.k].view(self.s,n,2*self.k);f=self.f[:self.s*n*2*self.m].view(self.s,n,2*self.m)
        stream=torch.cuda.current_stream().cuda_stream
        rc=self.noise(self.k,self.s,n,seed,offset,int(self.wide),z.data_ptr(),stream)
        if rc:raise RuntimeError(rc)
        torch.bmm(z,self.factor,out=f)
        rc=self.detect(self.m,self.s,n,int(pnr),seed,offset,int(self.wide),f.data_ptr(),self.atoms,self.values.data_ptr(),self.cdf.data_ptr(),self.x.data_ptr(),self.error.data_ptr(),stream)
        if rc:raise RuntimeError(rc)
        x=self.x[:n].cpu().numpy()
        if self.error.item():raise ArithmeticError('Invalid photoelectron rate')
        return x


def evaluate(t,r,out):
    fs=factors(t,r);m=len(t)
    eig=np.concatenate([np.linalg.eigvalsh(2*f@f.T) for f in fs])
    ideal=optical(t,r)
    profiles=brightness(ideal,eig)
    selected=[p for p in profiles if p['feasible']][-1]
    # The last feasible distribution can have a very small extreme atom.
    # Directly evaluate the rounded double-precision support and weights.
    rows=[]
    h=np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf'][:1024]
    w=np.exp(-2j*np.pi*np.arange(513)/1024)
    oracle=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    for profile in [profiles[0],selected] if selected['d']>1 else [selected]:
        values,weights=profile['values'],profile['weights']
        prob=sum(weight*np.exp(-.5*np.log1p((1-w[:,None])*v*eig).sum(1)) for weight,v in zip(weights,values))
        total=np.fft.irfft(prob);panels=[]
        for j in range(4):
            ref=np.load(DATA/f'iteration106_reference_v1/jiuzhang_{j}.npz')
            ids=ref['ports'];idx=np.r_[ids,m+ids];dim=len(idx)
            x=np.stack([2*f[:,idx].T@f[:,idx] for f in fs]);p=np.zeros(len(ref['target']))
            for weight,v in zip(weights,values):
                law=np.empty_like(p)
                assert oracle(len(ids),len(fs),np.eye(dim)[None,:,:]+v*x,np.zeros((len(fs),dim)),law,np.empty(2))==0
                p+=weight*law
            panels.append(float(abs(p-ref['target']).sum()/2))
            np.savez_compressed(out/f'field_{profile["d"]}_panel_{j}.npz',model=p,target=ref['target'],ports=ids)
        nz=eig[eig>1e-12]
        tail=0.
        for weight,v in zip(weights,values):
            dz=np.linspace(.05,.95,100)/(v*nz.max())
            tail+=weight*np.exp((-.5*np.log1p(-dz[:,None]*v*nz).sum(1)-1024*np.log1p(dz)).min())
        rows.append(dict(d=profile['d'],click_tvs=panels,total_tv=float(abs(total-h).sum()/2),total_ks=float(abs(total.cumsum()-h.cumsum()).max()),total_fft_tail=float(tail)))
        np.savez_compressed(out/f'field_{profile["d"]}_total.npz',model=total,target=h)
    save(out,'summary.json',dict(rows=rows,profiles=profiles,selected=selected,source='High-power optical transfer and squeezing; brightness determined from factorial moments of total photon count alone.'))
    print(rows,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','draw'));p.add_argument('output',type=Path)
    p.add_argument('--input',type=Path,help='Optional NPZ with transfer (ports, 2*sources) and squeezing (spectra, sources)');p.add_argument('--shots',type=int,default=2048);p.add_argument('--seed',type=int);p.add_argument('--pnr',action='store_true')
    a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):
        if a.action=='build':
            a.output.mkdir(parents=True,exist_ok=False)
            build(a.output,'coherent_field106_v1.cu')
            native=Native(BASE/'spatial.dll')
            evaluate(*actual_input(),a.output)
            manifest(a.output)
        else:
            if a.seed is None:p.error('Provide an experiment seed')
            import json
            if a.input is None:
                t,r=actual_input();profile=json.loads((OUT/'summary.json').read_text())['selected']
            else:
                with np.load(a.input) as data:t=np.asarray(data['transfer'],complex);r=np.atleast_2d(data['squeezing'])
                if t.ndim!=2 or r.ndim!=2 or t.shape[1]!=2*r.shape[1] or not np.isfinite(t).all() or not np.isfinite(r).all():raise ValueError('Expected finite twin-beam inputs')
                if np.linalg.eigvalsh(t.conj().T@t).max()>1+1e-10:raise ValueError('Transfer must be passive')
                f0=factors(t,r);eig=np.concatenate([np.linalg.eigvalsh(2*x@x.T) for x in f0])
                profile=([p for p in brightness(optical(t,r),eig) if p['feasible']][-1] if eig.max()>0 else dict(values=[1.],weights=[1.]))
            f=Field(t,r,profile)
            with a.output.open('xb') as dest:np.save(dest,f.draw(a.shots,a.pnr,a.seed))
