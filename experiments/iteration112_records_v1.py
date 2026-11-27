"""Return full GPU records with sparse, quantum-correlated photon measurements in occupied cells."""
import argparse
import ctypes as ct
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration103_gpu_v1 import build as build_gpu,I,U,CU,P
from iteration111_sampling_v1 import ROOT,DATA,pnr_table
from iteration112_dilution_v1 import (optics, click_effects, pnr_effects, number_law, marginal_overflow,
                                      overlap_factor, certificate)


def compositions(n,r):
    if r==1:yield (n,);return
    for i in range(n+1):
        for tail in compositions(n-i,r-1):yield (i,)+tail


def fock(s,cap):
    r=len(s);dims=[];ini=[];trans=[];fac=[];prev=[]
    for n in range(cap+1):
        states=list(compositions(n,r))
        index={v:i for i,v in enumerate(states)}
        dims.append(len(states))
        if prev:
            parent=np.empty((len(prev),r),np.int32)
            factor=np.empty((len(prev),r))
            for j,v in enumerate(prev):
                for k in range(r):
                    key=v[:k]+(v[k]+1,)+v[k+1:]
                    parent[j,k]=index[key];factor[j,k]=math.sqrt(v[k]+1)
            trans.append(parent);fac.append(factor)
        a=np.zeros(len(states))
        if n%2==0:
            for j,v in enumerate(states):
                if not any(i%2 for i in v):
                    a[j]=math.prod((-math.tanh(t)/2)**(i//2)*math.sqrt(math.factorial(i))/math.factorial(i//2) for t,i in zip(s,v))
            a/=np.linalg.norm(a)
        ini.append(a)
        prev=states
    return [np.array(dims,np.int32),np.cumsum([0]+dims[:-1],dtype=np.int32),np.vstack(trans),np.vstack(fac),np.concatenate(ini)]


def alias(q):
    size=1<<(len(q)-1).bit_length()
    mass=np.pad(q/q.sum(),(0,size-len(q))).astype(np.longdouble)*size
    lo=list(np.flatnonzero(mass<1));hi=list(np.flatnonzero(mass>=1))
    frac=np.ones(size,np.longdouble);other=np.arange(size,dtype=np.uint32)
    while lo and hi:
        i,j=lo.pop(),hi.pop()
        frac[i]=mass[i];other[i]=j;mass[j]-=1-mass[i]
        (lo if mass[j]<1 else hi).append(j)
    threshold=np.rint(frac*2**53).astype(np.uint64)
    prob=threshold.astype(float)/2**53
    law=(prob+np.bincount(other,weights=1-prob,minlength=size))/size
    error=float(abs(law-np.pad(q,(0,size-len(q)))).sum()/2)
    return threshold,other,error


class Device:
    def __init__(self,out):
        build_gpu(out,'local_register112_v1.cu')
        self.gpu=ct.CDLL(str(out/'gpu.dll'))
        self.emit=self.gpu.local_emit112;self.emit.argtypes=[I,I,I,CU,U,CU]+[P]*5
        self.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent)) if os.name=='nt' else None
        dll=out/'sparse.dll'
        subprocess.run(['g++','-std=c++17','-O3','-fopenmp','-shared','-fPIC',str(ROOT/'native/dilute_photons112_v1.cpp'),'-o',str(dll)],check=True)
        self.cpu=ct.CDLL(str(dll));self.defects=self.cpu.sparse_emit112
        self.defects.argtypes=[I]*6+[U,CU]+[P]*12


class Model:
    def __init__(self,lib,q,e,words,s,cap,cache,pnr):
        self.lib,self.cap,self.pnr=lib,cap,pnr
        self.r=len(s);self.words=np.ascontiguousarray(words,np.int32);self.s=s
        self.buff=None;self.size=len(q);self.q=q
        val,vec=np.linalg.eigh(e)
        assert val.min()>-1e-12
        positive=(vec*np.maximum(val,0)[:,None,:])@vec.conj().transpose(0,2,1)
        # The original effects sum to identity. Normalize the roundoff-level
        # PSD correction by congruence, so prefix measurement is a POVM.
        a,v=np.linalg.eigh(positive.sum(0))
        inv=(v/np.sqrt(a))@v.conj().T
        effect=inv@positive@inv
        self.effect_error=float(np.linalg.norm(effect-e,ord=2,axis=(1,2)).sum()/2)
        eig,vec=np.linalg.eigh(effect)
        self.prefix=np.ascontiguousarray(effect.cumsum(0))
        self.factor=np.ascontiguousarray(vec*np.sqrt(np.maximum(eig,0))[:,None,:])
        self.fock=cache
        cdf,other,self.rounding=alias(q)
        self.tables=[torch.as_tensor(np.ascontiguousarray(v),device='cuda') for v in (cdf,other,np.pad(words,(0,len(cdf)-len(words))).astype(np.uint32))]

    def configure(self,cells):
        self.cells=cells
        n=2*np.arange(self.cap//2+1)
        mass=number_law(self.s,self.cap)*[np.prod(1-np.arange(i)/cells) if i<=cells else 0. for i in n]
        law=np.zeros(self.cap+1);law[n]=mass/mass.sum()
        self.cdf=np.minimum(1,law.cumsum());self.cdf[-1]=1.
        self.buff=None

    def draw(self,n,seed,off=0,repair=True):
        if n<1 or self.cap>self.cells or not 0<=off<=2**32-n:raise ValueError('Invalid sampling dimensions')
        size=n*self.cells
        if self.buff is None or self.buff.numel()<size:self.buff=torch.empty(size,device='cuda',dtype=torch.int32)
        gpu=self.buff[:size].view(n,self.cells)
        rc=self.lib.emit(n,self.cells,len(self.tables[0]),int(self.pnr),seed,off,*[v.data_ptr() for v in self.tables],
                         gpu.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if rc:raise RuntimeError(rc)
        if repair:
            loc=np.empty((n,self.cap),np.int32);pkt=np.empty_like(loc);ns=np.empty(n,np.int32)
            args=[self.cdf,*self.fock,self.prefix,self.factor,self.words,loc,pkt,ns]
            rc=self.lib.defects(n,self.cells,self.r,self.cap,self.size,8,seed,off,*[v.ctypes.data for v in args])
            if rc:raise ArithmeticError(rc)
        out=gpu.cpu().numpy()
        if repair:
            rows,cols=np.nonzero(np.arange(self.cap)<ns[:,None])
            out[rows,loc[rows,cols]]=pkt[rows,cols]
        return out.view(np.uint8).reshape(n,4*self.cells)


def run(out,radix):
    out.mkdir(exist_ok=True,parents=True)
    lib=Device(out)
    zero=np.zeros((1,1,8),complex)
    x=optics(1)[0];base=pnr_table(x,zero,radix)[:,0,0].real
    order=np.argsort(-base);keep=order[:np.searchsorted(np.cumsum(base[order],dtype=np.longdouble),1-2e-10)+1]
    arrays={};summary=[]
    for buses in (1,4):
        x,bright,s,ket=optics(buses);r=len(s)
        cap=26 if r<=2 else 12
        cache=fock(s,cap)
        for pn in (False,True):
            start=perf_counter()
            if pn:q,e=pnr_effects(x,ket,radix,keep);codes=keep;rad=radix
            else:q,e=click_effects(x,ket);codes=np.arange(16);rad=2
            digits=np.stack([codes//rad**i%rad for i in range(4)],axis=1)
            words=(digits.astype(np.uint32)@(np.uint32(1)<<(8*np.arange(4,dtype=np.uint32)))).view(np.int32)
            lam,_=overlap_factor(q,e)
            engine=Model(lib,q,e,words,s,cap,cache,int(pn))
            setup=perf_counter()-start
            for cells in (256,32768,262144):
                engine.configure(cells)
                extra=0.
                if pn:
                    local=x+(bright-x)/cells
                    law=pnr_table(local,zero,radix)[:,0,0].real
                    extra=cells*(max(0.,float(1-np.sum(law[keep],dtype=np.longdouble)))+marginal_overflow(local,radix))
                tail=max(0.,1-number_law(s,cap).sum())
                # Bound collisions with the full squeezed-number law; separately
                # charge the tail of the actual cap after the parity projection.
                collision=certificate(s,cells,lam,cut=80)['collision_probability_upper']
                clean_tail=min(1.,tail/max(1-collision,1e-30))
                budget=min(1.,np.sqrt(collision)+clean_tail+extra+cells*engine.rounding+cap*engine.effect_error)
                no_repair=min(1.,certificate(s,cells,lam,cut=64)['optical_measurement_tv_budget']+extra+cells*engine.rounding)
                seed=112300000+10000*buses+1000*pn+10*int(np.log2(cells))
                size=max(8,min(2048,4194304//(4*cells)))
                times={}
                for repaired in (False,True):
                    tm=[]
                    for rep in range(9):
                        start=perf_counter()
                        arr=engine.draw(size,seed+rep,repair=repaired)
                        if rep>=2:tm.append((perf_counter()-start)*1e6/size)
                    times[str(int(repaired))]=tm
                shots=512 if cells==256 else 128
                total=[];first=[]
                for offset in range(0,shots,min(size,shots)):
                    arr=engine.draw(min(size,shots-offset),seed+100,offset)
                    total.extend(arr.sum(1,dtype=np.int64).tolist())
                    if not offset:
                        first=arr[:4].copy()
                        assert np.array_equal(first[1:3],engine.draw(2,seed+100,1))
                name=f'b{buses}p{int(pn)}c{cells}'
                arrays[name+'_first']=np.packbits(first,axis=1,bitorder='little') if not pn else first
                arrays[name+'_total']=np.array(total)
                summary.append(dict(buses=buses,r=r,pnr=pn,radix=rad,cells=cells,ports=4*cells,cap=cap,shots=shots,
                    setup_s=setup,kept=len(q),batch=size,timing=times,collision_probability_upper=collision,
                    selected_tail=clean_tail,register_tail=extra,rounding=engine.rounding,effect_correction=engine.effect_error,
                    tv_budget=budget,product_tv_budget=no_repair,mean_observed=float(np.mean(total))))
                print(name,budget,no_repair,[round(np.median(t),2) for t in times.values()],flush=True)
            np.savez_compressed(out/'records.npz',**arrays)
            (out/'records.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    ap.add_argument('--radix',type=int,default=30)
    args=ap.parse_args()
    with threadpool_limits(limits=3):run(args.output.resolve(),args.radix)
