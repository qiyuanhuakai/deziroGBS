"""Two coherent squeezed modes per distinguishable sector, with a passive rectangular transfer and loss."""
import argparse
import ctypes as ct
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,I,U,D,T,J,W,bind,save,manifest
from iteration97_forest_v1 import alias_table
from iteration104_spectral_v1 import discrete
from iteration103_sample_v1 import DATA

OUT=DATA/'iteration105_two_mode_v1'


def sectors(r,twin,eps):
    with mp.workdps(65):
        x,y=map(mp.mpf,map(float,r))
        if twin:
            if x!=y:raise ValueError('A signal-idler pair needs one common squeezing')
            a=mp.tanh(x)**2
            cap=0 if not a else int(mp.ceil(mp.log(eps)/mp.log(a)))
            p=[(1-a)*a**k for k in range(cap+1)]
            return discrete(p),np.empty(0),float(a**(cap+1)),2*cap
        def cutoff(v):
            a=mp.tanh(v)**2
            if not a:return 0
            return int(mp.ceil(mp.log(eps*(1-a)/2)/mp.log(a)))
        cap=cutoff(x)+cutoff(y)
        def amplitudes(v):
            val=[1/mp.sqrt(mp.cosh(v))]
            for j in range(1,cap+1):val.append(-mp.tanh(v)*mp.sqrt(mp.mpf(2*j-1)/(2*j))*val[-1])
            return val
        left,right=amplitudes(x),amplitudes(y)
        mass=[];pieces=[]
        for n in range(cap+1):
            a=[left[k]*right[n-k] for k in range(n+1)]
            prob=mp.fsum(z*z for z in a)
            if prob:
                row=np.zeros(2*n+1)
                row[::2]=[float(v/mp.sqrt(prob)) for v in a]
            else:row=np.zeros(2*n+1)
            pieces.append(row);mass.append(prob)
        tail=max(0.,float(1-mp.fsum(mass)))
        assert tail<=eps
        return discrete(mass),np.concatenate(pieces),tail,2*cap


def ports(matrix):
    gram=np.eye(2)-matrix.conj().T@matrix
    e,v=np.linalg.eigh(gram)
    if e[0]<-1e-12:raise ValueError('The optical transfer must be passive')
    whole=np.vstack([matrix,np.sqrt(e.clip(0))[:,None]*v.conj().T])
    residual=float(abs(whole.conj().T@whole-np.eye(2)).max())
    assert residual<2e-12
    w=abs(whole[:,0])**2+abs(whole[:,1])**2
    selected=np.flatnonzero(w>0)
    scaled=w[selected].astype(np.longdouble)
    budget=2**53-len(selected)
    sums=np.r_[np.longdouble(0),np.cumsum(scaled,dtype=np.longdouble)]
    cumul=np.floor(sums/sums[-1]*budget).astype(np.int64)
    cumul[-1]=budget
    mass=np.zeros(len(w),np.int64)
    mass[selected]=np.diff(cumul)+1
    bits,cut,al,keys,err,_=alias_table({i:int(x) for i,x in enumerate(mass)})
    assert err==0 and mass.sum()==2**53
    ratio=w[selected]/(mass[selected]/2**53)
    envelope=float(ratio.max())*(1+8*np.finfo(float).eps)
    directions=np.zeros((len(w),8))
    uu=whole[selected]/np.sqrt(w[selected,None])
    cross=uu[:,0]*uu[:,1].conj()
    directions[selected]=np.c_[uu[:,0].real,uu[:,0].imag,uu[:,1].real,uu[:,1].imag,abs(uu[:,0])**2,cross.real,cross.imag,ratio/envelope]
    return bits,cut,keys[al].astype(np.uint32),directions,residual,envelope


class TwoMode:
    def __init__(self,dll,transfer,squeezing,twin=None,epsilon=1e-13,independent_arms=False):
        self.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
        self.native=bind(ct.CDLL(str(dll)),'two105_draw',[I,I,I,I,I,U,ct.c_uint,I,I,J,W,T,D,W,T,D,D,T])
        trans=np.asarray(transfer,complex);r=np.asarray(squeezing,float)
        if trans.ndim==2:trans=trans[None,:,:]
        if r.ndim==1:r=r[None,:]
        if trans.ndim!=3 or r.shape!=(len(trans),2) or trans.shape[2]!=2 or not np.isfinite(r).all() or not np.isfinite(trans).all():
            raise ValueError('Expected two squeezed inputs per spectral sector')
        self.s,self.m,_=trans.shape
        if not 1<=self.s<=65536 or self.m<1 or not 0<epsilon<1:raise ValueError('Invalid dimensions or tail tolerance')
        if twin is None:twin=np.zeros(self.s,bool)
        if np.shape(twin)!=(self.s,):raise ValueError('One pair flag is required per sector')
        tc=[];ta=[];cuts=[];alt=[];dirs=[];waves=[];meta=[]
        self.tail=self.rounding=self.isometry=0.;self.max_n=0;self.envelope=0.
        pos=statepos=0;max_output=0
        for u,rr,t in zip(trans,r,twin):
            (nbit,ctab,nalt,tv),psi,tail,cap=sectors(rr,t,epsilon/self.s)
            self.bits,c,a,d,orth,en=ports(u)
            if independent_arms and not t:raise ValueError('Independent arm control requires twin beams')
            meta.append([nbit,pos,2 if independent_arms else int(t),statepos])
            tc.append(ctab);ta.append(nalt);waves.append(psi);dirs.append(d);cuts.append(c);alt.append(a)
            pos+=len(ctab);statepos+=len(psi)
            self.max_n=max(self.max_n,cap);self.tail+=tail;self.rounding+=tv
            self.isometry=max(self.isometry,orth);self.envelope=max(self.envelope,en);max_output+=cap
        if max(pos,statepos)>=2**31 or max_output>=2**32:raise ValueError('Optical tables exceed native index range')
        self.args=[np.array(meta,np.int32),np.concatenate(tc),np.concatenate(ta),np.concatenate(waves),np.concatenate(cuts),np.concatenate(alt),np.concatenate(dirs),np.sqrt(np.arange(self.max_n+1,dtype=float))]
    def draw(self,n,pnr,seed,offset=0,workers=1):
        if n<1 or n>=2**31 or workers<1 or not 0<=seed<2**64 or offset<0 or offset+n>2**32:raise ValueError('Invalid sample request')
        output=np.empty((n,self.m),np.uint32)
        rc=self.native(self.m,self.s,n,int(pnr),workers,seed,offset,self.bits,self.max_n,*self.args,output)
        if rc:raise ArithmeticError(f'Two-mode kernel returned {rc}')
        return output


def example(m):
    if m<2:raise ValueError('Two ports required for the example')
    rng=np.random.default_rng(105000000+m)
    mats=[]
    for j in range(2):
        q=rng.normal(size=(m,2))+1j*rng.normal(size=(m,2))
        q[0,0]+=.45*np.sqrt(m);q[-1,1]+=.5j*np.sqrt(m)
        q=np.linalg.qr(q,mode='reduced')[0]
        rotation=np.array([[1,1j],[1j,1]])/np.sqrt(2)
        mats.append(q@np.diag([.88,.72])@rotation)
    return np.array(mats),np.array([[.86,.66],[.57,-.43]])


def bench(out):
    out.mkdir(exist_ok=False,parents=True)
    dll=out/'two.dll'
    subprocess.run([shutil.which('g++'),'-std=c++20','-O3','-march=x86-64-v3','-ffp-contract=off','-shared','-fopenmp',str(ROOT/'native/two_mode105_v1.cpp'),'-o',str(dll)],check=True)
    # Three spectrally distinguishable pairs on disjoint balanced transfer
    # columns: each pair must still emit when an earlier source is occupied.
    disjoint=np.zeros((3,6,2))
    for i in range(3):disjoint[i,2*i:2*i+2]=np.eye(2)
    test=TwoMode(dll,disjoint,np.ones((3,2))*.85,np.ones(3,bool))
    check=test.draw(65536,True,105003000,workers=8)
    assert np.array_equal(check[:,::2],check[:,1::2])
    assert (abs(check.mean(0)-np.sinh(.85)**2)<.035).all()
    assert np.sum(np.all(check[:,::2]>0,axis=1))>2000
    rows=[]
    for m in (2,4,16,144,1152,4097,16384):
        t,r=example(m)
        start=perf_counter();model=TwoMode(dll,t,r);prep=perf_counter()-start
        if m in (2,16):np.savez_compressed(out/f'input_{m}.npz',transfer=t,squeezing=r)
        for pnr in (False,True):
            for workers in ((1,16) if m==16 else (16,)):
                n=2048 if m<8192 else 512
                times=[]
                for rep in range(9):
                    start=perf_counter();model.draw(n,pnr,105001000+m,n*rep,workers)
                    if rep>1:times.append((perf_counter()-start)*1e6/n)
                rows.append(dict(m=m,pnr=pnr,workers=workers,batch=n,us=float(np.median(times)),timing=times,prep_s=prep,tail=model.tail,initial_tv=model.rounding,orthogonality_error=model.isometry,max_envelope=model.envelope))
        x=model.draw(301,False,105002000+m,workers=1)
        y=model.draw(264,True,105002000+m,37,16)
        assert np.array_equal(x[37:],y>0)
    save(out,'summary.json',dict(rows=rows,squeezing=r.tolist(),scope='Two coherent squeezed modes within each of two distinguishable sectors; arbitrary passive transfer, including mode-dependent loss.'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('benchmark','draw'));p.add_argument('output',type=Path)
    p.add_argument('--input',type=Path);p.add_argument('--shots',type=int,default=2048);p.add_argument('--seed',type=int);p.add_argument('--workers',type=int,default=1);p.add_argument('--pnr',action='store_true')
    a=p.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if a.action=='benchmark':bench(a.output.resolve())
        else:
            if a.input is None or a.seed is None:p.error('--input and --seed are required')
            with np.load(a.input) as z:model=TwoMode(OUT/'two.dll',z['transfer'],z['squeezing'],z.get('twin'))
            x=model.draw(a.shots,a.pnr,a.seed,workers=a.workers)
            with a.output.open('xb') as f:np.save(f,x)
