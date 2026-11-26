"""Share proposal tables across identical distinguishable spectra and sample the occupied spectra directly."""
import argparse
import ctypes as ct
from pathlib import Path
import shutil
import subprocess
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,manifest
from iteration103_gpu_v1 import I,U,CU,P
from iteration106_sample_v1 import DATA
from iteration108_compact_v1 import Compact

OUT=DATA/'iteration108_replicas_v1'


class Replicas(Compact):
    def __init__(self,t,r,replicas,cap=4,tries=2048,workers=12,cache_mb=64):
        super().__init__(t,r,cap,tries,workers)
        if self.s!=1 or not 1<=replicas<=65536 or cache_mb<0:raise ValueError('Expected one spectral profile and 1..65536 orthogonal replicas')
        self.replicas=replicas;self.replib=ct.CDLL(str(OUT/'repeat.dll'))
        self.fn=self.replib.repeat108_draw;self.fn.argtypes=[I]*10+[U,CU]+[P]*12
        k=min(int(cache_mb*1024**2)//(8*self.m),self.m)
        prob=np.diff(self.single[0,2+self.m:],prepend=0.)
        order=np.argsort(-prob,kind='stable')[:k];order=order[prob[order]>0]
        self.cached_mass=float(prob[order].sum())
        self.index=np.full(self.m,-1,np.int32)
        self.cached=np.empty((max(1,len(order)),self.m))
        for lo in range(0,len(order),32):
            ids=order[lo:lo+32];a=t[ids].copy()
            a[:,::2]=t[ids,1::2]*self.lam[0];a[:,1::2]=t[ids,::2]*self.lam[0]
            weight=abs(a@t.T)**2
            cdf=np.cumsum(weight,axis=1,dtype=np.longdouble);cdf/=cdf[:,-1,None]
            self.cached[lo:lo+len(ids)]=cdf
            self.cached[lo:lo+len(ids),-1]=1.;self.index[ids]=np.arange(lo,lo+len(ids))
        risk=self.cert['full_joint_tv_bound']
        self.bound=float(-np.expm1(replicas*np.log1p(-risk))) if risk<1 else 1.

    def draw(self,n,pnr,seed,offset=0,stats=False):
        if not 1<=n<2**31 or not 0<=seed<2**64 or not 0<=offset<=2**32-n:raise ValueError('Invalid record domain')
        for wide in (0,1):
            x=np.empty((n,self.m),dtype=np.uint32 if wide else np.uint8);counts=np.zeros((self.cap+2,3),np.uint64)
            rc=self.fn(self.m,self.k,self.rows,self.replicas,n,self.cap,self.tries,int(pnr),wide,self.workers,seed,offset,
                       *(v.ctypes.data for v in [*self.args,self.single,self.index,self.cached,x,counts]))
            if not rc:return (x,counts) if stats else x
            if rc!=1:raise ArithmeticError(rc)
        raise ArithmeticError('Photon counter overflow')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','draw'));p.add_argument('output',type=Path)
    p.add_argument('--input',type=Path);p.add_argument('--replicas',type=int,default=1024);p.add_argument('--shots',type=int,default=128)
    p.add_argument('--cap',type=int,default=4);p.add_argument('--tries',type=int,default=2048);p.add_argument('--seed',type=int);p.add_argument('--offset',type=int,default=0)
    p.add_argument('--pnr',action='store_true');p.add_argument('--cache-mb',type=int,default=64)
    a=p.parse_args()
    with threadpool_limits(limits=4,user_api='blas'):
        if a.action=='build':
            a.output.mkdir(exist_ok=False,parents=True)
            subprocess.run([shutil.which('g++'),'-std=c++20','-O3','-march=x86-64-v3','-ffp-contract=off','-fopenmp','-shared',str(ROOT/'native/replicated_matching108_v1.cpp'),'-o',str(a.output/'repeat.dll')],check=True)
            manifest(a.output)
        else:
            if a.input is None or a.seed is None:p.error('Specify input and seed')
            with np.load(a.input) as z:model=Replicas(z['transfer'],z['squeezing'],a.replicas,a.cap,a.tries,cache_mb=a.cache_mb)
            with a.output.open('xb') as f:np.save(f,model.draw(a.shots,a.pnr,a.seed,a.offset))
