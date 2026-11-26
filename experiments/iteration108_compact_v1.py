"""Resolve single-pair losses analytically and return dense records or lossless sparse rows."""
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
from iteration108_matching_v1 import Matching,cum

OUT=DATA/'iteration108_compact_v1'


class Compact(Matching):
    def __init__(self,t,r,cap=4,tries=2048,workers=12,dll=OUT/'compact.dll'):
        super().__init__(t,r,cap,tries,workers,dll)
        gram=t.conj().T@t
        tables=[]
        for l in self.lam:
            norm=2*(l@l);one=np.zeros(self.m);two=np.zeros(self.m)
            if norm:
                for lo in range(0,self.m,2048):
                    sl=slice(lo,min(lo+2048,self.m));a=t[sl].copy()
                    a[:,::2]=t[sl,1::2]*l;a[:,1::2]=t[sl,::2]*l
                    detected=np.einsum('ij,ij->i',a.conj(),a@gram.T).real
                    both=detected/norm;all_first=(np.abs(a)**2).sum(1)/norm
                    two[sl]=both.clip(0);one[sl]=2*(all_first-both).clip(0)
            zero=1-one.sum()-two.sum()
            if zero< -1e-9:raise ArithmeticError('Single-pair probabilities are inconsistent')
            tables.append(np.r_[np.clip(zero,0,1),np.clip(1-two.sum(),0,1),cum(one),cum(two)])
        self.single=np.ascontiguousarray(tables)
        self.fn=self.lib.compact108_draw
        self.fn.argtypes=[I]*12+[U,CU]+[P]*13

    def draw(self,n,pnr,seed,offset=0,sparse=False,slots=4,stats=False):
        if not 1<=n<2**31 or not 0<=seed<2**64 or not 0<=offset<=2**32-n or not 1<=slots<2**31:raise ValueError('Invalid record request')
        dummy=np.empty(1,np.uint32);lengths=np.empty(n,np.int32);wide=0
        while True:
            x=dummy if sparse else np.empty((n,self.m),np.uint32 if wide else np.uint8)
            ids=np.empty((n,slots),np.int32) if sparse else dummy
            mult=np.empty((n,slots),np.uint32) if sparse and pnr else dummy
            counts=np.zeros((self.s,self.cap+2,3),np.uint64)
            rc=self.fn(self.m,self.k,self.s,self.rows,n,self.cap,self.tries,int(pnr),self.workers,int(sparse),slots,wide,seed,offset,
                *(v.ctypes.data for v in [*self.args,self.single,x,lengths,ids,mult,counts]))
            if not rc:break
            if rc==3 and sparse and lengths.max()>slots:slots=int(lengths.max());continue
            if rc==1 and not sparse and not wide:wide=1;continue
            raise ArithmeticError(f'Sparse matching failed ({rc})')
        if sparse:
            use=np.arange(slots)<lengths[:,None]
            x=dict(width=np.int64(self.m),offsets=np.r_[0,lengths.cumsum(dtype=np.int64)],ports=ids[use])
            if pnr:x['counts']=mult[use]
        return (x,counts) if stats else x


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','draw'));p.add_argument('output',type=Path)
    p.add_argument('--input',type=Path);p.add_argument('--shots',type=int,default=2048);p.add_argument('--seed',type=int);p.add_argument('--offset',type=int,default=0)
    p.add_argument('--cap',type=int,default=4);p.add_argument('--tries',type=int,default=2048);p.add_argument('--pnr',action='store_true');p.add_argument('--dense',action='store_true')
    a=p.parse_args()
    with threadpool_limits(limits=4,user_api='blas'):
        if a.action=='build':
            a.output.mkdir(exist_ok=False,parents=True)
            subprocess.run([shutil.which('g++'),'-std=c++20','-O3','-march=x86-64-v3','-ffp-contract=off','-fopenmp','-shared',str(ROOT/'native/compact_matching108_v1.cpp'),'-o',str(a.output/'compact.dll')],check=True)
            manifest(a.output)
        else:
            if a.input is None or a.seed is None:p.error('Specify input and experiment seed')
            with np.load(a.input) as z:model=Compact(z['transfer'],z['squeezing'],a.cap,a.tries)
            record=model.draw(a.shots,a.pnr,a.seed,a.offset,not a.dense)
            with a.output.open('xb') as f:
                if a.dense:np.save(f,record)
                else:np.savez_compressed(f,**record)
