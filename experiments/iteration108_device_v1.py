"""Dense direct-threshold and PNR readout from globally corrected optical strata."""
import argparse
import ctypes as ct
import hashlib
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import manifest
from iteration103_gpu_v1 import build,I,U,CU,P
from iteration106_field_v1 import factors
from iteration106_sample_v1 import DATA
from iteration107_fast_v1 import OUT as FAST
from iteration108_laws_v1 import prepare

OUT=DATA/'iteration108_device_v1'


def fingerprint(t,r):
    h=hashlib.sha256()
    for x in (np.ascontiguousarray(t,dtype='<c16'),np.ascontiguousarray(r,dtype='<f8')):
        h.update(np.array(x.shape,dtype='<i8').tobytes());h.update(x.tobytes())
    return h.hexdigest()


class Global:
    def __init__(self,t,r,cfg,dll=OUT/'gpu.dll'):
        self.lib=ct.CDLL(str(dll));self.normlib=ct.CDLL(str(FAST/'gpu.dll'))
        self.normal=self.normlib.normals107_launch;self.normal.argtypes=[I,I,I,U,CU,I,P,P,I,P,P,P]
        self.rates=self.lib.rates108_launch;self.rates.argtypes=[I,I,I]+[P]*5
        self.read=self.lib.read108_launch;self.read.argtypes=[I]*8+[U,CU]+[P]*11
        r=np.atleast_2d(r);self.s,self.k=r.shape;self.m=len(t);self.h=int(cfg['h'])
        if 'input_hash' in cfg and str(cfg['input_hash'])!=fingerprint(t,r):raise ValueError('Tables belong to another optical input')
        if t.shape!=(self.m,2*self.k) or self.s*self.k>=65536*((65535-222)//8) or self.m>=65536*((65535-280)//16):raise ValueError('Input bounds exceeded')
        def upload(x):return torch.as_tensor(x,device='cuda')
        self.order=upload(cfg['order']);self.bounds=upload(cfg['bounds']);self.groups=len(self.bounds)-1
        self.shape=list(map(int,cfg['shape']))
        self.region=upload(np.searchsorted(cfg['bounds'][1:],np.arange(self.m),side='right').astype(np.int32))
        self.factor=upload(factors(t[cfg['order']],r).astype(np.float32))
        self.val=upload(cfg['values'].astype(float));cdf=cfg['weights'].astype(float).cumsum();cdf[-1]=1
        self.cdf=upload(cdf);self.error=torch.zeros(1,device='cuda',dtype=torch.int32)
        self.rule=[(upload(cfg[k+'_move']),upload(cfg[k+'_cdf'])) for k in ('click','pnr')]
        self.capacity=0
    def draw(self,n,pnr,seed,offset=0):
        if n<1 or max(self.s*self.k,self.m)*n>=2**31-256 or not 0<=offset<=2**32-n or not 0<=seed<2**64:raise ValueError('Invalid record domain')
        if n>self.capacity:
            def alloc(size,dtype):return torch.empty(size,device='cuda',dtype=dtype)
            self.z=alloc(self.s*n*2*self.k,torch.float32);self.f=alloc(self.s*n*2*self.m,torch.float32)
            self.lam=alloc((n,self.m),torch.float64);self.cum=alloc((n,self.m),torch.float64);self.gain=alloc(n,torch.float64)
            self.flags=alloc(n*(1+self.groups),torch.int32);self.x=alloc((n,self.m),torch.uint8);self.capacity=n
        z=self.z[:self.s*n*2*self.k].view(self.s,n,-1);f=self.f[:self.s*n*2*self.m].view(self.s,n,-1)
        stream=torch.cuda.current_stream().cuda_stream
        args=(self.k,self.s,n,seed,offset,len(self.val),self.val.data_ptr(),self.cdf.data_ptr(),0)
        rc=self.normal(*args,z.data_ptr(),self.gain.data_ptr(),stream)
        if rc:raise RuntimeError(rc)
        torch.bmm(z,self.factor,out=f)
        rc=self.rates(self.m,self.s,n,f.data_ptr(),self.gain.data_ptr(),self.lam.data_ptr(),self.error.data_ptr(),stream)
        if rc:raise RuntimeError(rc)
        torch.cumsum(self.lam[:n],1,out=self.cum[:n])
        move,cdf=self.rule[int(pnr)]
        def emit(mode,x):
            params=[w.data_ptr() for w in (self.order,self.bounds,self.region,self.lam,self.cum,move,cdf,self.flags,x,self.error)]
            err=self.read(self.m,n,self.h,self.groups,*self.shape,mode,seed,offset,*params,stream)
            if err:raise RuntimeError(err)
            status=self.error.item()
            if status:self.error.zero_()
            if status==2:raise ArithmeticError('Invalid photon rate or allocation')
            return status
        x=self.x[:n]
        if emit(int(pnr),x):
            x=torch.empty((n,self.m),device='cuda',dtype=torch.int32)
            if emit(2,x):raise ArithmeticError('Wide count readout failed')
        return x.cpu().numpy()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','prepare','draw'));p.add_argument('output',type=Path)
    p.add_argument('--input',type=Path);p.add_argument('--tables',type=Path);p.add_argument('--head',type=int,default=8);p.add_argument('--bulk',type=int,default=4)
    p.add_argument('--pnr',action='store_true');p.add_argument('--shots',type=int,default=256);p.add_argument('--seed',type=int);p.add_argument('--offset',type=int,default=0)
    a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):
        if a.action=='build':
            a.output.mkdir(exist_ok=False,parents=True);build(a.output,'global_readout108_v1.cu');manifest(a.output)
        else:
            if a.input is None:p.error('Specify an optical input')
            with np.load(a.input) as z:t,r=z['transfer'],np.atleast_2d(z['squeezing'])
            if a.action=='prepare':
                cfg,_,_=prepare(t,r,a.head,a.bulk);cfg['input_hash']=fingerprint(t,r)
                with a.output.open('xb') as f:np.savez_compressed(f,**cfg)
            else:
                if a.seed is None or a.tables is None:p.error('Specify tables and an experiment seed')
                with np.load(a.tables) as z:model=Global(t,r,dict(z))
                with a.output.open('xb') as f:np.save(f,model.draw(a.shots,a.pnr,a.seed,a.offset))
