"""CUDA proposals and independent recordwise accept/reject for total-count calibration."""
import argparse
import ctypes as ct
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration103_gpu_v1 import build,P,CU
from iteration103_transport_gpu_v1 import Cuda,Sampler,Correction,OUT as ENGINE,RUN
from iteration100_spatial_v1 import I,U,save,manifest
from iteration103_sample_v1 import DATA

PILOT=DATA/'iteration104_pilot_v1'
FIT=DATA/'iteration104_fit_v1'
MAX=321


class Filter:
    def __init__(self,dll):
        torch.set_float32_matmul_precision('highest')
        model=Sampler(RUN/'draw.dll',RUN/'bridges.npz',False)
        self.g=Cuda(ENGINE/'gpu.dll',model)
        self.tail=Correction(ENGINE/'gpu.dll',model)
        self.lib=ct.CDLL(str(dll))
        self.reduce=self.lib.total104_launch
        self.reduce.argtypes=[I,I,U,CU,P,P,P,I,I,P,P,P]
        self.set_table(np.ones((1,1)))
    def set_table(self,pr):
        if not np.isfinite(pr).all() or np.min(pr)<0 or np.max(pr)>1:
            raise ValueError('Invalid acceptance probabilities')
        self.table=torch.as_tensor(np.ascontiguousarray(pr,dtype=np.float64),device='cuda')
    def propose(self,n,pnr,seed,off):
        if n<1 or n>65536 or off<0 or off+n>2**32 or not 0<=seed<2**64:
            raise ValueError('Invalid proposal count or counter')
        g=self.g
        if n>g.capacity:
            g.noise=torch.empty((n,2*g.groups),dtype=g.dtype,device='cuda')
            g.latent=torch.empty_like(g.noise)
            g.output=torch.empty((n,8*g.groups),dtype=torch.uint16,device='cuda')
            g.counts=torch.empty((n,g.groups),dtype=torch.uint16,device='cuda')
            g.capacity=n
        if not hasattr(self,'totals') or len(self.totals)<n:
            self.totals=torch.empty((n,2),dtype=torch.int32,device='cuda')
            self.accept=torch.empty(n,dtype=torch.int32,device='cuda')
        stream=torch.cuda.current_stream().cuda_stream
        rc=g.normal(g.groups,n,seed,off,0,g.noise.data_ptr(),stream)
        if rc:raise RuntimeError(rc)
        torch.mm(g.noise[:n],g.factor,out=g.latent[:n])
        self.tail.apply(g,n,seed,off,stream)
        rc=g.emit(g.groups,n,int(pnr),seed,off,0,g.latent.data_ptr(),*g.pointers,g.output.data_ptr(),g.counts.data_ptr(),g.error.data_ptr(),stream)
        if rc:raise RuntimeError(rc)
        nr,nc=self.table.shape
        rc=self.reduce(g.groups,n,seed,off,g.output.data_ptr(),g.counts.data_ptr(),self.table.data_ptr(),nc,nr,self.totals.data_ptr(),self.accept.data_ptr(),stream)
        if rc:raise RuntimeError(rc)
    def take(self,n,limit):
        idx=torch.nonzero(self.accept[:n]).flatten()[:limit]
        def selected(v):
            return torch.index_select(v.view(torch.int16),0,idx).cpu().numpy().view(np.uint16)
        return selected(self.g.output),selected(self.g.counts)
    def draw(self,n,pnr,seed,off=0,rate=1.):
        if n<1 or not 0<rate<=1:raise ValueError('Invalid record count or rate')
        parts=[]
        left=n
        while left:
            proposals=min(32768,max(32,int(np.ceil(left/rate))))
            self.propose(proposals,pnr,seed,off)
            off+=proposals
            x,c=self.take(proposals,left)
            if len(x):parts.append((x,c))
            left-=len(x)
        if self.g.error.item():raise RuntimeError('Invalid native optical output')
        if len(parts)==1:return *parts[0],off
        return np.concatenate([p[0] for p in parts]),np.concatenate([p[1] for p in parts]),off


def pilot(out):
    out.mkdir(parents=True,exist_ok=False)
    build(out,'total_filter104_v1.cu')
    model=Filter(out/'gpu.dll')
    x,c=model.g.sample(511,True,104000001,37,model.tail)
    model.propose(511,False,104000001,37)
    t=model.totals[:511].cpu().numpy()
    assert np.array_equal(t[:,0],(x>0).sum(1)) and np.array_equal(t[:,1],c.sum(1))
    assert np.array_equal(model.g.output[:511].cpu().numpy(),x>0)
    n=4*1048576
    hist=np.zeros(MAX*MAX,np.int64)
    tic=perf_counter()
    for off in range(0,n,16384):
        model.propose(16384,False,104001000,off)
        totals=model.totals[:16384].cpu().numpy().clip(0,MAX-1)
        hist+=np.bincount(totals[:,1]*MAX+totals[:,0],minlength=hist.size)
    assert not model.g.error.item() and hist.sum()==n
    np.savez_compressed(out/'joint.npz',hist=hist.reshape(MAX,MAX))
    save(out,'summary.json',dict(records=n,seed=104001000,seconds=perf_counter()-tic,cap=MAX-1))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('pilot','draw'))
    p.add_argument('output',type=Path)
    p.add_argument('--target',choices=('optical','hardware'),default='optical')
    p.add_argument('--seed',type=int)
    p.add_argument('--shots',type=int,default=2048)
    p.add_argument('--pnr',action='store_true')
    a=p.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if a.action=='pilot':pilot(a.output.resolve())
        else:
            if a.seed is None:p.error('--seed is required for draw')
            model=Filter(PILOT/'gpu.dll')
            with np.load(FIT/'tables.npz') as z:
                model.set_table(z[a.target])
                rate=float(z[a.target+'_rate'])
            x,c,off=model.draw(a.shots,a.pnr,a.seed,rate=rate)
            with a.output.open('xb') as f:np.savez_compressed(f,records=x,spatial_photons=c,next_offset=off)
