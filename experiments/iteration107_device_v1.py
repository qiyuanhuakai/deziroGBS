"""GPU click-word and pair-count transport. Both routes preserve full detector records."""
import argparse
import ctypes as ct
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration103_gpu_v1 import I,U,CU,P,build
from iteration106_words_v1 import prepare
from iteration106_sample_v1 import DATA
from iteration107_tables_v1 import OUT as TABLES,LIMIT

OUT=DATA/'iteration107_device_v1'


class Local:
    def __init__(self,dll=OUT/'gpu.dll',tables=TABLES/'tables.npz',label='optical',refinement=None):
        self.base=prepare(label);self.lib=ct.CDLL(str(dll))
        self.fix=self.lib.local107_launch
        self.fix.argtypes=[I,I,I,I,U,CU,P,P,P,P,P,P,P]
        with np.load(tables) as z:
            self.pairs=torch.as_tensor(z['pairs'],device='cuda')
            w=z['weights'].cumsum(1);w[:,-1]=1
            self.weights=torch.as_tensor(w,device='cuda')
            self.tables=[(torch.as_tensor(z[f'{k}_move'],device='cuda'),torch.as_tensor(z[f'{k}_cdf'],device='cuda')) for k in ('click','pnr')]
        self.refinement=refinement
        self.capacity=0
    def draw(self,n,pnr,seed,offset=0,packed=False,paired=False):
        b=self.base
        if n<1 or n*max(b.s*b.k,b.m)>=2**31 or not 0<=seed<2**64 or not 0<=offset<=2**32-n or (pnr and packed):raise ValueError('Invalid record request')
        if n>self.capacity:
            self.z=torch.empty(b.s*n*2*b.k,device='cuda');self.f=torch.empty(b.s*n*2*b.groups,device='cuda')
            self.scale=torch.empty(n,dtype=torch.float64,device='cuda')
            self.raw=torch.empty(n*b.m,dtype=torch.uint8,device='cuda');self.x=torch.empty_like(self.raw);self.capacity=n
        z=self.z[:b.s*n*2*b.k].view(b.s,n,2*b.k);f=self.f[:b.s*n*2*b.groups].view(b.s,n,2*b.groups)
        stream=torch.cuda.current_stream().cuda_stream
        rc=b.normal(b.k,b.s,n,seed,offset,len(b.g),b.g.data_ptr(),b.cdf.data_ptr(),z.data_ptr(),self.scale.data_ptr(),stream)
        if rc:raise RuntimeError(rc)
        torch.bmm(z,b.factor,out=f)
        raw=self.raw[:n*(b.m if pnr else b.groups)].view(n,-1)
        def emit(mode,buffer):
            rc=b.emit(b.groups,b.s,n,mode,seed,offset,f.data_ptr(),b.weight.data_ptr(),self.scale.data_ptr(),buffer.data_ptr(),b.error.data_ptr(),stream)
            if rc:raise RuntimeError(rc)
        emit(int(pnr),raw)
        mode=2 if pnr else int(packed)
        if pnr:
            status=b.error.item()
            if status:
                b.error.zero_()
                if status==2:raise ArithmeticError('Optical rate out of range')
                raw=torch.empty((n,b.m),dtype=torch.int32,device='cuda');emit(2,raw);mode=3
        prior=raw.cpu().numpy() if paired else None
        if paired and not pnr:prior=np.unpackbits(prior,axis=1,bitorder='little')
        x=raw if pnr else self.x[:n*(b.groups if packed else b.m)].view(n,-1)
        move,end=self.tables[int(pnr)]
        rc=self.fix(len(self.pairs),n,LIMIT,mode,seed,offset,self.pairs.data_ptr(),self.weights.data_ptr(),
                    move.data_ptr(),end.data_ptr(),raw.data_ptr(),x.data_ptr(),stream)
        if rc:raise RuntimeError(rc)
        if pnr and self.refinement is not None:
            self.refinement(x,self.pairs,seed,offset,stream)
        result=x.cpu().numpy()
        if b.error.item():raise ArithmeticError('Invalid optical record')
        return (result,prior) if paired else result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','draw'));p.add_argument('output',type=Path)
    p.add_argument('--shots',type=int,default=2048);p.add_argument('--seed',type=int);p.add_argument('--offset',type=int,default=0);p.add_argument('--pnr',action='store_true');p.add_argument('--packed',action='store_true')
    a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):
        if a.action=='build':
            from iteration100_spatial_v1 import manifest
            a.output.mkdir(exist_ok=False,parents=True);build(a.output,'local_transport107_v1.cu');manifest(a.output)
        else:
            if a.seed is None:p.error('Specify an experiment seed')
            with a.output.open('xb') as f:np.save(f,Local().draw(a.shots,a.pnr,a.seed,a.offset,a.packed))
