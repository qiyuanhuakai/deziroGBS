"""Generate only accepted detector records, with reproducible continuation across output batches."""
import argparse
import ctypes as ct
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration103_gpu_v1 import build,P,CU
from iteration104_sample_v1 import Filter,PILOT,FIT,DATA,I,U,save,manifest

OUT=DATA/'iteration104_packed_v1'


class Packed(Filter):
    def __init__(self,dll):
        super().__init__(PILOT/'gpu.dll')
        self.fastlib=ct.CDLL(str(dll))
        self.decide=self.fastlib.decide104_launch
        self.decide.argtypes=[I,I,U,CU]+[P]*10+[I,I]+[P]*4
        self.expand=self.fastlib.expand104_launch
        self.expand.argtypes=[I,I,I,I,U,CU]+[P]*10
        self.max_rows=0
        self.max_output=0
    def batch(self,n,left,pnr,seed,off):
        g=self.g
        if n>self.max_rows:
            g.noise=torch.empty((n,2*g.groups),dtype=g.dtype,device='cuda')
            g.latent=torch.empty_like(g.noise)
            self.words=torch.empty((n,g.groups),dtype=torch.int32,device='cuda')
            self.pass_mask=torch.empty(n,dtype=torch.int32,device='cuda')
            self.prefix=torch.empty_like(self.pass_mask)
            self.max_rows=n
        limit=min(n,left)
        if limit>self.max_output:
            self.records=torch.empty((limit,8*g.groups),dtype=torch.uint16,device='cuda')
            self.photons=torch.empty((limit,g.groups),dtype=torch.uint16,device='cuda')
            self.max_output=limit
        stream=torch.cuda.current_stream().cuda_stream
        err=g.normal(g.groups,n,seed,off,0,g.noise.data_ptr(),stream)
        if err:raise RuntimeError(err)
        torch.mm(g.noise[:n],g.factor,out=g.latent[:n])
        self.tail.apply(g,n,seed,off,stream)
        caps,starts,levels,leaves,masks,prior,weights,shortcut,direct,alternate=g.pointers
        err=self.decide(g.groups,n,seed,off,g.latent.data_ptr(),caps,starts,levels,leaves,masks,shortcut,direct,alternate,self.table.data_ptr(),*self.table.shape,self.words.data_ptr(),self.pass_mask.data_ptr(),g.error.data_ptr(),stream)
        if err:raise RuntimeError(err)
        torch.cumsum(self.pass_mask[:n],0,dtype=torch.int32,out=self.prefix[:n])
        err=self.expand(g.groups,n,limit,int(pnr),seed,off,self.words.data_ptr(),self.pass_mask.data_ptr(),self.prefix.data_ptr(),leaves,prior,weights,self.records.data_ptr(),self.photons.data_ptr(),g.error.data_ptr(),stream)
        if err:raise RuntimeError(err)
        # Left insertion point for the final requested accepted row. Returning
        # its successor makes splitting a request exactly reproducible.
        end=int(torch.searchsorted(self.prefix[:n],limit).item())
        if end<n:
            accepted=limit;consumed=end+1
        else:
            accepted=int(self.prefix[n-1].item());consumed=n
        return self.records[:accepted].cpu().numpy(),self.photons[:accepted].cpu().numpy(),consumed
    def draw(self,n,pnr,seed,off=0,rate=1.):
        if n<1 or not 0<rate<=1 or off<0 or not 0<=seed<2**64:raise ValueError('Invalid sampling request')
        left=n
        parts=[]
        while left:
            size=min(196608,2**32-off,max(16,int(np.ceil((left+5*np.sqrt(left*(1-rate)))/rate))))
            if size<=0:raise ValueError('Record domain exhausted')
            x,c,used=self.batch(size,left,pnr,seed,off)
            off+=used
            if len(x):parts.append((x,c))
            left-=len(x)
        if self.g.error.item():raise RuntimeError('Invalid packed detector record')
        if len(parts)==1:return *parts[0],off
        return np.concatenate([p[0] for p in parts]),np.concatenate([p[1] for p in parts]),off


def bench(out):
    out.mkdir(parents=True,exist_ok=False)
    build(out,'packed_filter104_v1.cu')
    model=Packed(out/'gpu.dll')
    old=Filter(PILOT/'gpu.dll')
    tables=dict(np.load(FIT/'tables.npz'))
    comparisons=[]
    for i,label in enumerate(('optical','hardware')):
        rate=float(tables[label+'_rate'])
        for alg in (model,old):alg.set_table(tables[label])
        for pnr in (False,True):
            seed=104025000+2*i+pnr
            old.propose(32768,pnr,seed,137)
            mask=old.accept[:32768].cpu().numpy().astype(bool)
            k=int(mask.sum())
            e=137+int(np.flatnonzero(mask)[-1])+1
            x,c=old.take(32768,k)
            y,d,actual=model.draw(k,pnr,seed,137,rate)
            assert actual==e and np.array_equal(x,y) and np.array_equal(c,d)
            half=k//2
            z,q,nxt=model.draw(half,pnr,seed,137,rate)
            zz,qq,end=model.draw(k-half,pnr,seed,nxt,rate)
            assert np.array_equal(y,np.concatenate((z,zz))) and np.array_equal(d,np.concatenate((q,qq))) and end==e
            comparisons.append(dict(label=label,pnr=pnr,proposals=32768,matched_full_records=k,next_counter=end))
    timings=[]
    cases=[(label,pnr,n) for label in ('optical','hardware') for pnr in (False,True) for n in (4,2048,16384)]
    rng=np.random.default_rng(104026000)
    for rep in range(12):
        for k in rng.permutation(len(cases)):
            label,pnr,n=cases[k]
            model.set_table(tables[label])
            torch.cuda.synchronize()
            start=perf_counter()
            x,c,end=model.draw(n,pnr,104027000+100*rep+int(k),rate=float(tables[label+'_rate']))
            elapsed=(perf_counter()-start)*1e6/n
            if rep>=2:timings.append(dict(label=label,pnr=pnr,n=n,us=elapsed,consumed=end))
    rows=[]
    for label,pnr,n in cases:
        times=[r['us'] for r in timings if (r['label'],r['pnr'],r['n'])==(label,pnr,n)]
        rows.append(dict(label=label,pnr=pnr,batch=n,median_us=float(np.median(times)),quartiles=np.percentile(times,[25,75]).tolist(),us=times))
    save(out,'summary.json',dict(timing=rows,record_equivalence=comparisons,device=torch.cuda.get_device_name(),scope='All 1,152 detectors and all 144 group counts returned to host.'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('benchmark','draw'))
    p.add_argument('output',type=Path)
    p.add_argument('--target',choices=('optical','hardware'),default='optical')
    p.add_argument('--seed',type=int)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--shots',type=int,default=2048)
    p.add_argument('--pnr',action='store_true')
    a=p.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if a.action=='benchmark':bench(a.output.resolve())
        else:
            if a.seed is None:p.error('--seed is required')
            model=Packed(OUT/'gpu.dll')
            with np.load(FIT/'tables.npz') as z:
                model.set_table(z[a.target])
                rate=float(z[a.target+'_rate'])
            x,c,end=model.draw(a.shots,a.pnr,a.seed,a.offset,rate)
            with a.output.open('xb') as f:np.savez_compressed(f,records=x,spatial_photons=c,next_offset=end)
