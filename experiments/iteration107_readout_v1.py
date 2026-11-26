"""GPU Fock allocation correction and independent full-record timing and validation."""
import argparse
import ctypes as ct
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from iteration103_gpu_v1 import build,I,U,CU,P
from iteration100_spatial_v1 import manifest,save
from iteration106_sample_v1 import DATA
from iteration107_device_v1 import Local
from iteration107_patterns_v1 import OUT as PATTERNS
from iteration107_holdout_v1 import code

OUT=DATA/'iteration107_readout_v1'


class Refinement:
    def __init__(self,dll=OUT/'gpu.dll'):
        self.lib=ct.CDLL(str(dll))
        self.fn=self.lib.refine107_launch
        self.fn.argtypes=[I,I,I,I,U,CU]+[P]*9
        with np.load(PATTERNS/'tables.npz') as z:
            self.states=len(z['codes'])
            self.params=[torch.as_tensor(z[k],device='cuda') for k in ('offset','sizes','choose','codes','move','cdf')]

    def __call__(self,x,pairs,seed,offset,stream):
        rc=self.fn(len(pairs),len(x),self.states,int(x.element_size()==4),seed,offset,pairs.data_ptr(),
                   *(param.data_ptr() for param in self.params),x.data_ptr(),stream)
        if rc:raise RuntimeError(rc)


def check_layout(refinement):
    # Force exactly one source state per sector to move into the last state.
    # Repeating for every bit of the local index distinguishes all 20,349 ranks.
    with np.load(PATTERNS/'tables.npz') as z:
        codes,offset,sizes=z['codes'],z['offset'],z['sizes']
    bins=((codes[:,None]>>(4*np.arange(16,dtype=np.uint64)))&15).astype(np.int32)
    bins=np.vstack([bins,np.full((1,16),256,np.int32)])
    ends=np.empty(len(codes),int);indices=np.empty_like(ends);cdf=np.zeros(len(codes),np.uint64)
    for a,b in zip(*np.nonzero(offset>=0)):
        lo=offset[a,b];size=sizes[a]*sizes[b];hi=lo+size
        ends[lo:hi]=hi-1;indices[lo:hi]=np.arange(size);cdf[hi-1]=1<<52
    pairs=torch.tensor([[0,1]],dtype=torch.int32,device='cuda')
    old=refinement.params
    try:
        for bit in range(int(indices.max()).bit_length()):
            hit=((indices>>bit)&1).astype(bool)
            refinement.params=[*old[:4],torch.as_tensor(hit.astype(np.float32),device='cuda'),torch.as_tensor(cdf,device='cuda')]
            for dtype in (np.uint8,np.int32):
                original=bins.astype(dtype)
                if dtype==np.uint8:original[-1]=255
                x=torch.as_tensor(original,device='cuda')
                refinement(x,pairs,107031000,0,torch.cuda.current_stream().cuda_stream)
                expected=original.copy();ids=np.flatnonzero(hit);expected[ids]=original[ends[ids]]
                assert np.array_equal(x.cpu().numpy(),expected)
    finally:refinement.params=old


def run(out):
    out.mkdir(exist_ok=False,parents=True);build(out,'photon_readout107_v1.cu')
    ref=Refinement(out/'gpu.dll');check_layout(ref)
    hw=np.load(DATA/'iteration104_record_order_v1/windows.npz')['histograms'][32:].sum(0)
    full=np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf']
    with np.load(DATA/'iteration107_projection_v1/pnr.npz') as z:
        print('PNR reference keys',z.files,flush=True)
    pairids=np.arange(34*8,36*8)
    rows=[];arrays={};timings=[]
    rng=np.random.default_rng(107032000)
    for il,label in enumerate(('optical','hardware')):
        model=Local(label=label,refinement=ref)
        for n in (4,2048,16384):
            times=[[],[],[]]
            for j in range(12):
                for route in rng.permutation(3):
                    model.refinement=ref if route==2 else None
                    start=perf_counter();model.draw(n,route>0,107033000+il*10+int(route),j*n)
                    if j>=3:times[route].append((perf_counter()-start)*1e6/n)
            for k,ts in enumerate(times):timings.append(dict(profile=label,batch=n,route=['click','pnr_totals','pnr_resolved'][k],us=float(np.median(ts)),repeats=ts))
        model.refinement=ref
        for pnr in (False,True):
            n=1048576;h=np.zeros(1153,np.int64);ph=np.zeros(1153,np.int64);hist=np.zeros(65536,np.int64);pn=np.zeros(169,np.int64);small=np.zeros(169,np.int64)
            changed=0
            for off in range(0,n,4096):
                seed=107034000+2*il+int(pnr)
                x=model.draw(4096,pnr,seed,off)
                if pnr and off<32768:
                    model.refinement=None;base=model.draw(4096,True,seed,off);model.refinement=ref
                    assert np.array_equal(x.reshape(4096,144,8).sum(2),base.reshape(4096,144,8).sum(2))
                    changed+=np.any(x!=base,axis=1).sum()
                h+=np.bincount(np.count_nonzero(x,axis=1),minlength=len(h))
                hist+=np.bincount(code(x,pairids),minlength=len(hist))
                if pnr:
                    ph+=np.bincount(x.sum(1).astype(int),minlength=len(ph))
                    c=np.bincount(np.minimum(x[:,[289,861]],12)@np.array([1,13]),minlength=169)
                    pn+=c
                    if off<32768:small+=c
            tag=label+('_pnr' if pnr else '_click')
            arrays.update({tag+'_'+k:v for k,v in dict(click_total=h,pattern=hist,photon_total=ph,pair=pn,pair32768=small).items()})
            common=min(len(full),len(ph))
            row=dict(profile=label,pnr=pnr,records=n,hardware_ks=float(abs(h.cumsum()/n-hw.cumsum()/hw.sum()).max()))
            if pnr:row.update(optical_total_ks=float(abs(ph.cumsum()[:common]/n-full.cumsum()[:common]).max()),changed_records_of_32768=int(changed))
            rows.append(row);print(row,flush=True)
        x=model.draw(211,True,107035000+il,9)
        assert np.array_equal(x[31:],model.draw(180,True,107035000+il,40))
    np.savez_compressed(out/'holdout.npz',**arrays)
    save(out,'summary.json',dict(runs=rows,timing=timings,layout_cases=len(codes_for_test()),hardware_holdout_records=int(hw.sum())))
    manifest(out)


def codes_for_test():
    with np.load(PATTERNS/'tables.npz') as z:return z['codes']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('validate','draw'));p.add_argument('output',type=Path)
    p.add_argument('--seed',type=int);p.add_argument('--offset',type=int,default=0);p.add_argument('--shots',type=int,default=2048)
    p.add_argument('--profile',choices=('optical','hardware'),default='optical');p.add_argument('--pnr',action='store_true')
    a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):
        if a.action=='validate':run(a.output.resolve())
        else:
            if a.seed is None:p.error('Specify an experiment seed')
            model=Local(label=a.profile,refinement=Refinement() if a.pnr else None)
            with a.output.open('xb') as f:np.save(f,model.draw(a.shots,a.pnr,a.seed,a.offset))
