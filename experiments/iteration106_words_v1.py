"""Dense host outputs from eight-detector click masks and lossless narrow-count PNR readout."""
import argparse
import ctypes as ct
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import I,U,save,manifest
from iteration103_gpu_v1 import CU,P,build
from iteration106_field_v1 import actual_input,factors,DATA
from iteration106_packed_v1 import compress,OUT as PACK
from iteration106_sample_v1 import read,panel_hist

OUT=DATA/'iteration106_words_v1'


class Words:
    def __init__(self,modes,weight,r,profile,dll=OUT/'gpu.dll'):
        self.s,self.k=r.shape;self.groups=len(modes);self.m=self.groups*8
        if weight.shape!=(self.m,1) or self.groups>65536 or self.s*self.k>65536:raise ValueError('Invalid eight-detector readout')
        self.factor=torch.as_tensor(factors(modes,r),dtype=torch.float32,device='cuda')
        self.weight=torch.as_tensor(abs(weight[:,0])**2,dtype=torch.float64,device='cuda')
        self.g=torch.as_tensor(profile['values'],dtype=torch.float64,device='cuda')
        self.cdf=torch.as_tensor(np.cumsum(profile['weights']),dtype=torch.float64,device='cuda')
        self.lib=ct.CDLL(str(dll));self.normal=self.lib.noise_word106_launch
        self.normal.argtypes=[I,I,I,U,CU,I,P,P,P,P,P]
        self.emit=self.lib.word106_launch;self.emit.argtypes=[I,I,I,I,U,CU,P,P,P,P,P,P]
        self.error=torch.zeros(1,dtype=torch.int32,device='cuda');self.capacity=0
    def draw(self,n,pnr,seed,offset=0,packed=False):
        if n<1 or n*max(self.s*self.k,self.m)>=2**31 or not 0<=seed<2**64 or not 0<=offset<=2**32-n:raise ValueError('Invalid draw')
        if packed and pnr:raise ValueError('Packed bits apply to the click output')
        if n>self.capacity:
            self.z=torch.empty(self.s*n*2*self.k,device='cuda');self.f=torch.empty(self.s*n*2*self.groups,device='cuda')
            self.scale=torch.empty(n,dtype=torch.float64,device='cuda')
            self.x=torch.empty(n*self.m,dtype=torch.uint8,device='cuda');self.capacity=n
        z=self.z[:self.s*n*2*self.k].view(self.s,n,2*self.k);f=self.f[:self.s*n*2*self.groups].view(self.s,n,2*self.groups)
        stream=torch.cuda.current_stream().cuda_stream
        rc=self.normal(self.k,self.s,n,seed,offset,len(self.g),self.g.data_ptr(),self.cdf.data_ptr(),z.data_ptr(),self.scale.data_ptr(),stream)
        if rc:raise RuntimeError(rc)
        torch.bmm(z,self.factor,out=f)
        def launch(mode,buffer):
            rc=self.emit(self.groups,self.s,n,mode,seed,offset,f.data_ptr(),self.weight.data_ptr(),self.scale.data_ptr(),buffer.data_ptr(),self.error.data_ptr(),stream)
            if rc:raise RuntimeError(rc)
        cols=self.groups if packed else self.m
        x=self.x[:n*cols].view(n,cols);launch(1 if pnr else (0 if packed else 3),x)
        value=x.cpu().numpy()
        status=self.error.item()
        if status:
            self.error.zero_()
            if status==2:raise ArithmeticError('Rate or random address out of range')
            x=torch.empty((n,self.m),dtype=torch.int32,device='cuda');launch(2,x)
            value=x.cpu().numpy().view(np.uint32)
            if self.error.item():raise ArithmeticError('Wide PNR readout failed')
        return value


def prepare(label,dll=OUT/'gpu.dll'):
    t,r=actual_input();m,w,_=compress(t,(np.sinh(r)**2).sum(0),1)
    stage=read(PACK/'summary.json')
    p=stage['profiles']['1'] if label=='optical' else read(DATA/'iteration106_device_v1/summary.json')['hardware_profile']
    return Words(m,w,np.array(stage['squeezing']),p,dll)


def run(out):
    out.mkdir(parents=True,exist_ok=False);build(out,'word_field106_v1.cu')
    models=[prepare(l,out/'gpu.dll') for l in ('optical','hardware')]
    # Interleave configurations in one shuffled schedule at each batch size.
    rows=[]
    rng=np.random.default_rng(106200000)
    for n in (4,64,2048,16384):
        configs=[(i,route,packed) for i in range(2) for route,packed in [(False,False),(False,True),(True,False)]]
        timing={key:[] for key in configs}
        for rep in range(18):
            for jj in rng.permutation(len(configs)):
                i,p,b=configs[jj];model=models[i]
                start=perf_counter();model.draw(n,p,106202000+int(jj),rep*n,b);us=(perf_counter()-start)*1e6/n
                if rep>=3:timing[i,p,b].append(us)
        for (i,p,b),times in timing.items():
            rows.append(dict(label=('optical','hardware')[i],pnr=p,packed=b,batch=n,us=float(np.median(times)),quartiles=np.quantile(times,[.25,.75]).tolist(),times=times))
    # Explicit overflow test. The wider exact route must reproduce the same
    # waveforms and zero bits when the record range changes.
    bright=Words(np.array([[.7,.7]],complex),np.ones((8,1),complex)/np.sqrt(8),np.array([[.9]]),dict(values=[8192.],weights=[1.]),out/'gpu.dll')
    v=bright.draw(2048,True,106209000);assert v.dtype==np.uint32 and v.max()>255
    assert np.array_equal(v[20:]>0,bright.draw(2028,False,106209000,20))
    assert np.array_equal(v[20:],bright.draw(2028,True,106209000,20))
    hw=np.load(DATA/'iteration104_record_order_v1/windows.npz')['histograms'][32:].sum(0)
    n=1048576;runs=[]
    for i,label in enumerate(('optical','hardware')):
        baseline=np.load(DATA/f'iteration106_device_v1/{label}_1.npz')
        for p in (False,True):
            h=np.zeros(1153,np.int64);ph=np.zeros(1024,np.int64);pair=np.zeros(169,np.int64)
            for off in range(0,n,4096):
                x=models[i].draw(4096,p,106210000+2*i+p,off)
                h+=np.bincount(np.count_nonzero(x,axis=1),minlength=len(h))
                if p:
                    ph+=np.bincount(x.sum(1).astype(np.int64),minlength=len(ph))
                    pair+=np.bincount(np.minimum(x[:,[289,861]],12)@np.array([1,13]),minlength=169)
            entry=dict(label=label,pnr=p,records=n,heldout_hw_ks=float(abs(h.cumsum()/n-hw.cumsum()/hw.sum()).max()))
            if p:
                entry['model_photon_ks']=float(abs(ph.cumsum()/n-baseline['model_total'].cumsum()).max())
                entry['model_pair_tv']=float(abs(pair/n-baseline['model_pair']).sum()/2)
                assert entry['model_photon_ks']<.004 and entry['model_pair_tv']<.003
            assert abs(h.sum()-n)==0
            x=models[i].draw(4096,True,106211000+i)
            masks=models[i].draw(4080,False,106211000+i,16,True)
            assert np.array_equal(x[16:]>0,np.unpackbits(masks,axis=1,bitorder='little'))
            np.savez_compressed(out/f'{label}_{int(p)}.npz',click=h,photon=ph,pair=pair)
            runs.append(entry);print(entry,flush=True)
    save(out,'summary.json',dict(rows=rows,runs=runs,high_rate_max=int(v.max()),high_rate_examples=len(v),
        conditional_threshold_uniform_full_tv_bound=1152*2**-33,hardware_holdout_records=int(hw.sum())))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','draw'));p.add_argument('output',type=Path)
    p.add_argument('--label',choices=('optical','hardware'),default='optical');p.add_argument('--shots',type=int,default=2048);p.add_argument('--seed',type=int);p.add_argument('--pnr',action='store_true');p.add_argument('--packed',action='store_true')
    a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):
        if a.action=='build':run(a.output.resolve())
        else:
            if a.seed is None:p.error('Provide an experiment seed')
            with a.output.open('xb') as f:np.save(f,prepare(a.label).draw(a.shots,a.pnr,a.seed,packed=a.packed))
