"""GPU Gaussian displacement, displaced single-mode quantum detection, and passive fan-out."""
import argparse
import ctypes as ct
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from iteration103_gpu_v1 import I,U,CU,P,build as build_gpu
from iteration106_field_v1 import DATA,actual_input
from iteration106_sample_v1 import benchmark,panel_hist
from iteration107_fast_v1 import oracle,gaussian_click,ONE
from iteration109_model_v1 import build,panel
from iteration109_sample_v1 import tv

class Local:
    def __init__(self,p,dll):
        self.normal_lib=ct.CDLL(str(DATA/'iteration107_fast_v1/gpu.dll'))
        self.normal=self.normal_lib.normals107_launch;self.normal.argtypes=[I,I,I,U,CU,I,P,P,I,P,P,P]
        self.lib=ct.CDLL(str(dll));self.emit=self.lib.local109;self.emit.argtypes=[I]*6+[U,CU]+[P]*7
        self.s,self.g,_=p['local'].shape;self.fan=p['fan']
        device=lambda a,dtype:torch.as_tensor(np.ascontiguousarray(a),device='cuda',dtype=dtype)
        # Rotate the shared field into each mode's own residual squeezing axes.
        factors=[np.einsum('kgi,gij->kjg',f,rot).reshape(len(f),2*self.g) for f,rot in zip(p['factors'],p['rot'])]
        rank=max(2,max(len(f) for f in factors));rank+=rank%2;self.k=rank//2
        if self.s*self.k>=65536*((65535-221)//8):raise ValueError('Normal stream index overflow')
        self.f=device(np.stack([np.pad(f,((0,rank-len(f)),(0,0))) for f in factors]),torch.float32)
        masks=np.arange(1<<self.fan);bits=(masks[:,None]>>np.arange(self.fan))&1
        eff=bits@p['weights'].T
        lut=[];base=np.zeros_like(eff)
        for spec in p['local']:
            for eigen in spec.T:
                denom=1+eff*eigen
                if np.min(denom)<=0:raise ValueError('Invalid vacuum denominator')
                base-=np.log(denom)/2;lut.append(eff/denom)
        self.e=device(p['local'].transpose(0,2,1),torch.float64)
        self.lut=device(np.stack([base,*lut]),torch.float64)
        self.cdf=device(np.cumsum(p['weights'],axis=1).T,torch.float64)
        self.unit=device([1.],torch.float64);self.error=torch.zeros(1,dtype=torch.int32,device='cuda');self.capacity=0
    def draw(self,n,pnr,seed,offset=0):
        m=self.g*self.fan
        if not n>0 or max(self.s*self.k*2,self.s*self.g*2,m)*n>=2**31-256 or not 0<=offset<=2**32-n or not 0<=seed<2**64:raise ValueError('Invalid batch')
        if n>self.capacity:
            self.z=torch.empty(self.s*n*2*self.k,device='cuda');self.field=torch.empty(self.s*n*2*self.g,device='cuda')
            self.gain=torch.empty(n,dtype=torch.float64,device='cuda');self.x=torch.empty((n,m),device='cuda',dtype=torch.uint8);self.capacity=n
        z=self.z[:self.s*n*2*self.k].view(self.s,n,-1);f=self.field[:self.s*n*2*self.g].view(self.s,n,-1)
        st=torch.cuda.current_stream().cuda_stream
        rc=self.normal(self.k,self.s,n,seed,offset,1,self.unit.data_ptr(),self.unit.data_ptr(),0,z.data_ptr(),self.gain.data_ptr(),st)
        if rc:raise RuntimeError(rc)
        torch.bmm(z,self.f,out=f)
        out=self.x[:n]
        for wide in (0,1):
            if wide:out=torch.empty_like(out,dtype=torch.int32)
            rc=self.emit(self.g,self.fan,self.s,n,int(pnr),wide,seed,offset,f.data_ptr(),self.e.data_ptr(),
                         self.lut.data_ptr(),self.cdf.data_ptr(),out.data_ptr(),self.error.data_ptr(),st)
            if rc:raise RuntimeError(rc)
            err=self.error.item()
            if not err:return out.cpu().numpy()
            self.error.zero_()
            if err!=1 or wide:raise ArithmeticError('Displaced probability recurrence out of range')
        raise ArithmeticError('Output counter overflow')

def synthetic(m):
    # Independent squeezed vacua with loss, coherently displaced by eight
    # shared Gaussian variables. Every mode participates in the dense field.
    rng=np.random.default_rng(109600000+m);s=.35+.38*rng.random(m);angle=rng.uniform(0,2*np.pi,m)
    rot=np.stack((np.cos(angle),-np.sin(angle),np.sin(angle),np.cos(angle)),axis=1).reshape(m,2,2)
    local=.72*np.stack((np.exp(-2*s)-1,np.exp(2*s)-1),axis=1)/2
    f=rng.normal(size=(8,m,2))/np.sqrt(32)
    return dict(factors=[f],rot=rot[None],local=local[None],weights=np.ones((m,1)),fan=1,m=m)

def run(out):
    out.mkdir(parents=True,exist_ok=False);build_gpu(out,'displaced_local109_v1.cu')
    torch.backends.cuda.matmul.allow_tf32=False
    t,r=actual_input();actual=build(t,r,8,1);d=Local(actual,out/'gpu.dll')
    src=DATA/'iteration104_record_order_v1/windows.npz';hw=np.load(src)['histograms'][32:].sum(0);qs=oracle(True)
    cases=[('first8',np.arange(8)),('space36',np.arange(288,296))]
    panels=[];arrays={};rows=[];scaling=[]
    for j,(name,ids) in enumerate(cases):
        ref=np.load(DATA/f'iteration104_frugal_device_v1/panel_{j}_frugal32_2.npz')['truth']
        law=gaussian_click(qs,panel(actual,ids),ONE)
        panels.append(dict(name=name,model_tv=tv(law,ref)));arrays[name+'_target']=ref;arrays[name+'_model']=law
    for pnr in (False,True):
        timing=[]
        for batch in (2048,16384):
            timing.append(benchmark(lambda n,s,o:d.draw(n,pnr,s,o),batch,109340000+pnr,11))
        shots=262144;hs=np.zeros((2,256),np.int64);tot=np.zeros(len(t)+1,np.int64);pair=np.zeros(169,np.int64)
        x0=d.draw(32,pnr,109341000+pnr)
        assert np.array_equal(x0,np.vstack([d.draw(7,pnr,109341000+pnr),d.draw(25,pnr,109341000+pnr,7)]))
        for off in range(0,shots,2048):
            x=d.draw(2048,pnr,109342000+pnr,off)
            for j,(_,ids) in enumerate(cases):hs[j]+=panel_hist(x,ids)
            tot+=np.bincount((x>0).sum(1),minlength=len(tot))
            if pnr:pair+=np.bincount(x[:,[289,861]].clip(0,12)@np.array([1,13]),minlength=169)
        mode='pnr' if pnr else 'click';arrays[mode+'_panels']=hs;arrays[mode+'_total']=tot;arrays[mode+'_pair']=pair
        rows.append(dict(route=mode,shots=shots,timing=timing,empirical_tv=[tv(h/shots,arrays[name+'_target']) for h,(name,_) in zip(hs,cases)],
                         hardware_ks=float(abs(tot.cumsum()/shots-hw.cumsum()/hw.sum()).max())))
    for m in (16,1152,8192,131072):
        p=synthetic(m);model=Local(p,out/'gpu.dll')
        mean=float(sum((f*f).sum() for f in p['factors'])+p['local'].sum()/2)
        batch=min(2048,max(1,4194304//m))
        info=dict(ports=m,mean_photons=mean,batch=batch)
        for route in ('click','pnr'):
            tm=benchmark(lambda n,s,o:model.draw(n,route=='pnr',s,o),batch,109350000+m+(route=='pnr'),9)
            info[route]=tm
        scaling.append(info)
    np.savez_compressed(out/'observed.npz',hardware=hw,**arrays)
    save(out,'summary.json',dict(actual=rows,panels=panels,scaling=scaling,device=torch.cuda.get_device_name()));manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(ap.parse_args().output.resolve())
