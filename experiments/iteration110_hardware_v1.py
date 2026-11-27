"""Tune collision multiplicity to a chronological training histogram with all spatial margins protected."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest,H,HIST
from iteration103_gpu_v1 import I,U,CU,P,build as build_gpu
from iteration106_field_v1 import actual_input,DATA
from iteration106_sample_v1 import benchmark
from iteration109_model_v1 import build
from iteration109_sample_v1 import tv
from iteration110_spectrum_v1 import Spectrum
from iteration110_device_v1 import Corrected,config

def cdf(h):
    value=np.rint(np.cumsum(h,dtype=np.longdouble)*2**52/np.sum(h)).astype(np.uint64);value[-1]=2**52
    return torch.as_tensor(value,device='cuda')


class Collision(Corrected):
    def __init__(self,model,tables,weights,sector_dll,dll,pilot,train):
        super().__init__(model,tables,weights,sector_dll)
        self.transport_emit=self.emit
        self.collision_lib=ct.CDLL(str(dll));self.collision=self.collision_lib.collision_emit110
        self.collision.argtypes=[I]*5+[U,CU]+[P]*8
        self.src=[cdf(h) for h in pilot];self.target=cdf(train)
        self.protect=[]
        for ids in (tables['direct'],tables['anchor']):
            mask=np.zeros(self.g,np.int32)
            for i in ids:mask[i//self.fan]|=1<<(i%self.fan)
            self.protect.append(torch.as_tensor(mask,device='cuda'))
        self.emit=self.emit_collision;self.inspect=False;self.audit=None

    def emit_collision(self,g,f,s,n,pn,wide,seed,off,field,eig,lut,cdf_,out,error,stream):
        rc=self.transport_emit(g,f,s,n,pn,wide,seed,off,field,eig,lut,cdf_,out,error,stream)
        if rc:return rc
        if self.inspect and (self.audit is None or len(self.audit)<n):self.audit=torch.empty((n,2),dtype=torch.int32,device='cuda')
        pointer=self.audit.data_ptr() if self.inspect else 0
        return self.collision(n,g,f,pn,wide,seed,off,self.protect[pn].data_ptr(),self.prior.data_ptr(),self.src[pn].data_ptr(),
                              self.target.data_ptr(),out,pointer,error,stream)


def histograms(bank):
    # These spatial observables are reserved for evaluation.
    k=bank.sum(1);active=(bank>0).sum(1);multiple=(bank>=2).sum(1)
    return [np.bincount(k.astype(int),minlength=1153),np.bincount(active,minlength=145),np.bincount(multiple,minlength=145)]


def hardware(raw=None):
    ordered=np.load(DATA/'iteration104_record_order_v1/windows.npz')['histograms'];train=ordered[:32].sum(0)
    plan=json.loads((HIST.parent/'plan.json').read_text());path,digest=next((p,h) for p,h in plan['hashes'].items() if p.endswith('high power (1).bin'))
    path=raw if raw is not None else Path(path)
    assert H(path)==digest
    data=np.memmap(path,dtype=np.uint8,mode='r').reshape(-1,144)[train.sum():]
    ref=[np.zeros(n,np.int64) for n in (1153,145,145)];pop=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
    for i in range(0,len(data),8192):
        for h,v in zip(ref,histograms(pop[data[i:i+8192]])):h+=v
    assert np.array_equal(ref[0],ordered[32:].sum(0))
    return train,ref


def ks(h,g):return float(abs(h.cumsum()/h.sum()-g.cumsum()/g.sum()).max())


def run(out,raw=None):
    out.mkdir(parents=True,exist_ok=False);build_gpu(out,'collision_readout110_v1.cu');torch.backends.cuda.matmul.allow_tf32=False
    train,hw=hardware(raw);t,r=actual_input();m=build(t,r,8,1);zz=np.load(DATA/'iteration110_joint_v1/tables.npz')
    pilot=np.load(DATA/'iteration110_device_v1/histograms.npz');weights=abs(t)**2@(np.sinh(r)**2).sum(0).repeat(2)
    rows=[];arrays={'train':train,**{f'hardware{i}':h for i,h in enumerate(hw)}};sect=DATA/'iteration110_device_v1/gpu.dll'
    for j,(name,model) in enumerate([('base',m),('fitted',Spectrum(m).materialize(zz['coeff']))]):
        cfg=config(zz,name);core=Corrected(model,cfg,weights,sect)
        engine=Collision(model,cfg,weights,sect,out/'gpu.dll',[pilot[f'{name}_{i}_totals'] for i in (0,1)],train)
        for pn in (0,1):
            frozen=cfg['anchor' if pn else 'direct'];seed=110200000+j*2+pn
            engine.inspect=False
            timing=[benchmark(lambda n,s,o:engine.draw(n,pn,s,o),batch,seed+1000,11) for batch in (2048,16384)]
            eight=benchmark(lambda n,s,o:np.ascontiguousarray((engine.draw(n,pn,s,o)[:,:8]>0).astype(np.uint8)),2048,seed+2000,9)
            engine.inspect=True
            check=engine.draw(63,pn,seed+3000)
            assert np.array_equal(check,np.vstack((engine.draw(17,pn,seed+3000),engine.draw(46,pn,seed+3000,17))))
            hist=[[np.zeros_like(v) for v in hw] for _ in range(2)]
            clipped=0;changed=0;shots=262144;batch=2048;first=np.zeros(256,int);pair=np.zeros(169,int)
            for off in range(0,shots,batch):
                before=core.draw(batch,pn,seed,off);after=engine.draw(batch,pn,seed,off)
                assert np.array_equal(before[:,frozen],after[:,frozen])
                ba=before.reshape(batch,144,8);aa=after.reshape(batch,144,8)
                k=(ba>0).sum(2);q=(aa>0).sum(2)
                assert np.array_equal(k>0,q>0)
                if pn:assert np.array_equal(ba.sum(2),aa.sum(2))
                audit=engine.audit[:batch].cpu().numpy();assert np.array_equal(q.sum(1),audit[:,1])
                clipped+=int(np.sum(audit[:,0]!=audit[:,1]));changed+=int(np.sum(k.sum(1)!=q.sum(1)))
                for h,b in zip(hist,[k,q]):
                    for v,a in zip(h,histograms(b)):v+=a
                first+=np.bincount((after[:,:8]>0)@(1<<np.arange(8)),minlength=256)
                if pn:pair+=np.bincount(after[:,[289,861]].clip(0,12)@np.array([1,13]),minlength=169)
            key=f'{name}_{pn}';arrays[key+'_first']=first;arrays[key+'_pair']=pair
            for a in range(2):
                for i,v in enumerate(hist[a]):arrays[key+f'_{a}_{i}']=v
            ref=np.load(DATA/'iteration109_device_v1/observed.npz')['first8_target']
            rows.append(dict(model=name,pnr=bool(pn),shots=shots,timing=timing,eight=eight,clipped=clipped/shots,
                        changed=changed/shots,before_ks=[ks(h,g) for h,g in zip(hist[0],hw)],
                        after_ks=[ks(h,g) for h,g in zip(hist[1],hw)],first8_optical_empirical_tv=tv(first/shots,ref)))
            print(name,pn,rows[-1]['after_ks'],flush=True)
    np.savez_compressed(out/'histograms.npz',**arrays)
    save(out,'summary.json',dict(rows=rows,hardware_train=int(train.sum()),hardware_holdout=int(hw[0].sum()),
                               hardware_split_ks=ks(train,hw[0]),observables=['total clicks','active spatial ports','spatial ports with multiple clicks']));manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--raw',type=Path);args=ap.parse_args()
    with threadpool_limits(limits=2,user_api='blas'):run(args.output.resolve(),args.raw)
