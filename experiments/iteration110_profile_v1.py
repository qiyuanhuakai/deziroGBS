"""Anonymous spatial multiplicity profiles trained on the first half of the hardware file."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest,H,HIST
from iteration103_gpu_v1 import build as build_gpu,I,U,CU,P
from iteration106_field_v1 import actual_input,DATA
from iteration106_sample_v1 import benchmark
from iteration109_model_v1 import build
from iteration109_sample_v1 import tv
from iteration110_device_v1 import config
from iteration110_spectrum_v1 import Spectrum
from iteration110_hardware_v1 import Collision,ks,histograms

PROTECTED=(34,35)
def observables(b):
    return histograms(b)+[np.bincount((b>=3).sum(1),minlength=145),
        np.bincount(b[:,:36].sum(1).astype(int),minlength=289),np.bincount(b[:,-36:].sum(1).astype(int),minlength=289),
        np.bincount((b[:,36]>0).astype(int)+2*(b[:,107]>0),minlength=4)]


def training(out,raw_path=None):
    order=np.load(DATA/'iteration104_record_order_v1/windows.npz')['histograms']
    cut=int(order[:32].sum());h=order[:32].sum(0)
    files=json.loads((HIST.parent/'plan.json').read_text())['hashes'];path,sha=next((p,s) for p,s in files.items() if p.endswith('high power (1).bin'))
    path=raw_path if raw_path is not None else Path(path)
    assert H(path)==sha;raw=np.memmap(path,np.uint8,mode='r').reshape(-1,144)
    pop=np.array([i.bit_count() for i in range(256)],dtype=np.uint8);keep=np.ones(144,bool);keep[list(PROTECTED)]=False
    counts=[]
    for off in range(0,cut,4096):
        v=pop[raw[off:min(cut,off+4096)]][:,keep];counts.append(np.stack([(v==j).sum(1) for j in range(9)],axis=1).astype(np.uint16))
    profiles,freq=np.unique(np.concatenate(counts),axis=0,return_counts=True);key=profiles@np.arange(9);sort=np.argsort(key,kind='stable')
    profiles=profiles[sort];freq=freq[sort];key=key[sort];starts=np.searchsorted(key,np.arange(1154)).astype(np.int32)
    assert np.all(profiles.sum(1)==142) and freq.sum()==cut
    ref=[np.zeros(n,np.int64) for n in (1153,145,145,145,289,289,4)]
    head=np.zeros(256,int);reverse=np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1)@(1<<np.arange(8))
    for off in range(cut,len(raw),4096):
        v=raw[off:off+4096]
        for dest,z in zip(ref,observables(pop[v])):dest+=z
        head+=np.bincount(reverse[v[:,0]],minlength=256)
    assert np.array_equal(ref[0],order[32:].sum(0))
    np.savez_compressed(out/'training.npz',profiles=profiles,frequency=freq,starts=starts,train_total=h,source_sha256=sha)
    return profiles,np.cumsum(freq,dtype=np.uint32),starts,h,ref,head


class Profile(Collision):
    def attach(self,dll,profiles,freq,starts):
        self.prof_lib=ct.CDLL(str(dll));self.profile=self.prof_lib.profile_emit110;self.profile.argtypes=[I]*4+[U,CU]+[P]*8
        mask=np.zeros(self.g,np.int32);mask[list(PROTECTED)]=1
        self.bank=[torch.as_tensor(np.ascontiguousarray(a),device='cuda') for a in (mask,starts,freq,profiles)]
        self.previous=self.emit;self.emit=self.emit_profile;self.check=None
    def emit_profile(self,g,f,s,n,pn,wide,seed,off,field,eig,lut,cdf,out,error,stream):
        rc=self.previous(g,f,s,n,pn,wide,seed,off,field,eig,lut,cdf,out,error,stream)
        if rc:return rc
        if self.inspect and (self.check is None or len(self.check)<n):self.check=torch.empty(n,dtype=torch.int32,device='cuda')
        return self.profile(n,g,f,wide,seed,off,*[a.data_ptr() for a in self.bank],self.prior.data_ptr(),out,self.check.data_ptr() if self.inspect else 0,stream)


def run(out,raw_path=None):
    out.mkdir(parents=True,exist_ok=False);build_gpu(out,'spatial_profile110_v1.cu')
    torch.backends.cuda.matmul.allow_tf32=False;profiles,freq,starts,h,ref,head=training(out,raw_path)
    t,r=actual_input();z=np.load(DATA/'iteration110_joint_v1/tables.npz');m=Spectrum(build(t,r,8,1)).materialize(z['coeff'])
    prior=np.load(DATA/'iteration110_device_v1/histograms.npz');weights=abs(t)**2@(np.sinh(r)**2).sum(0).repeat(2)
    args=(m,config(z,'fitted'),weights,DATA/'iteration110_device_v1/gpu.dll',DATA/'iteration110_hardware_v1/gpu.dll',
          [prior[f'fitted_{i}_totals'] for i in (0,1)],h)
    orig=Collision(*args);engine=Profile(*args);engine.attach(out/'gpu.dll',profiles,freq,starts)
    rows=[];arrays=dict(head_hardware=head,**{f'hardware_{i}':v for i,v in enumerate(ref)})
    for pn in (0,1):
        seed=110240000+pn;engine.inspect=False
        times=[benchmark(lambda n,s,o:engine.draw(n,pn,s,o),n,seed+1000,11) for n in (2048,16384)]
        eight=benchmark(lambda n,s,o:np.ascontiguousarray((engine.draw(n,pn,s,o)[:,:8]>0).astype(np.uint8)),2048,seed+2000,9)
        engine.inspect=True
        a=engine.draw(48,pn,seed+3000);assert np.array_equal(a,np.vstack((engine.draw(9,pn,seed+3000),engine.draw(39,pn,seed+3000,9))))
        hs=[[np.zeros_like(v) for v in ref] for _ in range(2)];first=np.zeros(256,int);miss=0;moved=0;shots=262144
        pairs=np.zeros(169,int);joint=np.zeros_like(z['target'],dtype=np.int64)
        for off in range(0,shots,2048):
            a=orig.draw(2048,pn,seed,off);b=engine.draw(2048,pn,seed,off)
            aa=a.reshape(-1,144,8);bb=b.reshape(-1,144,8)
            assert np.array_equal(aa[:,PROTECTED],bb[:,PROTECTED])
            for k in range(1,int(max(a.max(),b.max()))+1):assert np.array_equal((a==k).sum(1),(b==k).sum(1))
            miss+=int(np.sum(engine.check[:len(b)].cpu().numpy()==0));moved+=int(np.sum((a>0)&(b==0)))
            first+=np.bincount((b[:,:8]>0)@(1<<np.arange(8)),minlength=256)
            for hset,xx in zip(hs,(aa,bb)):
                for v,g in zip(hset,observables((xx>0).sum(2))):v+=g
            if pn:
                pairs+=np.bincount(b[:,[289,861]].clip(0,12)@np.array([1,13]),minlength=169)
                word=(b[:,z['anchor']]>0)@(1<<np.arange(len(z['anchor'])))
                np.add.at(joint,(b.sum(1,dtype=np.int64)%len(joint),word),1)
        opt=np.load(DATA/'iteration109_device_v1/observed.npz')['first8_target']
        info=dict(pnr=bool(pn),shots=shots,timing=times,equal8=eight,missing_profile_records=miss,relocated_packets_per_record=moved/shots,
                  before_ks=[ks(v,g) for v,g in zip(hs[0][:-1],ref[:-1])],after_ks=[ks(v,g) for v,g in zip(hs[1][:-1],ref[:-1])],
                  cross_pair_hardware_tv=tv(hs[1][-1]/shots,ref[-1]/ref[-1].sum()),first8_hardware_tv=tv(first/shots,head/head.sum()),
                  first8_optical_tv=tv(first/shots,opt))
        arrays[f'first{pn}']=first
        for side in (0,1):
            for j,counts in enumerate(hs[side]):arrays[f'route{pn}_{side}_{j}']=counts
        if pn:
            info['pnr_total_ks']=ks(joint.sum(1),z['target'].sum(1))
            arrays['pnr_joint']=joint;arrays['pnr_pair']=pairs
        rows.append(info);print(pn,info['after_ks'],flush=True)
    np.savez_compressed(out/'histograms.npz',**arrays)
    save(out,'summary.json',dict(rows=rows,unique_profiles=len(profiles),observables=['total clicks','active spatial ports','multiple-click ports','triple-or-more ports','clicks first 36 ports','clicks last 36 ports'],
        protected_spatial_ports=PROTECTED,hardware_train=int(h.sum()),hardware_holdout=int(ref[0].sum())));manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--raw',type=Path);args=ap.parse_args()
    with threadpool_limits(limits=2,user_api='blas'):run(args.output.resolve(),args.raw)
