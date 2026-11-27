"""Fit residual spatial windows on capture 1 and transfer frozen models to captures 10 and 11."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration111_windows_v1 import (DATA,Profile,Windows,Stats,compare,config,actual_input,Spectrum,build,benchmark)
from iteration110_hardware_v1 import cdf
from iteration100_spatial_v1 import HIST


def groups_from_training(z):
    def corr(v,n):
        p=v/n
        return p-np.outer(p.diagonal(),p.diagonal())
    ref=corr(z['train_joint'],z['train_head'].sum())
    sim=sum(corr(z[f'post{i}_joint'],z[f'post{i}_head'].sum()) for i in (0,1))/2
    score=abs(ref-sim)
    groups=[];remaining=set(range(144))-{0,34,35}
    while remaining:
        if len(remaining)>=2:
            i,j=max(((i,j) for i in sorted(remaining) for j in sorted(remaining) if i<j),key=lambda ij:score[ij])
            chosen=[i,j]
        else:chosen=list(remaining)
        remaining.difference_update(chosen)
        while remaining and len(chosen)<8:
            i=max(sorted(remaining),key=lambda a:score[a,chosen].sum())
            chosen.append(i);remaining.remove(i)
        groups.append(chosen+[-1]*(8-len(chosen)))
    return np.array(groups,np.int32)


def gates(pilot,target):
    pop=np.array([i.bit_count() for i in range(256)])
    gate=np.zeros_like(pilot,dtype=np.uint64);cdf_=np.zeros((len(pilot),9,256),np.uint32)
    for k in range(9):
        use=pop==k
        a=pilot[:,use];b=target[:,use]
        p=a/np.maximum(1,a.sum(1,keepdims=True))
        q=b/np.maximum(1,b.sum(1,keepdims=True))
        delta=np.where(b.sum(1,keepdims=True)>0,q-p,0)
        gate[:,use]=np.rint(2**52*np.maximum(-delta,0)/np.where(p>0,p,1)).astype(np.uint64)
        f=np.maximum(delta,0)
        cdf_[:,k,use]=np.rint(f/np.maximum(f.sum(1,keepdims=True),1e-30)*2**30).astype(np.uint32)
    return gate,cdf_.cumsum(2,dtype=np.uint32)


def rawstats(raw,groups):
    s=Stats(groups)
    pop=np.array([i.bit_count() for i in range(256)],np.uint8)
    reverse=np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1)@(1<<np.arange(8))
    for i in range(0,len(raw),4096):
        a=raw[i:i+4096]
        s.add(a!=0,pop[a].sum(1).astype(int),reverse[a[:,0]])
    return s


def samples(engine,pn,seed,n,groups):
    s=Stats(groups)
    for off in range(0,n,2048):
        a=engine.draw(min(n-off,2048),pn,seed,off).reshape(-1,144,8)>0
        s.add(a.any(2),a.sum((1,2)).astype(int),a[:,0]@(1<<np.arange(8)))
    return s


def engines():
    z=np.load(DATA/'iteration110_joint_v1/tables.npz')
    cap=np.load(DATA/'iteration110_profile_v1/training.npz')
    first=np.load(DATA/'iteration111_transport_v1/holdout.npz')
    prior=np.load(DATA/'iteration110_device_v1/histograms.npz')
    t,r=actual_input();m=Spectrum(build(t,r,8,1)).materialize(z['coeff'])
    weights=abs(t)**2@(np.sinh(r)**2).sum(0).repeat(2)
    args=(m,config(z,'fitted'),weights,DATA/'iteration110_device_v1/gpu.dll',DATA/'iteration110_hardware_v1/gpu.dll',
          [prior[f'fitted_{i}_totals'] for i in (0,1)],cap['train_total'])
    def make(pn):
        e=Windows(*args)
        e.attach(DATA/'iteration110_profile_v1/gpu.dll',cap['profiles'],cap['frequency'].cumsum(dtype=np.uint32),cap['starts'])
        e.add_windows(DATA/'iteration111_transport_v1/gpu.dll',first['groups'],first[f'deficit{pn}'])
        e.move=torch.as_tensor(first[f'gate{pn}'],device='cuda')
        return e
    return make


def attach(e,groups,gates_,deficit):
    gpu=[torch.as_tensor(np.ascontiguousarray(v),device='cuda') for v in (groups,deficit,gates_)]
    baseline=e.emit
    def emit(g,f,s,n,pn,wide,seed,off,field,eig,lut,cdf_,out,error,stream):
        rc=baseline(g,f,s,n,pn,wide,seed,off,field,eig,lut,cdf_,out,error,stream)
        if rc:return rc
        # Separate phase key for the second call to the same CUDA permutation kernel.
        return e.apply_windows(n,len(groups),wide,seed^(1<<62),off,*[a.data_ptr() for a in gpu],out,stream)
    e.emit=emit


def run(out,src):
    out.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False
    old=np.load(DATA/'iteration111_transport_v1/holdout.npz')
    groups=groups_from_training(old);opt=old['optical']
    first=src.parent/'jiuzhang3_iteration81_sample_v1/extracted_3/jiuzhang3/data/high power/high power (1).bin'
    raw=np.memmap(first,dtype=np.uint8,mode='r').reshape(-1,144)
    cut=int(old['train_head'].sum());train=rawstats(raw[:cut],groups)
    assert np.array_equal(train.joint,old['train_joint'])
    make=engines();models=[];arrays=dict(groups=groups,optical=opt)
    for pn in (0,1):
        e=make(pn)
        pilot=samples(e,pn,112710000+pn,262144,groups)
        gg,dd=gates(pilot.windows,train.windows)
        arrays[f'gate{pn}']=gg;arrays[f'deficit{pn}']=dd
        new=make(pn);attach(new,groups,gg,dd);models.append((e,new))
    plan=out/'calibration.npz'
    np.savez_compressed(plan,**arrays)
    # Model fitting finishes before either new capture is opened.
    (out/'sources.json').write_bytes((src/'files.json').read_bytes())
    captures={1:rawstats(raw[cut:],groups)};prefix={};hold={};hashes={}
    for item in json.loads((src/'files.json').read_text()):
        path=src/item['member'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        k=item['file'];hashes[k]=item['sha256']
        raw=np.memmap(path,np.uint8,mode='r').reshape(-1,144)
        prefix[k]=rawstats(raw[:32768],groups);hold[k]=rawstats(raw[32768:],groups)
        s=Stats(groups)
        for f in ('windows','joint','head','total','records'):setattr(s,f,getattr(prefix[k],f)+getattr(hold[k],f))
        captures[k]=s
    assert set(captures)=={1,10,11}
    rows=[];data={}
    def store(name,s):
        for f in ('windows','joint','head','total'):data[name+'_'+f]=getattr(s,f)
    store('train',train)
    for k,s in captures.items():store(f'hardware{k}',s)
    for k,s in hold.items():store(f'hold{k}',s);store(f'prefix{k}',prefix[k])
    for pn,(old_engine,e) in enumerate(models):
        seed=112720000+pn;shots=262144
        before=samples(old_engine,pn,seed,shots,groups)
        after=samples(e,pn,seed,shots,groups)
        assert np.array_equal(before.head,after.head) and np.array_equal(before.total,after.total)
        times=benchmark(lambda n,s,o:e.draw(n,pn,s,o),16384,seed+100,9)
        store(f'before{pn}',before);store(f'after{pn}',after)
        row=dict(pnr=bool(pn),shots=shots,timing=times,blind={},adapted={})
        for k,s in captures.items():
            row['blind'][k]=dict(before=compare(before,s,opt),after=compare(after,s,opt),records=s.records)
        saved=e.target
        for k in (10,11):
            e.target=cdf(prefix[k].total)
            adapted=samples(e,pn,seed+1000*k,shots,groups)
            store(f'adapted{pn}_{k}',adapted)
            row['adapted'][k]=dict(before=compare(after,hold[k],opt),after=compare(adapted,hold[k],opt),
                                  calibration=prefix[k].records,test=hold[k].records)
        e.target=saved;rows.append(row)
        print(pn,[(k,round(r['after']['pair_mean_tv'],6)) for k,r in row['blind'].items()],flush=True)
    np.savez_compressed(out/'observed.npz',**data)
    (out/'hardware.json').write_text(json.dumps(dict(rows=rows,sha256=hashes,training_records=cut,
        calibration_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
        cross_capture=compare(captures[10],captures[11],opt)),indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    ap.add_argument('--captures',type=Path,default=DATA.parent.parent/'external_sources/jiuzhang3_iteration112_independent')
    args=ap.parse_args()
    with threadpool_limits(limits=3):run(args.output.resolve(),args.captures.resolve())
