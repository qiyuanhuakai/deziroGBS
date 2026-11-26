"""GPU tail transport, recordwise coarse preservation, held-out device accuracy and synchronized timing."""
import argparse
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.stats import beta
from threadpoolctl import threadpool_limits

from iteration100_spatial_v1 import HIST,I,U,save,manifest
from iteration103_gpu_v1 import Cuda,build,P,CU
from iteration103_fit_v1 import FIT
from iteration103_sample_v1 import RUN,DATA,Sampler
from iteration103_tail_v1 import TAIL

OUT=DATA/'iteration103_transport_gpu_v1'


class Correction:
    def __init__(self,dll,model):
        self.lib=ct.CDLL(str(dll))
        self.patch=self.lib.tail103_launch
        self.patch.argtypes=[I,I,I,U,CU,I]+[P]*12
        z=np.load(TAIL/'prepared.npz')
        edges=z['edges']
        self.size=len(edges)
        reprs=np.zeros_like(model.thresholds)
        for g in model.order:
            off=model.offset[g]
            t=model.thresholds[off:off+model.caps[g]+1]
            for n in range(2,len(t)):
                lo,hi=t[n-1:n+1]
                if lo==hi:continue
                v=lo+1 if not np.isfinite(hi) else (hi+lo)/2 if np.isfinite(lo) else hi-1
                assert lo<=np.float32(v)<hi
                reprs[off+n]=v
        params=[edges,z['base'],z['keep'],z['starts'],z['cuts'],z['labels'],model.caps,model.offset,model.thresholds,reprs]
        self.device=[torch.as_tensor(p,device='cuda') for p in params]
        self.args=[x.data_ptr() for x in self.device]
    def apply(self,model,n,seed,offset,stream):
        rc=self.patch(self.size,model.groups,n,seed,offset,int(model.wide),*self.args,model.latent.data_ptr(),stream)
        if rc:raise RuntimeError(f'CUDA transport launch returned {rc}')


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    build(out,'tail_gpu103_v1.cu')
    start=perf_counter()
    cpu=Sampler(RUN/'draw.dll',RUN/'bridges.npz',False)
    torch.set_float32_matmul_precision('highest')
    fast=Cuda(out/'gpu.dll',cpu)
    wide=Cuda(out/'gpu.dll',cpu,True)
    tail=Correction(out/'gpu.dll',cpu)
    x,c=fast.sample(2048,False,103070000,correction=tail)
    cold=(perf_counter()-start)*1e6/2048
    y,d=fast.sample(1984,True,103070000,64,tail)
    assert np.array_equal(x[64:],y>0) and np.array_equal(c[64:],d)
    assert np.array_equal(y.reshape(len(y),144,8).sum(2),d)
    # Interleave model, route and precision. Include the full host arrays each time.
    cases=[(wideflag,patch,pnr) for wideflag in (False,True) for patch in (False,True) for pnr in (False,True)]
    ntime=16384
    times=[[] for _ in cases]
    rng=np.random.default_rng(103070001)
    for rep in range(13):
        for k in rng.permutation(len(cases)):
            iswide,patch,pnr=cases[k]
            start=perf_counter()
            (wide if iswide else fast).sample(ntime,pnr,103071000+int(k),ntime*rep,tail if patch else None)
            if rep>=2:times[k].append((perf_counter()-start)*1e6/ntime)
    timing=[dict(wide=w,patch=m,pnr=p,us=s,median_us=float(np.median(s)),quartiles=np.percentile(s,[25,75]).tolist())
            for (w,m,p),s in zip(cases,times)]
    inputs=np.load(FIT/'fit.npz')
    a,b=inputs['a'],inputs['b']
    hw=np.load(HIST)['total_click_histogram']
    ph=np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf']
    edges=np.load(TAIL/'prepared.npz')['edges']
    ea,eb=edges.T
    with np.load(DATA/'iteration103_census_v1/scan.npz') as cs:
        optical=cs['coarse_optical']
        ii={(int(i),int(j)):k for k,(i,j) in enumerate(zip(cs['a'],cs['b']))}
        edge_ref=np.array([optical[ii[tuple(pair)]] for pair in edges])
    n=1048576
    delta=np.zeros(3,np.int64)
    pilot_ports=[np.load(DATA/f'iteration100_projection_v1/panel_{j}.npz')['ports'] for j in range(5)]
    truths=[np.load(DATA/f'iteration100_projection_v1/panel_{j}.npz')['truth'] for j in range(5)]
    for g,h in [(36,107),(34,35),(25,139)]:
        pilot_ports.append(np.r_[8*g+np.arange(8),8*h+np.arange(8)])
        truths.append(np.load(DATA/f'iteration102_projection_v1/pair_{g}_{h}.npz')['truth'])
    rows=[]
    bw=(1<<np.arange(16)).astype(np.uint16)
    for pnr in (False,True):
        train=[np.zeros_like(q,dtype=np.int64) for q in truths]
        hold=[z.copy() for z in train]
        hc=np.zeros(len(hw),np.int64)
        hp=np.zeros(4096,np.int64)
        cross=np.zeros((288,288))
        edgehist=np.zeros((len(edges),49),np.int64)
        changes=np.zeros(144,np.int64)
        for isp,total,seed,hs in [(True,131072,103072000+pnr,train),(False,n,103073000+pnr,hold)]:
            for off in range(0,total,4096):
                z,c=fast.sample(4096,pnr,seed,off,tail)
                clicked=z>0
                for j,ports in enumerate(pilot_ports):
                    hs[j]+=np.bincount(clicked[:,ports]@bw[:len(ports)],minlength=len(hs[j]))
                if isp:continue
                _,base=fast.sample(4096,pnr,seed,off)
                assert np.array_equal(np.minimum(c,2),np.minimum(base,2))
                changes+=(c!=base).sum(0)
                if pnr:
                    x,d=wide.sample(4096,True,seed,off,tail)
                    delta+=np.array([np.any((x>0)!=clicked,axis=1).sum(),np.any(x!=z,axis=1).sum(),np.any(d!=c,axis=1).sum()])
                    assert np.array_equal(z.reshape(len(z),144,8).sum(2),c)
                hc+=np.bincount(clicked.sum(1),minlength=len(hc))
                hp+=np.bincount(c.sum(1).astype(np.int64),minlength=len(hp))
                marks=np.concatenate([c>=1,c>=2],axis=1).astype(float)
                cross+=marks.T@marks
                cc=c.clip(0,6)
                code=cc[:,ea]*7+cc[:,eb]+49*np.arange(len(ea),dtype=np.int32)
                edgehist+=np.bincount(code.ravel(),minlength=edgehist.size).reshape(edgehist.shape)
        events=[]
        for j,(p,q) in enumerate(zip(train,truths)):
            event=p/p.sum()>q
            hits=int(hold[j][event].sum())
            prob=float(q[event].sum())
            low=float(beta.ppf(.0001,hits,n-hits+1))
            events.append(dict(ports=pilot_ports[j].tolist(),hits=hits,optical=prob,error=hits/n-prob,lower99=max(0.,low-prob)))
            np.savez_compressed(out/f'panel_{int(pnr)}_{j}.npz',pilot=train[j],hold=hold[j])
        measured=np.column_stack([cross[144*i+a,144*j+b] for i in range(2) for j in range(2)])/n
        maxgap=float(abs(measured-inputs['target']).max())
        radius=float(np.sqrt(np.log(4*inputs['target'].size/.002)/(2*n)))
        edge_tv=abs(edgehist.reshape(-1,7,7)/n-edge_ref).sum((1,2))/2
        edge_radius=float(np.sqrt((49*np.log(2)+np.log(2*len(edges)/.002))/(2*n)))
        assert float(edge_tv.max())<=edge_radius
        rows.append(dict(pnr=pnr,records=n,seed=seed,hardware_ks=float(abs(hc.cumsum()/n-hw.cumsum()/hw.sum()).max()),
             photon_ks=float(abs(hp.cumsum()/n-np.pad(ph,(0,len(hp)-len(ph))).cumsum()).max()),
             all_survival_events_max=maxgap,sampling_radius99=radius,matched_count7_tv=float(edge_tv.max()),matched_radius99=edge_radius,
             mean_changed_groups=float(changes.sum()/n),heldout=events))
        np.savez_compressed(out/f'hold_{int(pnr)}.npz',survival=measured,edge_counts=edgehist,changes=changes,click=hc,photon=hp)
    # A separate simultaneous family for the three coupled output differences.
    upper=[float(beta.ppf(1-.01/3,k+1,n-k)) for k in delta]
    save(out,'summary.json',dict(timing=timing,batch=ntime,cold_first_batch_us=cold,rows=rows,fp_mismatches=delta.tolist(),
        fp_coupling99=upper,hardware_radius99=float(np.sqrt(np.log(2*5/.002)/(2*n))+np.sqrt(np.log(2*5/.002)/(2*hw.sum()))),
        photon_radius99=float(np.sqrt(np.log(2*5/.002)/(2*n))),device=torch.cuda.get_device_name()))
    manifest(out)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=('benchmark','draw'))
    ap.add_argument('output',type=Path)
    ap.add_argument('--pnr',action='store_true')
    ap.add_argument('--wide',action='store_true')
    ap.add_argument('--shots',type=int,default=2048)
    ap.add_argument('--seed',type=int)
    args=ap.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.action=='benchmark':main(args.output.resolve())
        else:
            if args.seed is None:ap.error('--seed is required for draw')
            torch.set_float32_matmul_precision('highest')
            cpu=Sampler(RUN/'draw.dll',RUN/'bridges.npz',False)
            gpu=Cuda(OUT/'gpu.dll',cpu,args.wide)
            x,c=gpu.sample(args.shots,args.pnr,args.seed,correction=Correction(OUT/'gpu.dll',cpu))
            with args.output.open('xb') as f:np.savez_compressed(f,records=x,spatial_photons=c)
