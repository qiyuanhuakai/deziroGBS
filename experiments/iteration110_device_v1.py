"""Apply an optical total-and-word coupling directly to batched CUDA records."""
import argparse
import ctypes as ct
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from iteration103_gpu_v1 import build as build_gpu,I,U,CU,P
from iteration106_field_v1 import actual_input,DATA
from iteration106_sample_v1 import benchmark,panel_hist
from iteration109_device_v1 import Local
from iteration109_model_v1 import build
from iteration110_spectrum_v1 import Spectrum
from iteration109_sample_v1 import tv


class Corrected(Local):
    def __init__(self,model,tables,weights,dll):
        super().__init__(model,DATA/'iteration109_device_v1/gpu.dll')
        self.repair_lib=ct.CDLL(str(dll));self.repair=self.repair_lib.transport110
        self.repair.argtypes=[I]*6+[U,CU]+[P]*8
        def gpu(a):return torch.as_tensor(np.ascontiguousarray(a),device='cuda')
        weights=np.asarray(weights,float)
        if weights.shape!=(self.fan*self.g,) or np.any(weights<0) or not np.isfinite(weights).all():raise ValueError('Invalid optical allocation weights')
        # All products with supported totals fit uint64, including at the
        # largest supported lookup-table size.
        norm=weights.sum()
        self.prior=gpu(np.rint(weights*(2**30/norm) if norm else weights).astype(np.uint32))
        self.plans=[]
        for key,ids in [('click',tables['direct']),('pnr',tables['anchor'])]:
            ids=np.asarray(ids,np.int32);bits=len(ids);count=len(tables[key+'_move'])
            if bits>20 or not bits or not np.array_equal(np.unique(ids),np.sort(ids)) or ids.min()<0 or ids.max()>=len(weights):raise ValueError('Invalid anchor modes')
            if count>=2**31 or count%(1<<bits) or (key=='click' and count!=(1<<bits)):raise ValueError('Invalid table layout')
            bins=count>>bits;bit=np.full(len(weights),-1,np.int32);bit[ids]=np.arange(bits)
            move=tables[key+'_move'];cdf=tables[key+'_cdf']
            if move.dtype!=np.uint64 or cdf.dtype!=np.uint64 or cdf.shape!=(count,) or cdf[-1]!=2**52 or (cdf[1:]<cdf[:-1]).any() or (move>2**52).any():raise ValueError('Invalid coupling thresholds')
            self.plans.append((bits,bins,gpu(ids),gpu(bit),gpu(move),gpu(cdf)))
        self.original_emit=self.emit;self.emit=self.emit_with_repair

    def emit_with_repair(self,groups,fan,spectra,n,pn,wide,seed,offset,field,eig,lut,cdf,out,error,stream):
        rc=self.original_emit(groups,fan,spectra,n,pn,wide,seed,offset,field,eig,lut,cdf,out,error,stream)
        if rc:return rc
        bits,bins,ids,bit,move,threshold=self.plans[pn]
        return self.repair(n,groups*fan,pn,wide,bits,bins,seed,offset,ids.data_ptr(),bit.data_ptr(),
                           self.prior.data_ptr(),move.data_ptr(),threshold.data_ptr(),out,error,stream)


def config(z,label):
    return dict(anchor=z['anchor'],direct=z['direct'],**{k:z[label+'_'+k] for k in ('click_move','click_cdf','pnr_move','pnr_cdf')})


def run(out,shots):
    out.mkdir(parents=True,exist_ok=False);build_gpu(out,'sector_readout110_v1.cu')
    torch.backends.cuda.matmul.allow_tf32=False
    t,r=actual_input();model=build(t,r,8,1);spec=Spectrum(model)
    z=np.load(DATA/'iteration110_joint_v1/tables.npz');ids=z['anchor'];direct=z['direct']
    weights=abs(t)**2@(np.sinh(r)**2).sum(0).repeat(2)
    refs=[np.load(DATA/f'iteration106_packed_v1/panel_{k}.npz') for k in (0,1,3,5,6)]
    refs=[(v['ports'].tolist(),v['target']) for v in refs]
    first=np.arange(8);fref=np.load(DATA/'iteration109_device_v1/observed.npz')['first8_target']
    refs.append((first.tolist(),fref))
    hw=np.load(DATA/'iteration104_record_order_v1/windows.npz')['histograms'][32:].sum(0)
    pair=np.load(DATA/'iteration107_projection_v1/pnr.npz')['target'];rows=[];arrays={}
    for j,(name,m) in enumerate([('base',model),('fitted',spec.materialize(z['coeff']))]):
        before=Local(m,DATA/'iteration109_device_v1/gpu.dll');after=Corrected(m,config(z,name),weights,out/'gpu.dll')
        for pn in (False,True):
            key=f'{name}_{int(pn)}';seed=110070000+2*j+pn
            timings=[]
            for batch in (2048,16384):
                timings.append(benchmark(lambda n,s,o:after.draw(n,pn,s,o),batch,seed+2000,11))
            equal=benchmark(lambda n,s,o:np.ascontiguousarray((after.draw(n,pn,s,o)[:,:8]>0).astype(np.uint8)),2048,seed+3000,9)
            replay=after.draw(31,pn,seed+1000)
            assert np.array_equal(replay,np.vstack([after.draw(12,pn,seed+1000),after.draw(19,pn,seed+1000,12)]))
            cuts=[np.zeros(1<<len(ports),np.int64) for ports,_ in refs]
            ct=np.zeros(len(t)+1,np.int64);pt=np.zeros(169,np.int64);joint=np.zeros_like(z['target'],np.int64)
            touched=0;over=0;batch=2048
            outside=np.ones(len(t),bool);outside[direct]=False
            for off in range(0,shots,batch):
                n=min(batch,shots-off);x=after.draw(n,pn,seed,off);raw=before.draw(n,pn,seed,off)
                touched+=np.any(x!=raw,axis=1).sum()
                if not pn:assert np.array_equal(x[:,outside],raw[:,outside])
                click=x>0;ct+=np.bincount(click.sum(1),minlength=len(ct))
                for h,(ports,_) in zip(cuts,refs):h+=panel_hist(x,ports)
                if pn:
                    sums=x.sum(1,dtype=np.int64);over+=np.count_nonzero(sums>=len(joint))
                    word=click[:,ids]@(1<<np.arange(len(ids)))
                    np.add.at(joint,(sums%len(joint),word),1)
                    pt+=np.bincount(x[:,[289,861]].clip(0,12)@np.array([1,13]),minlength=169)
            row=dict(model=name,pnr=pn,shots=shots,timing=timings,equal_first8=equal,changed=touched/shots,
                     hardware_total_click_ks=float(abs(ct.cumsum()/shots-hw.cumsum()/hw.sum()).max()),
                     empirical_panel_tv=[tv(h/shots,p) for h,(_,p) in zip(cuts,refs)])
            arrays[key+'_totals']=ct
            for k,h in enumerate(cuts):arrays[key+f'_panel{k}']=h
            if pn:
                arrays[key+'_joint']=joint;arrays[key+'_pair']=pt
                row.update(total_ks=float(abs((joint.sum(1)/shots-z['target'].sum(1)).cumsum()).max()),
                           joint_empirical_tv=tv(joint/shots,z['target']),pair_empirical_tv=tv(pt/shots,pair),outside_grid=int(over))
            rows.append(row);print(name,pn,row['changed'],timings[-1]['us'],flush=True)
    np.savez_compressed(out/'histograms.npz',hardware=hw,**arrays)
    save(out,'summary.json',dict(rows=rows,panels=[ports for ports,_ in refs],device=torch.cuda.get_device_name()));manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--shots',type=int,default=262144)
    args=ap.parse_args()
    with threadpool_limits(limits=2,user_api='blas'):run(args.output.resolve(),args.shots)
