"""Equal observable, equal batch and equal record count against the archived upstream GPU jobs."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from iteration104_validate_v1 import interval
from iteration106_sample_v1 import DATA,read,benchmark
from iteration107_readout_v1 import Local,Refinement


def run(out):
    out.mkdir(exist_ok=False,parents=True)
    d=DATA/'iteration104_frugal_device_v1';verify=read(DATA/'iteration104_device_release_v1/validation.json')
    model=Local(refinement=Refinement());outrows=[];bins={}
    for panel,ports in enumerate((np.arange(8),np.arange(288,296),np.arange(144))):
        batch=2048 if panel<2 else 64
        for pnr in (False,True):
            def draw(n,seed,offset):
                x=model.draw(n,pnr,seed,offset)
                return np.ascontiguousarray((x[:,ports]>0).astype(np.uint8))
            row=dict(panel=panel,pnr=pnr,batch=batch,us=benchmark(draw,batch,107061000+2*panel+pnr,12)['us'])
            if panel<2:
                with np.load(d/f'panel_{panel}_frugal32_2.npz') as z:q=z['truth']
                hh=[]
                for phase,n in enumerate((32768,262144)):
                    hist=np.zeros(256,np.int64)
                    for off in range(0,n,2048):
                        x=draw(2048,107062000+4*panel+2*pnr+phase,off)
                        hist+=np.bincount(x@(1<<np.arange(8)),minlength=256)
                    hh.append(hist)
                row.update(interval(q,hh[1],hh[0],int(hh[1].sum()),.0001));row['shots']=262144
                bins[f'p{panel}_pnr{int(pnr)}']=hh[1]
            outrows.append(row)
    with np.load(DATA/'iteration107_resolved_law_v1/pnr.npz') as z:target=z['target']
    with np.load(DATA/'iteration107_readout_v1/holdout.npz') as z:h=z['optical_pnr_pair32768']
    pe=read(DATA/'iteration106_release_v1/validation.json')['pe']
    np.savez_compressed(out/'histograms.npz',**bins)
    save(out,'summary.json',dict(rows=outrows,upstream_gpu=[x for x in verify['rows'] if x['method'].startswith('frugal')],
         pe_pair=dict(ports=[289,861],shots=int(h.sum()),pnr_empirical_tv=float(abs(h/h.sum()-target).sum()/2),
                      pe_empirical_tv=pe['pe_empirical_tv'],pe_us=pe['pe_us']),upstream_commit_evidence='iteration104_device_release_v1/validation.json'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):run(p.parse_args().output.resolve())
