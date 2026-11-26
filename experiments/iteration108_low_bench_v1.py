"""Independent fixed-width and growing-source tests of the singleton and two-photon readout."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from iteration106_width_v1 import case
from iteration106_sample_v1 import DATA,benchmark
from iteration107_fast_v1 import oracle,gaussian_click,ONE
from iteration107_cliques_v1 import optical_x
from iteration108_laws_v1 import prepare
from iteration108_sectors_v1 import prepare_low,Sectors,Global


def weak(r,strength):
    w=np.sinh(r)**2
    return np.arcsinh(np.sqrt(strength*w/w.sum()))


def run(out):
    out.mkdir(exist_ok=False,parents=True);rows=[];quad=oracle(True)
    for i,(m,k,level,n) in enumerate(((16,2,0,262144),(131072,2,0,65536),(256,16,.12,262144),(4096,64,.12,131072),(8192,128,.12,131072),(8192,128,.015,131072))):
        t,r=case(m,k)
        if level:r=weak(r,level)
        start=perf_counter();cfg,st,raw=prepare(t,r);data,metrics=prepare_low(t,r,cfg,raw)
        setup=perf_counter()-start
        baseline=Global(t,r,cfg);batch=min(2048,4194304//m)
        ids=np.unique(np.r_[np.arange(min(4,m)),np.linspace(4,m-1,4,dtype=int)])
        q=gaussian_click(quad,optical_x(t[ids],r),ONE)
        first=np.zeros(2);runs=[];arr={}
        for hybrid in ([False,True] if not level else [False]):
            model=Sectors(cfg,data,baseline if hybrid else None)
            for pn in (False,True):
                speed=benchmark(lambda n,s,o:model.draw(n,pn,s,o),batch,108031000+8*i+2*hybrid+pn,7)
                hist=np.zeros(len(q),int);classes=np.zeros(4,int);sumhist=np.zeros(1024,int)
                for off in range(0,n,batch):
                    x=model.draw(min(batch,n-off),pn,108032000+8*i+2*hybrid+pn,off)
                    hist+=np.bincount((x[:,ids]>0)@(1<<np.arange(len(ids))),minlength=len(hist))
                    hw=np.count_nonzero(x,axis=1);classes+=np.bincount(np.minimum(hw,3),minlength=4)
                    sumhist+=np.bincount(x.sum(1).astype(int),minlength=len(sumhist))
                row=dict(hybrid=hybrid,pnr=pn,us=speed['us'],panel_empirical_tv=float(abs(hist/n-q).sum()/2),empty_gap=abs(classes[0]/n-metrics['vacuum']),singleton_gap=abs(classes[1]/n-metrics['single_click_mass']))
                if pn:row['two_photon_gap']=abs(sumhist[2]/n-metrics['two_photon'])
                runs.append(row);arr[f'panel{int(hybrid)}{int(pn)}']=hist;arr[f'sum{int(hybrid)}{int(pn)}']=sumhist
                assert np.array_equal(model.draw(85,pn,108033000+i,17)[8:],model.draw(77,pn,108033000+i,25))
        print(m,k,level,metrics['full_click_tv_bound'],metrics['full_pnr_tv_bound'],runs,flush=True)
        rows.append(dict(m=m,k=k,emitted_pairs=float(np.sum(np.sinh(r)**2)),weak=level,shots=n,setup=setup,batch=batch,analytic=metrics,runs=runs))
        np.savez_compressed(out/f'holdout_{i}.npz',target=q,**arr)
    save(out,'summary.json',dict(rows=rows));manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):run(p.parse_args().output.resolve())
