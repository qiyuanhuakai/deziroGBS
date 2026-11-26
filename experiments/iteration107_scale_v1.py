"""High source rank, odd width and 131,072-output dense records with high-address checks."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from iteration106_width_v1 import case
from iteration106_field_v1 import factors
from iteration106_sample_v1 import DATA,benchmark
from iteration107_fast_v1 import Fast,tables,prepare,optical_x,oracle,gaussian_click,ONE


def run(out):
    out.mkdir(exist_ok=False,parents=True)
    q=oracle(True);rows=[]
    for i,(m,k) in enumerate(((4097,32),(8192,128),(131072,2))):
        t,r=case(m,k);start=perf_counter()
        small=prepare(t,r);pars=tables(t,r,small)
        model=Fast(t,r,pars);torch.cuda.synchronize();ready=perf_counter()-start
        ids=pars['primary_ids'];size=1<<len(ids);f=factors(t[ids],r)
        p=gaussian_click(q,np.stack([2*v.T@v for v in f]),pars)
        target=gaussian_click(q,optical_x(t[ids],r),ONE)
        move=pars['click_move'][:size].astype(float);cdf=pars['click_cdf'][:size]
        post=(1-move)*p+(p@move)*np.diff(cdf,prepend=np.uint64(0))/2**52
        n=min(1024,4194304//m);times=[]
        for pnr in (False,True):
            ts=benchmark(lambda n,s,o:model.draw(n,pnr,s,o),n,107051000+2*i+pnr,9);times.append(ts)
            a=model.draw(32,pnr,107052000+2*i+pnr)
            b=model.draw(23,pnr,107052000+2*i+pnr,9)
            assert a.shape==(32,m) and np.array_equal(a[9:],b)
        # Exercise the correction-only overflow path with a certain replacement
        # into count 299, leaving the original vacuum output safely below 255.
        meta=dict(m=m,sources=k,batch=n,ready_s=ready,primary_ids=ids.tolist(),primary_base_tv=float(abs(target-p).sum()/2),
                  primary_corrected_tv=float(abs(post-target).sum()/2),click=times[0],pnr=times[1],n_click_words=len(pars['click_move']),
                  n_count_cells=len(pars['pnr_move']),table_bytes=sum(pars[name].nbytes for name in ('click_plan','click_move','click_cdf','pnr_plan','pnr_move','pnr_cdf')),
                  tested_replay=True)
        print(meta,flush=True);rows.append(meta)
    t=np.zeros((1,2),complex);r=np.zeros((1,1))
    conf=tables(t,r,prepare(t,r));conf['pnr_plan'][0,16]=300;conf['pnr_plan'][0,-1]=300
    conf['pnr_move']=np.ones(300,np.float32);conf['pnr_cdf']=np.zeros(300,np.uint64);conf['pnr_cdf'][-1]=1<<52
    result=Fast(t,r,conf).draw(127,True,107053000)
    assert result.dtype==np.int32 and np.all(result==299)
    save(out,'summary.json',dict(rows=rows,correction_only_overflow=int(result.max())))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):run(p.parse_args().output.resolve())
