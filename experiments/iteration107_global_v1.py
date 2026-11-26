"""Compare full-output photon totals and all-zero click events to the quantum Gaussian law."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from iteration106_field_v1 import optical
from iteration106_width_v1 import case
from iteration107_fast_v1 import Fast,prepare,tables,OUT as FAST
from iteration107_resolved_law_v1 import chernoff


def run(out):
    out.mkdir(exist_ok=False,parents=True);rows=[];laws={}
    for i,(m,k,n) in enumerate(((16,2,1048576),(4096,64,262144),(8192,128,131072),(131072,2,65536))):
        t,r=case(m,k)
        if m==4096:
            with np.load(FAST/'case4096.npz') as z:params=dict(z)
        else:params=tables(t,r,prepare(t,r))
        model=Fast(t,r,params)
        e=optical(t,r);length=2048
        z=np.exp(-2j*np.pi*np.arange(length//2+1)/length)
        exact=np.fft.irfft(np.exp(-.5*np.log1p((1-z[:,None])*e).sum(1)),n=length)
        tail=chernoff(e,length);assert abs(exact.sum()-1)<1e-12 and exact.min()>-1e-13 and tail<1e-12
        # Pre-specified CDF and vacuum observables; one joint 99% sampling
        # allowance over four sizes and both output types.
        rad=float(np.sqrt(np.log(16/.01)/(2*n)))
        batch=min(2048,4194304//m);h=np.zeros(length,np.int64);empty=0
        for off in range(0,n,batch):
            count=min(batch,n-off)
            x=model.draw(count,True,107071000+2*i,off)
            h+=np.bincount(x.sum(1).astype(int),minlength=length)
            y=model.draw(count,False,107071001+2*i,off)
            empty+=int(np.sum(~np.any(y,axis=1)))
        vacuum=float(np.exp(-.5*np.log1p(e).sum()))
        gap=float(abs(h.cumsum()/n-exact.cumsum()).max());vac=abs(empty/n-vacuum)
        rows.append(dict(m=m,sources=k,shots=n,pnr_total_ks=gap,dkw_joint_radius=rad,
            total_count_full_pnr_tv_lower=max(0,gap-rad-tail),direct_vacuum_gap=float(vac),
            empty_clicks=empty,optical_vacuum=vacuum,full_click_tv_lower=max(0,float(vac)-rad-tail),
            fft_tail=tail))
        print(rows[-1],flush=True);laws['observed_'+str(m)]=h;laws['optical_'+str(m)]=exact
    np.savez_compressed(out/'totals.npz',**laws)
    save(out,'summary.json',dict(rows=rows,scope='99% simultaneous DKW bounds with independent experiment and record domains; numerical optical reference.'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):run(p.parse_args().output.resolve())
