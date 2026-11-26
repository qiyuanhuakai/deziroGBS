"""Full-record tests of the vacuum and global photon law and independent optical detector panels."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from iteration104_validate_v1 import interval
from iteration106_field_v1 import actual_input,optical
from iteration106_width_v1 import case
from iteration106_sample_v1 import DATA,benchmark
from iteration107_cliques_v1 import optical_x
from iteration107_fast_v1 import gaussian_click,oracle,ONE
from iteration108_laws_v1 import prepare
from iteration108_device_v1 import Global,fingerprint


def law_total(cube):
    x=np.indices(cube.shape,sparse=True)
    return np.bincount((x[0]+x[1]+x[2]).ravel(),weights=cube.ravel())


def run(out):
    out.mkdir(exist_ok=False,parents=True);rows=[];quad=oracle(True)
    hardware=np.load(DATA/'iteration104_record_order_v1/windows.npz')['histograms'][32:].sum(0)
    for i,(m,k,n) in enumerate(((4,2,262144),(16,2,262144),(4096,64,262144),(8192,128,131072),(131072,2,65536),(1152,25,1048576))):
        jz=m==1152;t,r=actual_input() if jz else case(m,k)
        start=perf_counter();cfg,meta,raw=prepare(t,r);cfg['input_hash']=fingerprint(t,r)
        model=Global(t,r,cfg);torch.cuda.synchronize();setup=perf_counter()-start
        prefix=np.arange(min(8,m));other=(np.array([0,159,169,197,291,326,346,379]) if jz else np.unique(np.r_[np.arange(min(4,m)),np.linspace(min(4,m-1),m-1,min(4,m),dtype=int)]))
        panels=[prefix,other];truth=[gaussian_click(quad,optical_x(t[ids],r),ONE) for ids in panels]
        pc=raw['click'];pmf=law_total(raw['pnr']);e=optical(t,r)
        z=np.exp(-2j*np.pi*np.arange(1025)/2048)
        expect=np.fft.irfft(np.exp(-.5*np.log1p((1-z[:,None])*e).sum(1)),n=2048)
        check=float(abs(expect[:len(pmf)]-pmf).sum()/2+abs(expect[len(pmf):]).sum()/2);assert check<2e-8
        h=int(cfg['h']);bounds=cfg['bounds'];batch=min(2048,4194304//m)
        entry=dict(m=m,sources=k,spectra=len(r),shots=n,setup=setup,analytic=meta,total_reference_tv=check,panels=[ids.tolist() for ids in panels],runs=[])
        histograms={}
        for pn in (False,True):
            ts=benchmark(lambda n,s,o:model.draw(n,pn,s,o),batch,108010000+2*i+pn,9)
            if jz:big=benchmark(lambda n,s,o:model.draw(n,pn,s,o),16384,108011000+pn,9)
            counts=[np.zeros(1<<len(ids),int) for ids in panels];coarse=np.zeros(len(pc),int)
            photons=np.zeros(2048,int);clicks=np.zeros(m+1,int);project=np.zeros(raw['pnr'].shape,int)
            for off in range(0,n,batch):
                x=model.draw(min(batch,n-off),pn,108012000+2*i+pn,off)
                hw=np.count_nonzero(x,axis=1);clicks+=np.bincount(hw,minlength=m+1)
                for j,ids in enumerate(panels):counts[j]+=np.bincount((x[:,ids]>0)@(1<<np.arange(len(ids))),minlength=len(counts[j]))
                label=(x[:,:h]>0)@(1<<np.arange(h))
                for g,(a,b) in enumerate(zip(bounds[:-1],bounds[1:])):label+=(np.any(x[:,a:b],axis=1).astype(int)<<(h+g))
                coarse+=np.bincount(label,minlength=len(pc))
                if pn:
                    a,b=x[:,:2].astype(np.int64).T;total=x.sum(1,dtype=np.int64);c=total-a-b
                    assert np.all(a<project.shape[0]) and np.all(b<project.shape[1]) and np.all(c<project.shape[2])
                    np.add.at(project,(a,b,c),1);photons+=np.bincount(total,minlength=2048)
            replay=model.draw(251,pn,108013000+2*i+pn,41)
            assert np.array_equal(replay[83:],model.draw(168,pn,108013000+2*i+pn,124))
            run=dict(pnr=pn,timing=ts,panel_empirical_tv=[float(abs(c/n-p).sum()/2) for c,p in zip(counts,truth)],
                     click_coarse_empirical_tv=float(abs(coarse/n-pc).sum()/2),empty_gap=float(abs(clicks[0]/n-meta['vacuum'])))
            if pn:run.update(total_ks=float(abs(photons.cumsum()/n-expect.cumsum()).max()),pnr_coarse_empirical_tv=float(abs(project/n-raw['pnr']).sum()/2))
            if jz:run.update(big=big,hardware_ks=float(abs(clicks.cumsum()/n-hardware.cumsum()/hardware.sum()).max()))
            histograms.update({f'{int(pn)}_{name}':val for name,val in dict(click=clicks,photon=photons,coarse=coarse,project=project,panel0=counts[0],panel1=counts[1]).items()})
            entry['runs'].append(run)
        rows.append(entry);print(m,meta['vacuum'],[(z['timing']['us'],z['empty_gap'],z.get('total_ks'),z['panel_empirical_tv']) for z in entry['runs']],flush=True)
        np.savez_compressed(out/f'law{m}.npz',**raw,panel0=truth[0],panel1=truth[1],optical_total=expect)
        np.savez_compressed(out/f'obs{m}.npz',**histograms)
        if jz or m==4096:
            np.savez_compressed(out/f'tables{m}.npz',**cfg)
            if jz:np.savez_compressed(out/'jiuzhang.npz',transfer=t,squeezing=r)
    save(out,'summary.json',dict(rows=rows,hardware_n=int(hardware.sum())))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):run(p.parse_args().output.resolve())
