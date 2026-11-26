"""Matched upstream Frugal and photonic-engines observables plus untouched Jiuzhang panels."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from iteration104_validate_v1 import interval
from iteration106_sample_v1 import DATA,read,benchmark
from iteration108_device_v1 import Global,fingerprint
from iteration108_laws_v1 import prepare

BENCH=DATA/'iteration108_bench_v1'


def run(out):
    out.mkdir(exist_ok=False,parents=True);rows=[];hist={}
    with np.load(BENCH/'jiuzhang.npz') as z:t,r=z['transfer'],z['squeezing']
    d=DATA/'iteration104_frugal_device_v1'
    with np.load(DATA/'iteration107_projection_v1/pnr.npz') as z:pair=z['target']
    pri=DATA/'iteration106_packed_v1'
    selected=np.r_[289,861,np.setdiff1d(np.r_[np.arange(288,296),np.arange(856,864)],[289,861])]
    order=np.r_[selected,np.setdiff1d(np.arange(len(t)),selected)]
    start=perf_counter();spot,meta,raw=prepare(t,r,16,2,order);setup=perf_counter()-start
    spot['input_hash']=fingerprint(t,r)
    cp=raw['pnr'].sum(2);xx,yy=np.indices(cp.shape);bucket=np.minimum(xx,12)+13*np.minimum(yy,12)
    optical=np.bincount(bucket.ravel(),weights=cp.ravel(),minlength=169)
    assert abs(optical-pair).sum()/2<1e-9
    # The physical ordering in the earlier complete binary128 reference differs.
    with np.load(pri/'panel_1.npz') as z:ports=z['ports'];truth=z['target']
    transform=sum(((np.arange(65536)>>list(selected).index(i))&1)<<k for k,i in enumerate(ports))
    direct=np.bincount(transform,weights=raw['click'].reshape(-1,65536).sum(0))
    assert abs(truth-direct).sum()/2<1e-8
    with np.load(BENCH/'tables1152.npz') as z:base=dict(z)
    models=[Global(t,r,base),Global(t,r,spot)]
    tests=[np.arange(8),np.arange(288,296),np.arange(144)]
    ext=[np.load(d/f'panel_{i}_frugal32_2.npz')['truth'] for i in range(2)]
    for j,model in enumerate(models):
        for pn in (False,True):
            for i,ids in enumerate(tests):
                batch=2048 if i!=2 else 64
                def draw(n,s,o):return np.ascontiguousarray((model.draw(n,pn,s,o)[:,ids]>0).astype(np.uint8))
                tm=benchmark(draw,batch,108061000+j*100+10*i+pn,11)
                info=dict(configuration=j,pnr=pn,observed=len(ids),first=int(ids[0]),batch=batch,us=tm['us'])
                if i<2:
                    hh=[]
                    for phase,n in enumerate((32768,262144)):
                        h=np.zeros(256,np.int64)
                        for off in range(0,n,4096):
                            x=draw(4096,108062000+100*j+10*i+2*pn+phase,off)
                            h+=np.bincount(x@(1<<np.arange(8)),minlength=256)
                        hh.append(h)
                    info.update(interval(ext[i],hh[1],hh[0],262144,.0001));hist[f'p{j}{int(pn)}{i}']=hh[1]
                rows.append(info)
    hp=np.zeros(169,int);small=hp.copy();events=np.zeros(65536,int)
    for off in range(0,1048576,4096):
        x=models[1].draw(4096,True,108063000,off)
        val=np.bincount(np.minimum(x[:,[289,861]],12)@np.array([1,13]),minlength=169)
        hp+=val
        if off<32768:small+=val
        events+=np.bincount((x[:,ports]>0)@(1<<np.arange(16)),minlength=65536)
    old=read(DATA/'iteration107_peers_v1/summary.json')
    speed=[benchmark(lambda n,s,o:models[1].draw(n,p,s,o),16384,108064000+p,12)['us'] for p in (False,True)]
    np.savez_compressed(out/'checks.npz',pnr=hp,pnr32768=small,target=pair,pnr_click=events,click_reference=truth,**hist)
    np.savez_compressed(out/'targeted.npz',**spot)
    save(out,'summary.json',dict(rows=rows,setup=setup,focused_timing=speed,optical=meta,frugal_gpu=old['upstream_gpu'],
        focused_pnr_target_check=float(abs(optical-pair).sum()/2),focused_click_target_check=float(abs(direct-truth).sum()/2),
        pnr_equal_tv=float(abs(small/32768-pair).sum()/2),pnr_million_tv=float(abs(hp/1048576-pair).sum()/2),
        pe_equal_tv=old['pe_pair']['pe_empirical_tv'],pe_us=old['pe_pair']['pe_us']))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):run(p.parse_args().output.resolve())
