"""Complete-sector certificate and actual dense records as both source rank and width grow."""
import argparse
import ctypes as ct
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from iteration100_baselines_v1 import frugal,pair_reference,purification,pe_sample
from iteration106_width_v1 import case
from iteration106_sample_v1 import DATA,benchmark
from iteration107_cliques_v1 import optical_x
from iteration107_fast_v1 import oracle,gaussian_click,ONE
from iteration108_low_bench_v1 import weak
from iteration108_matching_v1 import Matching


def run(out):
    out.mkdir(exist_ok=False,parents=True);rows=[];quad=oracle(True)
    jobs=[(16,2,0.,4,65536),(256,16,.12,4,262144),(4096,64,.12,4,131072),(8192,128,.12,4,65536),
          (131072,2,.12,4,65536),(131072,128,.12,4,32768),(131072,128,.015,4,32768)]
    prev=None
    for i,(m,k,level,cap,n) in enumerate(jobs):
        if prev!=(m,k):t,original=case(m,k)
        prev=(m,k);r=weak(original,level) if level else original
        start=perf_counter();model=Matching(t,r,cap);prep=perf_counter()-start
        idx=np.unique(np.r_[np.arange(4),np.linspace(4,m-1,4,dtype=int)])
        truth=gaussian_click(quad,optical_x(t[idx],r),ONE)
        batch=min(2048,4194304//m);runs=[]
        pairq=np.eye(4)+optical_x(t[:2],r);pnrtruth=pair_reference(pairq,12)
        arr=dict(truth=truth,pnr_truth=pnrtruth)
        for pn in (False,True):
            speed=benchmark(lambda n,s,o:model.draw(n,pn,s,o),batch,108042000+2*i+pn,11)
            h=np.zeros(len(truth),int);p=np.zeros(169,int);tally=np.zeros((len(r),cap+2,3),np.uint64)
            full=np.zeros(1024,int);a32=np.zeros(169,int)
            for off in range(0,n,batch):
                x,stats=model.draw(min(batch,n-off),pn,108043000+2*i+pn,off,stats=True);tally+=stats
                h+=np.bincount((x[:,idx]>0)@(1<<np.arange(len(idx))),minlength=len(h))
                full+=np.bincount(x.sum(1).astype(int),minlength=1024)
                if pn:
                    v=np.bincount(x[:,:2].clip(0,12)@np.array([1,13]),minlength=169);p+=v
                    if off<32768:a32+=v
            row=dict(pnr=pn,us=speed['us'],quartiles=speed['quartiles'],panel_empirical_tv=float(abs(h/n-truth).sum()/2),
                     fallbacks=int(tally[:,:,2].sum()),tally=tally.tolist())
            if pn:row.update(pair_empirical_tv=float(abs(p/n-pnrtruth).sum()/2),equal32768_tv=float(abs(a32/32768-pnrtruth).sum()/2))
            runs.append(row);arr[f'panel{int(pn)}']=h;arr[f'full{int(pn)}']=full
        x=model.draw(513,True,108044000+i,3)
        assert np.array_equal(x[17:]>0,model.draw(496,False,108044000+i,20))
        entry=dict(m=m,k=k,emitted_pairs=float((np.sinh(r)**2).sum()),shots=n,batch=batch,preparation_s=prep,isometry_error=model.isometry,
                   certificate=model.cert,runs=runs,ports=idx.tolist())
        if i in (0,2,5):
            lib=ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'))
            xx=optical_x(t[:16],r);q=gaussian_click(quad,xx,ONE);entry['frugal']=[]
            for order in (2,4):
                pr,ts=frugal(lib,np.eye(32)[None,:,:]+xx,order,108045000+100*i+10*order)
                entry['frugal'].append(dict(K=order,observed=16,us=float(np.median(ts)),prefix_tv=float(abs(q-pr).sum()/2)))
            arr['optical_prefix16']=q
        if i==2:
            st,u,_=purification(pairq[0]);times=[];hist=np.zeros(169,int)
            for j in range(16):
                start=perf_counter()
                x,stats=pe_sample(st,u,2048,seed=108046000+j,device='cpu',return_stats=True,impl='v2',window=4,max_photons=64)
                if j:times.append((perf_counter()-start)*1e6/2048)
                assert not np.any(stats.censored)
                hist+=np.bincount(x[:,:2].clip(0,12)@np.array([1,13]),minlength=169)
            entry['pe']=dict(us=float(np.median(times)),empirical_tv=float(abs(hist/32768-pnrtruth).sum()/2),shots=32768)
            arr['pe_pair']=hist
        np.savez_compressed(out/f'case{i}.npz',**arr)
        rows.append(entry);print(m,k,level,model.cert['full_joint_tv_bound'],[(a['us'],a['fallbacks']) for a in runs],flush=True)
        del model,x
    save(out,'summary.json',dict(rows=rows));manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=4,user_api='blas'):run(p.parse_args().output.resolve())
