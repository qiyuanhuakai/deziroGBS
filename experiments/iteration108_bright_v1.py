"""Many detected photons formed by incoherently combining many independently squeezed spectra."""
import argparse
import ctypes as ct
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest,bind,I,D,B
from iteration100_baselines_v1 import frugal
from iteration106_field_v1 import optical,actual_input
from iteration106_width_v1 import case
from iteration106_sample_v1 import DATA,benchmark
from iteration107_fast_v1 import gaussian_click,oracle,ONE
from iteration107_cliques_v1 import optical_x
from iteration108_low_bench_v1 import weak
from iteration108_replicas_v1 import Replicas


def repeated_bits(p,replicas):
    q=p.astype(np.longdouble).copy();bits=(len(q)-1).bit_length()
    for j in range(bits):
        a=q.reshape(-1,2,1<<j);a[:,1]+=a[:,0]
    q=(q/q[-1])**replicas
    for j in range(bits):
        a=q.reshape(-1,2,1<<j);a[:,1]-=a[:,0]
    assert q.min()>-3e-9 and abs(q.sum()-1)<1e-10
    return q.astype(float)


def run(out):
    out.mkdir(exist_ok=False,parents=True);quad=oracle(True);rows=[]
    for i,(m,k,replicas,n,cache) in enumerate(((1152,25,512,16384,32),(4096,64,1024,8192,144),(8192,128,1024,2048,128))):
        if m==1152:t,r=actual_input();r=r[:1]
        else:t,r=case(m,k)
        r=weak(r,.12);st=perf_counter();model=Replicas(t,r,replicas,cap=5,tries=8192,cache_mb=cache)
        setup=perf_counter()-st;e=optical(t,r);z=np.exp(-2j*np.pi*np.arange(1025)/2048)
        pmf=np.fft.irfft(np.exp(-.5*replicas*np.log1p((1-z[:,None])*e).sum(1)),n=2048)
        ids=np.unique(np.r_[np.arange(4),np.linspace(4,m-1,4,dtype=int)])
        p=repeated_bits(gaussian_click(quad,optical_x(t[ids],r),ONE),replicas)
        stats=[];batch=128
        for pn in (False,True):
            speed=benchmark(lambda b,s,o:model.draw(b,pn,s,o),batch,108081000+2*i+pn,7)
            h=np.zeros(256,np.int64);sums=np.zeros(2048,np.int64);clickmean=0;tally=np.zeros((7,3),np.uint64)
            for off in range(0,n,batch):
                x,v=model.draw(batch,pn,108082000+2*i+pn,off,stats=True);tally+=v
                h+=np.bincount((x[:,ids]>0)@(1<<np.arange(8)),minlength=256)
                sums+=np.bincount(x.sum(1,dtype=np.int64),minlength=len(sums))
                clickmean+=np.count_nonzero(x)
            assert tally[:,0].sum()==n*replicas
            d=dict(pnr=pn,us=speed['us'],quartiles=speed['quartiles'],mean_clicks=clickmean/n,panel_empirical_tv=float(abs(h/n-p).sum()/2),fallbacks=int(tally[:,2].sum()))
            if pn:d.update(mean_photons=float(sums@np.arange(len(sums))/n),total_ks=float(abs(sums.cumsum()/n-pmf.cumsum()).max()))
            stats.append(d);np.savez_compressed(out/f'case{i}_{int(pn)}.npz',panel=h,truth=p,total=sums,optical_total=pmf,tally=tally)
        x=model.draw(37,True,108083000+i)
        assert np.array_equal(x[7:]>0,model.draw(30,False,108083000+i,7))
        entry=dict(m=m,k=k,replicas=replicas,emitted_pairs=.12*replicas,expected_photons=float(pmf@np.arange(len(pmf))),
              bound=model.bound,per_replica=model.cert['full_joint_tv_bound'],prep_s=setup,batch=batch,shots=n,
              cache_bytes=model.cached.nbytes,conditional_second_cached_mass=model.cached_mass,runs=stats)
        if m==4096:
            lib=ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'))
            sampler=bind(lib,'fc96_sample16',[I,I,ct.c_uint,D,B,D,ct.c_char_p,I]);msg=ct.create_string_buffer(2048)
            xx=optical_x(t[:16],r);gg=np.eye(32)[None,:,:]+xx;full=repeated_bits(gaussian_click(quad,xx,ONE),replicas)
            entry['frugal']=[]
            for K in (2,4):
                approx,_=frugal(lib,gg,K,108084000+30*K);fq=repeated_bits(approx,replicas)
                times=[]
                for j in range(4):
                    # Fuse all independent spectra in one upstream call.
                    count=16*replicas;v=np.empty((count,16),np.uint8);st=np.zeros(7);start=perf_counter()
                    assert sampler(K,count,108085000+20*K+j,2*gg[0]-np.eye(32),v,st,msg,len(msg))==0
                    obs=np.bitwise_or.reduce(v.reshape(16,replicas,16),axis=1);assert obs.shape==(16,16)
                    if j:times.append((perf_counter()-start)*1e6/16)
                entry['frugal'].append(dict(K=K,observed=16,batch=16,us=float(np.median(times)),tv=float(abs(fq-full).sum()/2)))
            entry['full_output_same_batch_us']=benchmark(lambda b,s,o:model.draw(b,False,s,o),16,108086000,9)['us']
        print(m,replicas,entry['bound'],stats,flush=True);rows.append(entry)
    save(out,'summary.json',dict(rows=rows));manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=4,user_api='blas'):run(p.parse_args().output.resolve())
