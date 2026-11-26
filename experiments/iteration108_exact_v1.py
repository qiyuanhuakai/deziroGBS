"""Full 16-bit numerical TV, including the entire rejected/over-cap fallback law."""
import argparse
import ctypes as ct
from pathlib import Path
import shutil
import subprocess
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,save,manifest,I,D,bind
from iteration100_baselines_v1 import frugal
from iteration106_width_v1 import case
from iteration106_sample_v1 import DATA,benchmark
from iteration107_fast_v1 import oracle,gaussian_click,ONE
from iteration107_cliques_v1 import optical_x
from iteration108_low_bench_v1 import weak
from iteration108_matching_v1 import certificate
from iteration108_compact_v1 import Compact


def run(out):
    out.mkdir(exist_ok=False,parents=True)
    eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'
    subprocess.run([shutil.which('g++'),'-O3','-ffp-contract=off','-std=c++20','-march=x86-64-v3','-fopenmp','-shared','-I',str(eigen),str(ROOT/'native/matching_law108_v1.cpp'),'-o',str(out/'laws.dll')],check=True)
    quad=oracle(True);fn=bind(ct.CDLL(str(out/'laws.dll')),'matching_law108',[I]*5+[D]*4)
    rows=[]
    for i,(m,k,level,ns) in enumerate(((16,2,0,1),(16,6,1.2,1),(16,6,2.5,1),(16,3,1.2,2),(131072,2,0,1))):
        t,orig=case(m,k);r=weak(orig,level) if level else orig
        if ns==2:r=np.vstack((r,-.72*r[:,::-1]))
        ids=np.arange(16);xx=optical_x(t[ids],r);truth=gaussian_click(quad,xx,ONE)
        tries=512;cert=certificate(np.tanh(r),6,tries)
        keep=np.array([-np.expm1(tries*np.log1p(-np.minimum(1.-1e-16,np.array(s['mean_acceptance'])))) for s in cert['spectra']])
        buf=np.empty((8,65536))
        assert fn(16,k,ns,6,12,np.ascontiguousarray(t[ids]).view(float),np.ascontiguousarray(np.tanh(r)),keep,buf)==0
        check=float(abs(buf[0]-truth).sum()/2);assert check<2e-9 and max(abs(buf.sum(1)-1))<1e-10 and buf.min()>-1e-10
        tv=(abs(buf[1:]-truth).sum(1)/2).tolist();tim=[]
        batch=min(2048,4194304//m)
        for cap in (3,4,5,6):
            mod=Compact(t,r,cap,tries)
            tm=[benchmark(lambda n,s,o:mod.draw(n,pn,s,o),batch,108071000+80*i+2*cap+pn,7)['us'] for pn in (False,True)]
            tim.append(dict(cap=cap,batch=batch,us=tm,full_tv_bound=mod.cert['full_joint_tv_bound']))
        lib=ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'));f=[]
        for K in (2,4):
            p,ts=frugal(lib,np.eye(32)[None,:,:]+xx,K,108072000+100*i+20*K)
            f.append(dict(K=K,us=float(np.median(ts)),tv=float(abs(truth-p).sum()/2)))
        # Native frozen-event test of the complete analytical cap-4 law.
        mod=Compact(t,r,4,tries);n=262144 if m==16 else 32768;seen=np.zeros(65536,np.int64)
        for off in range(0,n,batch):
            x=mod.draw(min(batch,n-off),False,108073000+i,off)
            seen+=np.bincount(x[:,:16]@(1<<np.arange(16)),minlength=65536)
        event=buf[5]>truth
        gap=float(abs(seen[event].sum()/n-buf[5,event].sum()))
        rows.append(dict(m=m,k=k,spectra=ns,emitted_pairs=float((np.sinh(r)**2).sum()),tries=tries,oracle_check=check,
                         tv_by_cap=tv,run=tim,frugal=f,native_n=n,cap4_validation_event_gap=gap))
        print(m,k,ns,tv,[(p['K'],p['tv']) for p in f],tim[-1],flush=True)
        np.savez_compressed(out/f'laws_{i}.npz',target=truth,laws=buf,observed=seen)
    save(out,'summary.json',dict(rows=rows));manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(p.parse_args().output.resolve())
