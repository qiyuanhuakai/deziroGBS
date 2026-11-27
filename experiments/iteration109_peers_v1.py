"""Equal-device, equal-output GPU trials and independent numerical controls."""
import argparse
import ctypes as ct
import json
import os
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import expm
from scipy.special import eval_genlaguerre,gammaln
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import I,D,B,bind,save,manifest
from iteration104_validate_v1 import interval
from iteration106_field_v1 import DATA,actual_input
from iteration106_sample_v1 import benchmark,panel_hist
from iteration109_device_v1 import Local
from iteration109_model_v1 import build
from iteration109_twins_v1 import Twins,optics,covariance
from iteration109_sample_v1 import tv
import torch

def load_frugal():
    rt=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Redist'
    omp=next(rt.glob('MSVC/*/debug_nonredist/x64/Microsoft.VC143.OpenMP.LLVM/libomp140.x86_64.dll'))
    handles=[os.add_dll_directory(str(p)) for p in (omp.parent,Path(shutil.which('nvcc')).parent,Path(shutil.which('g++')).parent)]
    lib=ct.CDLL(str(DATA/'iteration109_frugal_build_v1/frugal_fp32.dll'))
    return handles,lib,bind(lib,'fg72_sample',[I,I,I,ct.c_uint,I,D,B,D,ct.c_char_p,I])

def run(out):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(exist_ok=True,parents=True);torch.backends.cuda.matmul.allow_tf32=False
    held=np.load(DATA/'iteration109_twins_v1/laws.npz');truth=held['target'];arrays={};rows=[]
    l,f=optics(16);tw=Twins(16,l,f,DATA/'iteration109_twins_v1/gpu.dll')
    cov=2*covariance(16,l,f)+np.eye(32);handles,lib,fr=load_frugal()
    def record(k,n,seed):
        if k==0:return tw.draw(n,False,seed)
        x=np.empty((n,16),np.uint8);s=np.zeros(6);msg=ct.create_string_buffer(4096)
        rc=fr(16,k,n,seed,16,cov,x,s,msg,len(msg))
        if rc or s[-1]!=32:raise RuntimeError(msg.value)
        return x
    shots=262144;test=32768;batch=2048;timings={k:[] for k in (0,2,4)};rng=np.random.default_rng(109451000)
    for rep in range(10):
        for k in rng.permutation([0,2,4]):
            k=int(k);st=perf_counter();x=record(k,batch,109452000+100*k+rep)
            us=(perf_counter()-st)*1e6/batch
            if rep:timings[k].append(us)
    for k in (0,2,4):
        hists=[]
        for part,n in enumerate((test,shots)):
            bins=np.zeros(65536,np.int64)
            for off in range(0,n,8192):
                v=record(k,8192,109460000+10000*k+1000*part+off//8192)
                bins+=panel_hist(v,np.arange(16))
            hists.append(bins)
        trial,h=hists;estimate=interval(truth,h,trial,shots,.0001)
        # Fixed event selected using the separate CPU model; supplies power
        # against Frugal even when the complete histogram is sampling-noisy.
        support=held['frugal4']>truth;gap=abs(h[support].sum()/shots-truth[support].sum())
        arrays[f'gpu_{k}']=h;arrays[f'trial_{k}']=trial
        rows.append(dict(k=k,batch=batch,us=float(np.median(timings[k])),times=timings[k],shots=shots,**estimate,
                         frugal4_event_gap=gap))
    # Same transfer and same eight returned host bits as the archived Frugal comparison.
    t,r=actual_input();device=Local(build(t,r,8),DATA/'iteration109_device_v1/gpu.dll')
    actual=[]
    observed=np.load(DATA/'iteration109_device_v1/observed.npz')
    for idx,ids in enumerate((np.arange(8),np.arange(288,296))):
        ref=observed[('first8','space36')[idx]+'_target']
        for pn in (False,True):
            def draw(n,s,o):return np.ascontiguousarray((device.draw(n,pn,s,o)[:,ids]>0).astype(np.uint8))
            tm=benchmark(draw,2048,109560000+idx*2+pn,11)
            trial=np.zeros(256,np.int64)
            for off in range(0,32768,2048):trial+=panel_hist(draw(2048,109561000+idx*2+pn,off),np.arange(8))
            h=observed[('pnr' if pn else 'click')+'_panels'][idx]
            actual.append(dict(panel=idx,pnr=pn,timing=tm,**interval(ref,h,trial,int(h.sum()),.0001)))
            arrays[f'actual_trial_{idx}_{int(pn)}']=trial
    # Independent Fock-basis matrix exponential of the displacement operator.
    op=np.diag(np.sqrt(np.arange(1,90)),1);fockerr=0.
    for rate in (.001,.28,1.5,6.):
        mat=expm(np.sqrt(rate)*(op.T-op))**2
        for n in range(15):
            for m in range(23):
                a,b=sorted((n,m));delta=b-a
                p=np.exp(-rate+gammaln(1+a)-gammaln(1+b)+delta*np.log(rate))*eval_genlaguerre(a,delta,rate)**2
                fockerr=max(fockerr,abs(mat[m,n]-p))
    assert fockerr<1e-12
    # Quantify sampling noise for this specific 65,536-word law.
    probs=np.maximum(truth,0);probs/=probs.sum()
    spread=[tv(rng.multinomial(shots,probs)/shots,truth) for _ in range(256)]
    np.savez_compressed(out/'samples.npz',**arrays)
    save(out,'summary.json',dict(synthetic_gpu=rows,actual_gpu=actual,ideal_empirical_tv_95=np.quantile(spread,[.025,.975]).tolist(),
                                 fock_matrix_max_abs_error=fockerr,control_entries=4*15*23));manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(ap.parse_args().output.resolve())
