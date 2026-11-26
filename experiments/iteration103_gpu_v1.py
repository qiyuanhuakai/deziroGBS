"""CUDA raw continuation sampler, including host transfer and matched FP32/FP64 record checks."""
import argparse
import ctypes as ct
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.stats import beta
from threadpoolctl import threadpool_limits

from iteration100_spatial_v1 import ROOT,HIST,I,U,save,manifest,H
from iteration103_sample_v1 import Sampler,RUN,DATA

GPU=DATA/'iteration103_gpu_v1'
P=ct.c_void_p
CU=ct.c_uint32


def build(out,source='continuation_gpu103_v1.cu'):
    vc=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat'
    nvcc=shutil.which('nvcc')
    src=ROOT/'native'/source
    with tempfile.TemporaryDirectory(prefix='gbs103-nvcc-') as tmp:
        temp=Path(tmp)
        shutil.copyfile(src,temp/'kernel.cu')
        if source!='continuation_gpu103_v1.cu':
            shutil.copyfile(ROOT/'native/continuation_gpu103_v1.cu',temp/'continuation_gpu103_v1.cu')
        (temp/'env.bat').write_text(f'@call "{vc}" >nul\n@set\n',encoding='ascii')
        env=os.environ.copy()
        for line in subprocess.check_output(['cmd','/d','/c','env.bat'],cwd=temp,text=True,errors='replace').splitlines():
            if '=' in line:
                k,v=line.split('=',1)
                if k:env[k]=v
        cmd=[nvcc,'-O3','-arch=sm_120','--fmad=false','-std=c++17','-shared','-Xcompiler','/MD,/utf-8','kernel.cu','-o','gpu.dll']
        proc=subprocess.run(cmd,cwd=temp,env=env,capture_output=True,text=True,errors='replace')
        if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
        shutil.copyfile(temp/'gpu.dll',out/'gpu.dll')
    save(out,'build.json',dict(command=cmd,source_sha256=H(src),device=torch.cuda.get_device_name(),torch=torch.__version__))


class Cuda:
    def __init__(self,dll,model,wide=False):
        self.lib=ct.CDLL(str(dll))
        self.normal=self.lib.normal103_launch
        self.normal.argtypes=[I,I,U,CU,I,P,P]
        self.emit=self.lib.emit103_launch
        self.emit.argtypes=[I,I,I,U,CU,I]+[P]*15
        self.wide=wide
        self.dtype=torch.float64 if wide else torch.float32
        self.groups=len(model.order)
        self.factor=torch.as_tensor(model.factor.T.copy(),device='cuda',dtype=self.dtype)
        params=(model.caps,model.offset,model.thresholds,model.starts,model.masks,model.prior,model.weights,
                model.shortcut_cut,model.shortcut_mask,model.shortcut_fallback)
        self.params=[torch.as_tensor(a,device='cuda') for a in params]
        self.pointers=[v.data_ptr() for v in self.params]
        self.error=torch.zeros(1,device='cuda',dtype=torch.int32)
        self.capacity=0

    def sample(self,n,route,seed,offset=0,correction=None):
        if n<1 or offset<0 or offset+n>1<<32 or not 0<=seed<1<<64:raise ValueError('Invalid random domain or sample length')
        if n>self.capacity:
            self.noise=torch.empty((n,2*self.groups),dtype=self.dtype,device='cuda')
            self.latent=torch.empty_like(self.noise)
            self.output=torch.empty((n,8*self.groups),dtype=torch.uint16,device='cuda')
            self.counts=torch.empty((n,self.groups),dtype=torch.uint16,device='cuda')
            self.capacity=n
        stream=torch.cuda.current_stream().cuda_stream
        rc=self.normal(self.groups,n,seed,offset,int(self.wide),self.noise.data_ptr(),stream)
        if rc:raise RuntimeError(f'Normal CUDA launch returned {rc}')
        torch.mm(self.noise[:n],self.factor,out=self.latent[:n])
        if correction is not None:correction.apply(self,n,seed,offset,stream)
        rc=self.emit(self.groups,n,int(route),seed,offset,int(self.wide),self.latent.data_ptr(),*self.pointers,
                     self.output.data_ptr(),self.counts.data_ptr(),self.error.data_ptr(),stream)
        if rc:raise RuntimeError(f'Emission CUDA launch returned {rc}')
        result=self.output[:n].cpu().numpy()
        counts=self.counts[:n].cpu().numpy()
        if self.error.item():raise RuntimeError('Invalid GPU count or optical occupancy mask')
        return result,counts


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    build(out)
    cpu=Sampler(RUN/'draw.dll',RUN/'bridges.npz',False)
    torch.set_float32_matmul_precision('highest')
    fast=Cuda(out/'gpu.dll',cpu)
    wide=Cuda(out/'gpu.dll',cpu,True)
    for model in (fast,wide):
        a,c=model.sample(2240,False,103050000)
        x,d=model.sample(2112,True,103050000,128)
        assert np.array_equal(a[128:],x>0) and np.array_equal(c[128:],d)
        assert np.array_equal(x.reshape(len(x),-1,8).sum(2),d)
    rows=[]
    for name,model in [('fp32',fast),('fp64',wide)]:
        for route in (False,True):
            for n in (2048,8192,32768):
                times=[]
                for rep in range(8):
                    start=perf_counter()
                    model.sample(n,route,103051000+int(route),rep*n)
                    if rep:times.append((perf_counter()-start)*1e6/n)
                rows.append(dict(precision=name,pnr=route,batch=n,us=times,median_us=float(np.median(times))))
    # Both precisions use identical per-group Philox domains, cosine/sine pairs
    # and conditional marks. Any output disagreement bounds the TV by coupling.
    n=1048576
    block=4096
    mismatches=np.zeros(3,np.int64)
    hc=np.zeros(1153,np.int64)
    hp=np.zeros(4096,np.int64)
    for off in range(0,n,block):
        x,c=fast.sample(block,True,103052000,off)
        y,d=wide.sample(block,True,103052000,off)
        mismatches+=np.array([np.any((x>0)!=(y>0),axis=1).sum(),np.any(x!=y,axis=1).sum(),np.any(c!=d,axis=1).sum()])
        hc+=np.bincount((x>0).sum(1),minlength=len(hc))
        hp+=np.bincount(c.sum(1).astype(int),minlength=len(hp))
    upper=[float(beta.ppf(1-.01/3,k+1,n-k)) for k in mismatches]
    hw=np.load(HIST)['total_click_histogram']
    ph=np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf']
    np.savez_compressed(out/'hist.npz',click=hc,photon=hp)
    save(out,'summary.json',dict(rows=rows,n=n,coupled_disagreements=mismatches.tolist(),joint99_coupling_tv_upper=upper,
        hardware_ks=float(abs(hc.cumsum()/n-hw.cumsum()/hw.sum()).max()),
        photon_ks=float(abs(hp.cumsum()/n-np.pad(ph,(0,len(hp)-len(ph))).cumsum()).max()),
        timing='device sampling, both full host arrays, and error synchronization; warm device tables',
        seeds=dict(coupling=103052000),device=torch.cuda.get_device_name()))
    manifest(out)
    print([(r['precision'],r['pnr'],r['batch'],r['median_us']) for r in rows],upper,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=('benchmark','draw'))
    ap.add_argument('output',type=Path)
    ap.add_argument('--shots',type=int,default=2048)
    ap.add_argument('--pnr',action='store_true')
    ap.add_argument('--wide',action='store_true')
    ap.add_argument('--seed',type=int)
    args=ap.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.action=='benchmark':run(args.output.resolve())
        else:
            if args.seed is None:ap.error('--seed is required for draw')
            torch.set_float32_matmul_precision('highest')
            m=Cuda(GPU/'gpu.dll',Sampler(RUN/'draw.dll',RUN/'bridges.npz',False),args.wide)
            x,c=m.sample(args.shots,args.pnr,args.seed)
            with args.output.open('xb') as f:np.savez_compressed(f,records=x,spatial_photons=c)
