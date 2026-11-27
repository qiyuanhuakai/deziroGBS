"""Complete 16-port comparisons with Frugal CPU/GPU and photonic-engines."""
import argparse
import ctypes as ct
import os
import shutil
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.linalg import null_space
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from iteration100_baselines_v1 import frugal,pair_reference
from iteration92_pe_common_v1 import GaussianInput,pe_sample
from iteration106_field_v1 import DATA
from iteration106_sample_v1 import benchmark,panel_hist
from iteration109_sample_v1 import tv
from iteration109_peers_v1 import load_frugal
from iteration110_collective_v1 import prepare,banks,Collective


def run(out):
    out.mkdir(exist_ok=False,parents=True);base=DATA/'iteration110_collective_v1'
    v=np.load(base/'optical.npz');truth=v['full_optical'];width=[4]*4
    runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent)) if os.name=='nt' else None
    u,r,eta=[v[n] for n in ('unitary','squeezing','efficiency')]
    plans,_=prepare(ct.CDLL(str(base/'laws.dll')),u,r,eta,width,68)
    model=Collective(plans,width,1,base/'gpu.dll');x=banks(v['core_x'],width,np.arange(16))
    cov=2*x+np.eye(32);q=x+np.eye(32);handles,gpulib,gpu=load_frugal()
    cpulib=ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'))
    arrays={};rows=[];timings={k:[] for k in (0,2,4)};rng=np.random.default_rng(110300000)
    def sample(k,n,seed):
        if k==0:return model.draw(n,False,seed)
        x=np.empty((n,16),np.uint8);s=np.zeros(6);msg=ct.create_string_buffer(4096)
        rc=gpu(16,k,n,seed,16,cov,x,s,msg,len(msg))
        if rc or s[-1]!=32:raise RuntimeError(msg.value)
        return x
    for j in range(11):
        for k in rng.permutation([0,2,4]):
            k=int(k);start=perf_counter();sample(k,2048,110301000+100*k+j);us=(perf_counter()-start)*1e6/2048
            if j:timings[k].append(us)
    shots=262144
    for k in (0,2,4):
        h=np.zeros(65536,np.int64)
        for off in range(0,shots,8192):h+=panel_hist(sample(k,8192,110303000+1000*k+off//8192),np.arange(16))
        arrays[f'gpu{k}']=h
        row=dict(cutoff=k,gpu_us=float(np.median(timings[k])),timings=timings[k],shots=shots,gpu_empirical_tv=tv(h/shots,truth))
        if k:
            p,t=frugal(cpulib,q[None],k,110310000+20*k);row.update(cpu_us=float(np.median(t)),cpu_model_tv=tv(p,truth));arrays[f'cpu{k}']=p
        else:row['model_tv']=tv(v['full_model'],truth)
        rows.append(row)
    phys=np.repeat(np.sqrt(eta)[:,None]*u,4,axis=0)/2
    transfer=np.r_[phys,np.sqrt(1-eta)[:,None]*u];optics=np.column_stack((transfer,null_space(transfer.conj().T)))
    gs=GaussianInput(r,tuple(range(4)),20);B,_,_=gs.bargmann(optics)
    error=float(abs(B+(transfer*np.tanh(r))@transfer.T).max());assert error<1e-12
    e=np.linalg.eigvalsh(x);z=np.exp(-2j*np.pi*np.arange(257)/512)
    tot=np.fft.irfft(np.exp(-.5*np.log1p((1-z[:,None])*e).sum(1)),n=512)
    ids=np.array([0,8,16,24]);pt=pair_reference(q[np.ix_(ids,ids)][None],12);pnrows=[]
    ns=4096;batch=128
    for i,backend in enumerate(('collective','cpu','cuda')):
        h=np.zeros(169,int);hh=np.zeros(512,int);times=[];censored=0
        for off in range(0,ns,batch):
            start=perf_counter()
            if backend=='collective':a=model.draw(batch,True,110320000,off)
            else:
                a,st=pe_sample(gs,optics,batch,seed=110321000+i*1000+off//batch,device=backend,impl='v2',window=4,max_photons=96,return_stats=True)
                a=a[:,:16].copy();censored+=int(np.sum(st.censored))
            us=(perf_counter()-start)*1e6/batch
            if off:times.append(us)
            h+=np.bincount(a[:,[0,8]].clip(0,12)@np.array([1,13]),minlength=169)
            hh+=np.bincount(a.sum(1).astype(int),minlength=512)
        assert censored==0
        pnrows.append(dict(backend=backend,us=float(np.median(times)),shots=ns,censored=censored,
                           pair_empirical_tv=tv(h/ns,pt),total_empirical_ks=float(abs((hh/ns-tot).cumsum()).max())))
        arrays[backend+'_pair']=h;arrays[backend+'_total']=hh
        print(backend,pnrows[-1],flush=True)
    np.savez_compressed(out/'laws.npz',target=truth,pair_target=pt,total_target=tot,**arrays)
    save(out,'summary.json',dict(click=rows,pnr=pnrows,bargmann_error=error));manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(ap.parse_args().output.resolve())
