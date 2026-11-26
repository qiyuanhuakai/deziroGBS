"""Same RTX 5080 Laptop, same observed ports and batch sizes for direct-click Frugal and both calibrated models."""
import argparse
import ctypes as ct
import os
from pathlib import Path
import shutil
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ORDER,I,D,B,bind,Native,save,manifest,H
from iteration100_projection_v1 import physical
from iteration103_sample_v1 import BASE,DATA
from iteration104_validate_v1 import interval
from iteration104_packed_v1 import Packed,OUT,FIT


def run(out):
    out.mkdir(exist_ok=False,parents=True)
    cuda=Path(shutil.which('nvcc')).parent
    omp=next((Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Redist').glob('MSVC/*/debug_nonredist/x64/Microsoft.VC143.OpenMP.LLVM/libomp140.x86_64.dll'))
    handles=[os.add_dll_directory(str(p)) for p in (cuda,omp.parent,Path(shutil.which('g++')).parent)]
    files=[DATA/'iteration72_frugal_gpu_build_v5/frugal_fp32.dll',DATA/'iteration75_frugal_gpu_build_v3/frugal_fp64.dll']
    fr=[]
    for f in files:
        lib=ct.CDLL(str(f))
        fr.append(bind(lib,'fg72_sample',[I,I,I,ct.c_uint,I,D,B,D,ct.c_char_p,I]))
    native=Native(BASE/'spatial.dll')
    oracle=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    spec=dict(np.load(ORDER))
    gpu=Packed(OUT/'gpu.dll')
    table=dict(np.load(FIT/'tables.npz'))
    rows=[]
    accuracy=[]
    for panel,ports in enumerate((np.arange(8),np.arange(288,296),np.arange(144))):
        m=len(ports);batch=2048 if m==8 else 64
        q=physical(spec,ports)
        cov=2*q-np.eye(2*m)
        def source(method,k,n,seed):
            if method in ('optical','hardware'):
                records,_,_=gpu.draw(n,False,seed,rate=float(table[method+'_rate']))
                return records[:,ports].astype(np.uint8),[]
            ix=int(method=='frugal64')
            samples=[];stats=[]
            for j,v in enumerate(cov):
                x=np.empty((n,m),np.uint8);st=np.zeros(6)
                msg=ct.create_string_buffer(4096)
                rc=fr[ix](m,k,n,seed+j,16,v,x,st,msg,len(msg))
                if rc:raise RuntimeError(msg.value)
                assert st[-1]==(64 if ix else 32)
                samples.append(x);stats.append(st.tolist())
            return samples[0]|samples[1],stats
        cases=[('optical',0),('hardware',0)]+[(name,k) for name in ('frugal32','frugal64') for k in ((2,4) if m==8 else (2,))]
        times=[[] for _ in cases];phases=[[] for _ in cases]
        rng=np.random.default_rng(104090000+panel)
        reps=7 if m==8 else 4
        for rep in range(reps):
            for i in rng.permutation(len(cases)):
                name,k=cases[i]
                if k==0:
                    gpu.set_table(table[name])
                    import torch
                    torch.cuda.synchronize()
                tic=perf_counter()
                rec,st=source(name,k,batch,104091000+1000*panel+50*rep+2*int(i))
                us=(perf_counter()-tic)*1e6/batch
                assert np.logical_or(rec==0,rec==1).all()
                if rep:times[i].append(us);phases[i].append(st)
        for i,(name,k) in enumerate(cases):
            rows.append(dict(panel=panel,modes=m,method=name,K=k,batch=batch,us=float(np.median(times[i])),timing=times[i],upstream_phases=phases[i]))
        if m==8:
            truth=np.empty(256)
            assert oracle(8,2,q,np.zeros((2,16)),truth,np.empty(2))==0
            keys=(1<<np.arange(8)).astype(np.uint16)
            for i,(name,k) in enumerate(cases):
                if k==0:gpu.set_table(table[name])
                trial=np.zeros(256,np.int64);obs=trial.copy()
                for part,(hist,n) in enumerate(((trial,32768),(obs,262144))):
                    for off in range(0,n,8192):
                        x,_=source(name,k,8192,104100000+20000*panel+2000*i+500*part+2*(off//8192))
                        hist+=np.bincount(x@keys,minlength=256)
                bounds=interval(truth,obs,trial,int(obs.sum()),.0001)
                accuracy.append(dict(panel=panel,method=name,K=k,n=int(obs.sum()),**bounds))
                np.savez_compressed(out/f'panel_{panel}_{name}_{k}.npz',truth=truth,trial=trial,hold=obs)
        print(panel,'finished',flush=True)
    save(out,'summary.json',dict(timing=rows,accuracy=accuracy,ports=[list(range(8)),list(range(288,296)),list(range(144))],backend='Pinned official Frugal CUDA kernels; legacy archived MSVC host compatibility changes, 16 host workers. Each spectrum prepares and samples independently. Both complete host outputs are combined with OR. Our rates include whole-device host transfer and conversion to the same selected uint8 array.',dll_hashes={f.name+str(j):H(f) for j,f in enumerate(files)}))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):run(p.parse_args().output.resolve())
