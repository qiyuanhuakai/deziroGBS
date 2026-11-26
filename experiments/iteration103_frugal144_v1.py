"""Measure a 144-detector Frugal K2 reduction, including the same optical prefix event on both implementations."""
import argparse
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_projection_v1 import physical
from iteration100_spatial_v1 import ORDER,I,D,B,Native,bind,save,manifest
from iteration103_sample_v1 import DATA,BASE,RUN,Sampler
from iteration103_transport_gpu_v1 import Cuda,Correction,OUT,torch


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    keep=Native(BASE/'spatial.dll')
    f=bind(ct.CDLL(str(DATA/'iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll')),'fc73_sample',[I,I,I,ct.c_uint,I,D,B,D,ct.c_char_p,I])
    spec=dict(np.load(ORDER))
    q=2*physical(spec,np.arange(144))-np.eye(288)
    n=64
    times=[]
    spectra=[]
    for i in range(3):
        x=[]
        phases=[]
        t=perf_counter()
        for j,v in enumerate(q):
            bits=np.empty((n,144),np.uint8)
            st=np.zeros(7)
            msg=ct.create_string_buffer(2048)
            rc=f(144,2,n,103111000+2*i+j,1,v,bits,st,msg,len(msg))
            if rc:raise RuntimeError(msg.value)
            x.append(bits)
            phases.append(st.tolist())
        samples=np.bitwise_or.reduce(x)
        times.append((perf_counter()-t)*1e6/n)
        spectra.append(phases)
    base=Sampler(RUN/'draw.dll',RUN/'bridges.npz',False)
    gpu=Cuda(OUT/'gpu.dll',base,False)
    correction=Correction(OUT/'gpu.dll',base)
    torch.set_float32_matmul_precision('highest')
    gt=[]
    for i in range(12):
        t=perf_counter()
        full,_=gpu.sample(n,False,103112000,n*i,correction)
        prefix=full[:,:144].copy()
        if i>=2:gt.append((perf_counter()-t)*1e6/n)
    # Causal prefix, using the actual first 16 physical detectors of both circuits.
    rr=json.loads((DATA/'iteration103_baselines_v1/summary.json').read_text())['click'][-1]
    assert rr['groups']==[0,1]
    pr=json.loads((DATA/'iteration103_projection_v1/summary.json').read_text())['rows'][0]
    assert pr['groups']==[0,1]
    np.savez_compressed(out/'draws.npz',frugal=samples,gpu=prefix)
    save(out,'summary.json',dict(modes=144,records_per_call=n,threads=1,K=2,frugal_us=float(np.median(times)),times=times,frugal_phases=spectra,gpu_full_then_slice_us=float(np.median(gt)),gpu_times=gt,model_raw_prefix_tv=rr['complete_tv'],model_patched_prefix_tv_interval=pr['patched_click_interval'],frugal_prefix_tv=rr['frugal'][0]['complete_tv']))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):main(p.parse_args().output.resolve())
