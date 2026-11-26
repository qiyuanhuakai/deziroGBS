"""Evaluate the same coherent matching sampler on the physically occupied Jiuzhang spectra."""
import argparse
import ctypes as ct
from pathlib import Path
import shutil
import subprocess
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,save,manifest,I,D,bind
from iteration106_field_v1 import actual_input
from iteration106_sample_v1 import DATA,benchmark
from iteration107_fast_v1 import gaussian_click,oracle,ONE
from iteration107_cliques_v1 import optical_x
from iteration108_compact_v1 import Compact


def run(out):
    out.mkdir(exist_ok=False,parents=True)
    eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'
    subprocess.run([shutil.which('g++'),'-O3','-ffp-contract=off','-std=c++20','-march=x86-64-v3','-fopenmp','-shared','-I',str(eigen),str(ROOT/'native/matching_law108_v1.cpp'),'-o',str(out/'laws.dll')],check=True)
    t,r=actual_input();model=Compact(t,r,4,512);quad=oracle(True)
    fn=bind(ct.CDLL(str(out/'laws.dll')),'matching_law108',[I]*5+[D]*4)
    keep=np.array([1-(1-np.array(p['mean_acceptance']))**512 for p in model.cert['spectra']])
    panels=[np.arange(8),np.arange(288,296)];rows=[];arrays={}
    for j,ids in enumerate(panels):
        target=gaussian_click(quad,optical_x(t[ids],r),ONE);buf=np.empty((6,256))
        assert fn(8,25,2,4,12,np.ascontiguousarray(t[ids]).view(float),np.ascontiguousarray(np.tanh(r)),keep,buf)==0
        assert abs(buf[0]-target).sum()/2<1e-10
        rows.append(dict(ports=ids.tolist(),no_interference_tv=float(abs(buf[1]-target).sum()/2),model_tv=float(abs(buf[-1]-target).sum()/2)))
        arrays[f'panel{j}']=buf;arrays[f'target{j}']=target
    hw=np.load(DATA/'iteration104_record_order_v1/windows.npz')['histograms'][32:].sum(0)
    runs=[];batch=256;n=8192
    for pn in (False,True):
        ts=benchmark(lambda n,s,o:model.draw(n,pn,s,o),batch,108101000+pn,9)
        hist=np.zeros(1153,int);v=np.zeros((2,6,3),np.uint64)
        for off in range(0,n,batch):
            x,y=model.draw(batch,pn,108102000+pn,off,stats=True);v+=y;hist+=np.bincount(np.count_nonzero(x,axis=1),minlength=1153)
        assert np.all(v[:,:,0].sum(1)==n)
        runs.append(dict(pnr=pn,us=ts['us'],batch=batch,hardware_total_click_ks=float(abs(hist.cumsum()/n-hw.cumsum()/hw.sum()).max()),fallbacks=v[:,:,2].sum(1).tolist()))
        arrays[f'hw{int(pn)}']=hist
    np.savez_compressed(out/'actual.npz',**arrays)
    save(out,'summary.json',dict(panels=rows,shots=n,emitted_pairs_by_spectrum=(np.sinh(r)**2).sum(1).tolist(),runs=runs,certificate=model.cert));manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(p.parse_args().output.resolve())
