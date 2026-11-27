"""Independently recover joint optical words and counts by real conditional covariances."""
import argparse
import ctypes as ct
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,save,manifest
from iteration106_field_v1 import actual_input,DATA
from iteration109_model_v1 import build
from iteration110_spectrum_v1 import Spectrum
from iteration110_joint_v1 import ANCHOR,optical_parts,model_parts


def recover(lib,parts,bits,size):
    fn=lib.conditioned110;fn.argtypes=[ct.c_int]*4+[ct.c_void_p]*3
    def one(e,head):
        probs=np.empty((1<<bits,size));e=np.ascontiguousarray(e);head=np.ascontiguousarray(head)
        rc=fn(bits,len(e),size,12,e.ctypes.data,head.ctypes.data,probs.ctypes.data)
        if rc:raise ArithmeticError(f'Real conditional covariance errors: {rc}')
        return probs
    first,*rest=[one(e,h) for e,h in parts]
    for z in rest:first=np.array([np.convolve(a,b)[:size] for a,b in zip(first,z)])
    words=first[::-1].astype(np.longdouble)
    for i in range(bits):
        h=1<<i;reshaped=words.reshape(-1,2*h,size);reshaped[:,h:]-=reshaped[:,:h]
    return np.asarray(words.T,float)


def run(out):
    if out.exists() and any(p.name!='conditioned.dll' for p in out.iterdir()):raise FileExistsError(out)
    out.mkdir(exist_ok=True,parents=True);compiler=shutil.which('g++')
    incl=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'
    subprocess.run([compiler,'-O3','-ffp-contract=off','-std=c++20','-DEIGEN_DONT_PARALLELIZE','-DEIGEN_DONT_VECTORIZE','-mstackrealign','-fopenmp','-shared','-I',str(incl),
                    str(ROOT/'native/conditioned_number110_v1.cpp'),'-o',str(out/'conditioned.dll')],check=True)
    rt=os.add_dll_directory(str(Path(compiler).parent)) if os.name=='nt' else None
    lib=ct.CDLL(str(out/'conditioned.dll'));t,r=actual_input();m=build(t,r,8,1)
    z=np.load(DATA/'iteration110_joint_v1/tables.npz');best=Spectrum(m).materialize(z['coeff']);rows=[];arrays={}
    for name,parts in [('target',optical_parts(t,r,ANCHOR)),('base_q',model_parts(m,ANCHOR)),('fitted_q',model_parts(best,ANCHOR))]:
        truth=recover(lib,parts,len(ANCHOR),1024);gap=float(abs(truth-z[name]).sum()/2)
        assert gap<1e-9 and max(abs(truth.sum()-1),-truth.min())<1e-11
        arrays[name]=truth;rows.append(dict(name=name,joint_tv_vs_fourier=gap,max_cell_error=float(abs(truth-z[name]).max())))
    np.savez_compressed(out/'laws.npz',**arrays);save(out,'summary.json',dict(checks=rows));manifest(out);print(rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(ap.parse_args().output.resolve())
