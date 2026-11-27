"""Covariance of true photons, fired bins and unobserved collisions on the Jiuzhang 3 transfer."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration106_field_v1 import actual_input
from iteration107_cliques_v1 import optical_x
from iteration114_rank_v1 import ROOT

def run(out):
    out.mkdir(exist_ok=True,parents=True)
    dll=out/('saturation.dll' if os.name=='nt' else 'saturation.so')
    subprocess.run(['g++','-O3','-fopenmp','-shared','-fPIC',str(ROOT/'native/saturation114_v1.cpp'),'-o',str(dll)],check=True)
    import shutil
    runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent)) if os.name=='nt' else None
    lib=ct.CDLL(str(dll.resolve()))
    lib.saturation114.argtypes=[ct.c_int]*3+[ct.c_void_p]*3
    t,r=actual_input();x=np.ascontiguousarray(optical_x(t,r));m=len(t);g=144
    mean=np.empty((2,m));c=np.empty((3,g,g))
    start=perf_counter()
    assert lib.saturation114(m,len(r),8,*[a.ctypes.data for a in (x,mean,c)])==0
    elapsed=perf_counter()-start
    var_n,var_h,cross=c.sum((1,2));var_d=var_n+var_h-2*cross
    old=np.load(ROOT/'results/iteration113_device_v1/banks.npz')['number_bins']
    h,n=np.indices(old.shape[1:])
    en=(old*n).sum((1,2));eh=(old*h).sum((1,2))
    independently=np.array([(old*n**2).sum((1,2))-en**2,(old*h**2).sum((1,2))-eh**2,
                            (old*n*h).sum((1,2))-en*eh])
    diag_error=float(abs(independently-np.diagonal(c,axis1=1,axis2=2)).max())
    assert diag_error<1e-8 and abs(var_n-(mean[0].sum()+.5*(x*x).sum()))<1e-8
    assert np.linalg.eigvalsh(np.block([[c[0],c[2]],[c[2].T,c[1]]])).min()>-1e-8
    # Linear least-squares estimate of total arriving photons given all bank counts.
    nh=c[2].sum(0)
    weights=np.linalg.solve(c[1],nh)
    resolved=float(nh@weights)
    bank_rms=float(np.sqrt(var_n-resolved))
    intercept=float(mean[0].sum()-weights@mean[1].reshape(g,8).sum(1))
    out_arrays=dict(mean=mean,covariance=c,bank_weights=weights)
    assert bank_rms<=np.sqrt(var_n-cross**2/var_h)+1e-10
    files=json.loads((ROOT/'results/iteration112_hardware_v1/sources.json').read_text())
    root=ROOT.parent/'external_sources/jiuzhang3_iteration112_independent'
    pop=np.array([i.bit_count() for i in range(256)],np.uint8)
    records=[]
    half=np.triu_indices(g,1)
    for f in files:
        path=root/f['member']
        with path.open('rb') as reader:assert hashlib.file_digest(reader,'sha256').hexdigest()==f['sha256']
        raw=np.memmap(path,np.uint8,'r').reshape(-1,g)
        assert len(raw)==f['rows']
        sums=np.zeros(g);second=np.zeros((g,g))
        for off in range(0,len(raw),8192):
            a=pop[raw[off:off+8192]].astype(float)
            sums+=a.sum(0);second+=a.T@a
        observed_mean=sums/len(raw)
        observed_cov=second/len(raw)-np.outer(observed_mean,observed_mean)
        diff=(observed_cov-c[1])[half]
        records.append(dict(file=f['file'],rows=len(raw),mean=observed_mean.sum(),variance=observed_cov.sum(),
                            std=float(np.sqrt(observed_cov.sum())),linear_predicted_mean=float(intercept+weights@observed_mean),pair_cov_mae=float(abs(diff).mean()),
                            pair_cov_rms=float(np.sqrt((diff**2).mean())),pair_cov_max=float(abs(diff).max())))
        out_arrays[f'cov{f["file"]}']=observed_cov;out_arrays[f'mean{f["file"]}']=observed_mean
    hw=(out_arrays['cov10']-out_arrays['cov11'])[half]
    summary=dict(reference_seconds=elapsed,model_mean_photons=mean[0].sum(),model_mean_fired=mean[1].sum(),
                 var_photons=var_n,var_fired=var_h,cov_photons_fired=cross,var_merged=var_d,
                 std_merged=float(np.sqrt(var_d)),correlation=cross/np.sqrt(var_h*var_n),
                 bank_linear_residual_rms=bank_rms,bank_linear_intercept=intercept,bank_linear_weight_range=[weights.min(),weights.max()],
                 linear_slope=cross/var_h,linear_residual_rms=float(np.sqrt(var_n-cross**2/var_h)),
                 diagonal_independent_max_abs=diag_error,
                 hardware_pair_cov_mae=float(abs(hw).mean()),hardware_pair_cov_rms=float(np.sqrt((hw*hw).mean())),
                 pairs=len(hw),hardware=records)
    np.savez_compressed(out/'covariances.npz',**out_arrays)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=4):run(p.parse_args().output)
