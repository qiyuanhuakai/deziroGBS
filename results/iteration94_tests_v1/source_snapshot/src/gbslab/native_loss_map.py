"""Native passive-loss reduction to an isometry and active excess covariance.

Input and output covariance convention is X=(V-I)/2, with grouped q,p axes.
Reduction preserves the Gaussian state in real arithmetic; downstream finite
branch samplers remain approximate. This routine does not clip eigenvalues.
"""
import ctypes as C
import os
from pathlib import Path
import time
import numpy as np

def prepare_lossy_gaussian_map(transmission, excess_covariances, library, *, runtime_dirs=()):
    t=np.asarray(transmission)
    x=np.asarray(excess_covariances)
    if t.ndim!=2 or not 1<=t.shape[1]<=128 or not t.shape[1]<=t.shape[0]<=65536:
        raise ValueError('Loss map requires 1..128 inputs and inputs..65536 outputs')
    if np.iscomplexobj(x):raise ValueError('Excess covariances must be real')
    if x.ndim==2:x=x[None,...]
    n=t.shape[1]
    if x.ndim!=3 or not 1<=len(x)<=8 or x.shape[1:]!=(2*n,2*n):
        raise ValueError('Supply 1..8 covariances with grouped q,p axes')
    t=np.ascontiguousarray(t,np.complex128);x=np.ascontiguousarray(x,np.float64)
    if not np.isfinite(t).all() or not np.isfinite(x).all():raise ValueError('Nonfinite loss input')
    runtime=[]
    try:
        if os.name=='nt':runtime=[os.add_dll_directory(str(Path(p).resolve())) for p in runtime_dirs]
        lib=C.CDLL(str(Path(library).resolve()));dp=C.POINTER(C.c_double)
        fn=lib.loss83_prepare;fn.argtypes=[C.c_int,C.c_int,C.c_int,dp,dp,dp,dp,dp,C.c_char_p,C.c_int];fn.restype=C.c_int
        v=np.zeros_like(t);y=np.zeros_like(x);stats=np.zeros(5);error=C.create_string_buffer(4096)
        ptr=lambda a:a.ctypes.data_as(dp)
        start=time.perf_counter();rc=fn(len(t),n,len(x),ptr(t),ptr(x),ptr(v),ptr(y),ptr(stats),error,len(error));elapsed=time.perf_counter()-start
        if rc:raise ValueError(error.value.decode())
        return dict(isometry=v,excess_covariances=y,wall_seconds=elapsed,statistics=dict(zip(('isometry_residual','relative_reconstruction_residual','maximum_transmission_squared_singular_value','minimum_uncertainty_eigenvalue','pre_symmetrization_residual'),stats.tolist())))
    finally:
        for handle in runtime:handle.close()
