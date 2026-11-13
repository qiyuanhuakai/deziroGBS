"""Explicit product-pure squeezing core with stable photon-factor norms.

Displacement mixtures represent mixed inputs. After compression, positive-P Gaussian states finish by coherent-mixture
Poisson sampling; other states use the general Gaussian moment closure. This does not
preserve global parity after compression and is not an exact GBS sampler.
"""
import ctypes as C
import time
import numpy as np
from .native_polynomial_pnr import NativePolynomialPNRSampler,DP,UP,STATS

class NativeClassicalTailPNRSampler(NativePolynomialPNRSampler):
    def __init__(self,v,tanh_squeezing,library,**kwargs):
        t=np.asarray(tanh_squeezing,dtype=np.float64)
        if t.ndim!=1 or not np.isfinite(t).all() or np.any(abs(t)>=1):
            raise ValueError('tanh_squeezing must be a finite vector inside (-1,1)')
        self.squeezing=np.ascontiguousarray(t).copy()
        x=np.diag(np.r_[-t/(1+t),t/(1-t)])
        super().__init__(v,x,library,**kwargs)
        if len(t)!=self.inputs:raise ValueError('Squeezing and interferometer dimensions differ')
        self.lib.pnrc74_sample.argtypes=[C.c_void_p,DP,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int]
        self.lib.pnrc74_sample.restype=C.c_int

    def sample(self,count,seed,*,workers=None):
        if not self.handle:raise RuntimeError('Sampler is closed')
        if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=np.iinfo(np.int32).max:raise ValueError('Invalid sample count')
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:raise ValueError('seed must be uint64')
        workers=self.workers if workers is None else workers;self._validate_workers(workers)
        start=time.perf_counter();counts=np.zeros((count,self.modes),np.uint32);stats=np.zeros(12);error=C.create_string_buffer(4096)
        rc=self.lib.pnrc74_sample(self.handle,self.squeezing.ctypes.data_as(DP),count,seed,workers,self.retained,self.mean.ctypes.data_as(DP),self.noise_variance.ctypes.data_as(DP),counts.ctypes.data_as(UP),stats.ctypes.data_as(DP),error,len(error))
        if rc:raise ArithmeticError(error.value.decode())
        return dict(counts=counts,statistics=dict(zip(STATS+('classical_completions','classical_photons','classicality_checks'),stats.tolist())),wall_seconds=time.perf_counter()-start)
