"""PNR Gaussian moment functional with retained photon-subtraction factors.

Analytic photon jumps remove the finite-tap approximation. After retained+1
jumps, exact first/second moments are compressed to a Gaussian, introducing
an approximation. No finite tap, photon cutoff or silent trajectory rejection.
"""
import ctypes as C
import time
import numpy as np
from .native_direct_click import NativeDirectClickSampler, DP

UP=C.POINTER(C.c_uint32)
STATS=('photons','completed_samples','maximum_photons','compressions',
       'maximum_photon_factors','flow_calls','root_iterations',
       'bisection_fallback_roots','maximum_survival_residual')


class NativePolynomialPNRSampler(NativeDirectClickSampler):
    def __init__(self,*args,retained=1,**kwargs):
        if isinstance(retained,bool) or not isinstance(retained,int) or not 0<=retained<=3:
            raise ValueError('retained must be an integer in [0,3]')
        super().__init__(*args,**kwargs);self.retained=retained
        self.lib.pnr74_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,
            DP,DP,UP,DP,C.c_char_p,C.c_int];self.lib.pnr74_sample.restype=C.c_int

    def sample(self,count,seed,*,workers=None):
        if not self.handle:raise RuntimeError('Sampler is closed')
        if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=np.iinfo(np.int32).max:
            raise ValueError('Invalid sample count')
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:
            raise ValueError('seed must be a uint64 integer')
        workers=self.workers if workers is None else workers;self._validate_workers(workers)
        start=time.perf_counter();counts=np.zeros((count,self.modes),np.uint32);stats=np.zeros(9);error=C.create_string_buffer(4096)
        rc=self.lib.pnr74_sample(self.handle,count,seed,workers,self.retained,
            self.mean.ctypes.data_as(DP),self.noise_variance.ctypes.data_as(DP),
            counts.ctypes.data_as(UP),stats.ctypes.data_as(DP),error,len(error))
        if rc:raise ArithmeticError(error.value.decode())
        return dict(counts=counts,statistics=dict(zip(STATS,stats.tolist())),wall_seconds=time.perf_counter()-start)
