"""Exact pure prefix up to first compression; later closure remains approximate.
Retention is not an output photon cutoff. Sparse polynomial workspace is bounded.
"""
import ctypes as C
import time
import numpy as np
from .native_structured_pnr import NativeStructuredPNRSampler
from .native_polynomial_pnr import DP,UP,STATS
class NativePrefixPNRSampler(NativeStructuredPNRSampler):
    def __init__(self,*args,retained=15,**kwargs):
        if isinstance(retained,bool) or not isinstance(retained,int) or not 0<=retained<=31:raise ValueError('retained must be an integer in [0,31]')
        super().__init__(*args,retained=0,**kwargs);self.retained=retained
        self.lib.pnrp75_sample.argtypes=[C.c_void_p,DP,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int]
        self.lib.pnrp75_sample.restype=C.c_int
    def sample(self,count,seed,*,workers=None):
        if not self.handle:raise RuntimeError('Sampler is closed')
        if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=np.iinfo(np.int32).max:raise ValueError('Invalid sample count')
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:raise ValueError('seed must be uint64')
        workers=self.workers if workers is None else workers;self._validate_workers(workers)
        start=time.perf_counter();counts=np.zeros((count,self.modes),np.uint32);stats=np.zeros(9);error=C.create_string_buffer(4096)
        rc=self.lib.pnrp75_sample(self.handle,self.squeezing.ctypes.data_as(DP),count,seed,workers,self.retained,self.mean.ctypes.data_as(DP),self.noise_variance.ctypes.data_as(DP),counts.ctypes.data_as(UP),stats.ctypes.data_as(DP),error,len(error))
        if rc:raise ArithmeticError(error.value.decode())
        return dict(counts=counts,statistics=dict(zip(STATS,stats.tolist())),wall_seconds=time.perf_counter()-start)
