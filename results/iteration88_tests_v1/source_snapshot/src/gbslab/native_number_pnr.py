"""Equal-squeezing, undisplaced pure PNR conditioned on an exact total draw.
No Gaussian closure or total-count resampling; resource failures are explicit.
"""
import ctypes as C
import time
import numpy as np
from .native_structured_pnr import NativeStructuredPNRSampler
from .native_polynomial_pnr import DP,UP
STATS77=('photons','completed_samples','maximum_photons','total_draws','detector_steps','maximum_factors','maximum_terms','maximum_rank')
class NativeNumberPNRSampler(NativeStructuredPNRSampler):
    def __init__(self,v,squeezing,library,**kwargs):
        t=np.asarray(squeezing,dtype=np.float64)
        if t.ndim==0:t=np.full(np.asarray(v).shape[1],float(t))
        if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or np.any(t<0) or np.any(t>=1) or not np.all(t==t[0]):raise ValueError('Number sampler requires equal squeezing in [0,1)')
        super().__init__(v,t,library,retained=0,**kwargs)
        if np.any(self.mean) or np.any(self.noise_variance):self.close();raise ValueError('Number sampler requires zero displacement and no noise')
        self.lib.number77_sample.argtypes=[C.c_void_p,C.c_double,C.c_int,C.c_uint64,C.c_int,C.c_int,UP,C.POINTER(C.c_uint64),DP,C.c_char_p,C.c_int];self.lib.number77_sample.restype=C.c_int
    def sample(self,count,seed,*,workers=None,fixed_total=None):
        if not self.handle:raise RuntimeError('Sampler is closed')
        if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=np.iinfo(np.int32).max:raise ValueError('Invalid sample count')
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:raise ValueError('seed must be uint64')
        if fixed_total is not None and (isinstance(fixed_total,bool) or not isinstance(fixed_total,int) or fixed_total<0 or fixed_total%2 or fixed_total>np.iinfo(np.int32).max):raise ValueError('fixed_total must be a nonnegative even integer')
        if np.any(self.mean) or np.any(self.noise_variance):raise ValueError('Number sampler requires zero displacement and no noise')
        if not np.isfinite(self.squeezing).all() or not np.all(self.squeezing==self.squeezing[0]):raise ValueError('Number sampler requires equal finite squeezing')
        workers=self.workers if workers is None else workers;self._validate_workers(workers);start=time.perf_counter();counts=np.zeros((count,self.modes),np.uint32);totals=np.zeros(count,np.uint64);stats=np.zeros(8);error=C.create_string_buffer(4096)
        rc=self.lib.number77_sample(self.handle,float(self.squeezing[0]),count,seed,workers,-1 if fixed_total is None else fixed_total,counts.ctypes.data_as(UP),totals.ctypes.data_as(C.POINTER(C.c_uint64)),stats.ctypes.data_as(DP),error,len(error))
        if rc:raise ArithmeticError(error.value.decode())
        return dict(counts=counts,total_photons=totals,statistics=dict(zip(STATS77,stats.tolist())),wall_seconds=time.perf_counter()-start)
