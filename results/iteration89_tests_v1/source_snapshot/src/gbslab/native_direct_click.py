"""Direct threshold sampling with approximate Gaussian conditional moments.

This C++ route produces only binary clicks. No-click conditioning is Gaussian
and exact in real arithmetic; a click is followed by moment closure, which
discards non-Gaussian information. This is not an exact threshold GBS sampler.
All per-mode conditioning, Gaussian noise and random decisions execute in C++.
"""
import ctypes as C
import os
from pathlib import Path
import time
import numpy as np
from .native_gaussian_reset import DP, NativeGaussianResetSampler

BP=C.POINTER(C.c_uint8)
STAT_NAMES=('clicks','detector_steps','zero_click_samples','workers','completed_samples')


class NativeDirectClickSampler:
    def __init__(self,isometry,excess_covariance,library,*,mean=None,
                 noise_variance=None,workers=1,runtime_dirs=()):
        self.handle=None
        self._validate_workers(workers)
        if np.iscomplexobj(excess_covariance):raise ValueError('Covariance must be real')
        v=np.ascontiguousarray(isometry,dtype=np.complex128)
        x=np.ascontiguousarray(excess_covariance,dtype=np.float64)
        if v.ndim!=2 or not 1<=v.shape[1]<=v.shape[0] or max(v.shape)>np.iinfo(np.int32).max//2:
            raise ValueError('Invalid isometry dimensions')
        self.modes,self.inputs=v.shape;d=2*self.inputs
        if x.shape!=(d,d) or not np.isfinite(x).all() or not np.isfinite(v).all():
            raise ValueError('Invalid covariance or isometry')
        self.mean=NativeGaussianResetSampler._vector(mean,d,'mean')
        self.noise_variance=NativeGaussianResetSampler._vector(noise_variance,d,'noise variance')
        if np.any(self.noise_variance<0):raise ValueError('Negative noise variance')
        self.workers=workers
        self.runtime=[os.add_dll_directory(str(Path(p).resolve())) for p in runtime_dirs] if os.name=='nt' else []
        self.lib=C.CDLL(str(Path(library).resolve()))
        self.lib.tc70_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int]
        self.lib.tc70_create.restype=C.c_void_p
        self.lib.tc70_destroy.argtypes=[C.c_void_p];self.lib.tc70_destroy.restype=None
        self.lib.tc70_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,DP,DP,BP,DP,C.c_char_p,C.c_int]
        self.lib.tc70_sample.restype=C.c_int
        error=C.create_string_buffer(4096);start=time.perf_counter()
        self.handle=self.lib.tc70_create(self.modes,self.inputs,v.ctypes.data_as(DP),x.ctypes.data_as(DP),error,len(error))
        self.model_seconds=time.perf_counter()-start
        if not self.handle:raise ValueError(error.value.decode())

    @staticmethod
    def _validate_workers(workers):
        if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=256:
            raise ValueError('workers must be an integer in [1,256]')

    def sample(self,count,seed,*,workers=None):
        if not self.handle:raise RuntimeError('Sampler is closed')
        if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=np.iinfo(np.int32).max:
            raise ValueError('Invalid sample count')
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:
            raise ValueError('seed must be a uint64 integer')
        workers=self.workers if workers is None else workers;self._validate_workers(workers)
        clicks=np.zeros((count,self.modes),np.uint8);stats=np.zeros(5);error=C.create_string_buffer(4096)
        start=time.perf_counter();rc=self.lib.tc70_sample(self.handle,count,seed,workers,
            self.mean.ctypes.data_as(DP),self.noise_variance.ctypes.data_as(DP),
            clicks.ctypes.data_as(BP),stats.ctypes.data_as(DP),error,len(error));elapsed=time.perf_counter()-start
        if rc:raise ArithmeticError(error.value.decode())
        return {'clicks':clicks,'statistics':dict(zip(STAT_NAMES,stats.tolist())),'wall_seconds':elapsed}

    def close(self):
        if self.handle:self.lib.tc70_destroy(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
    def __del__(self):
        if getattr(self,'handle',None):self.close()
