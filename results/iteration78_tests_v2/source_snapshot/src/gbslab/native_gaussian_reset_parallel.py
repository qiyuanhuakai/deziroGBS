"""C++ trajectory-parallel approximate PNR sampler with safeguarded Newton flow.

This retains the Gaussian-reset approximation and its measured distribution
bias. Worker count changes scheduling, not row seeds. Component time statistics
sum across workers and are not elapsed batch time. Use ``wall_seconds`` instead.
OpenBLAS, if supplied, runs with one internal thread and its setting is restored.
Concurrent calls that change that process-wide BLAS setting are unsupported.
"""
import ctypes as C
import time
import numpy as np
from .native_gaussian_reset import NativeGaussianResetSampler, DP, UP, SELECTORS, STAT_NAMES

PARALLEL_STAT_NAMES=STAT_NAMES+(
    'spectral_seconds','inversion_seconds','reconstruction_seconds',
    'survival_evaluations','maximum_root_iterations','maximum_log_root_residual',
    'newton_fallbacks','vacuum_log_seconds','maximum_bracket_width',
    'flow_calls','workers','completed_samples')


class NativeParallelGaussianResetSampler(NativeGaussianResetSampler):
    def __init__(self,*args,workers=1,**kwargs):
        self._validate_workers(workers)
        if kwargs.get('threads',1)!=1:
            raise ValueError('Internal BLAS threads must be one; use workers for trajectories')
        super().__init__(*args,**kwargs)
        try:
            self.workers=workers
            self.lib.g70_sample_parallel.argtypes=[C.c_void_p,C.c_int,C.c_uint64,
                C.c_int,C.c_int,C.c_void_p,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int]
            self.lib.g70_sample_parallel.restype=C.c_int
        except Exception:
            self.close()
            raise

    @staticmethod
    def _validate_workers(workers):
        if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=256:
            raise ValueError('workers must be an integer in [1,256]')

    def sample(self,count,seed,*,selector='diagonal',workers=None,flow='newton',backend='eigen'):
        if not self.handle:raise RuntimeError('Sampler is closed')
        if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=np.iinfo(np.int32).max:
            raise ValueError('Invalid sample count')
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:
            raise ValueError('seed must be a uint64 integer')
        workers=self.workers if workers is None else workers
        self._validate_workers(workers)
        if selector not in SELECTORS:raise ValueError('Invalid detector selector')
        if flow not in ('newton','bisection'):raise ValueError('Invalid flow solver')
        if backend not in ('eigen','lapack'):raise ValueError('Invalid spectral backend')
        if flow=='bisection' and backend!='eigen':raise ValueError('Bisection reference uses Eigen')
        if (selector=='openblas' or backend=='lapack') and self.blas is None:
            raise ValueError('Explicit ILP64 OpenBLAS library required')
        algorithm=0 if flow=='bisection' else (2 if backend=='lapack' else 1)
        eigen_function=C.cast(self.blas.scipy_LAPACKE_dsyevd_work64_,C.c_void_p) if backend=='lapack' else None
        output=np.zeros((count,self.modes),np.uint32);stats=np.zeros(20);error=C.create_string_buffer(4096)
        old=self.getter() if self.blas is not None else None
        start=time.perf_counter()
        try:
            if old is not None:self.setter(1)
            rc=self.lib.g70_sample_parallel(self.handle,count,seed,SELECTORS[selector],
                algorithm,eigen_function,workers,self.mean.ctypes.data_as(DP),
                self.noise_variance.ctypes.data_as(DP),output.ctypes.data_as(UP),
                stats.ctypes.data_as(DP),error,len(error))
        finally:
            if old is not None:self.setter(old)
        elapsed=time.perf_counter()-start
        if rc:raise ArithmeticError(error.value.decode())
        return {'counts':output,'statistics':dict(zip(PARALLEL_STAT_NAMES,stats.tolist())),
                'wall_seconds':elapsed}
