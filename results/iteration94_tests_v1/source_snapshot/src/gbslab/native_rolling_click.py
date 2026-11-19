"""Direct clicks retaining a rolling window of inclusion-exclusion branches.

With sufficient branches this follows the known exact Gaussian-difference
algorithm in real arithmetic. A finite budget triggers approximate moment
compression; numerical cancellation can also cause an explicit batch failure.
Budget one uses iteration 70's moment closure. This class generates no PNRs.
"""
import ctypes as C
import time
import numpy as np
from .native_direct_click import NativeDirectClickSampler, DP, BP

STATS=('clicks','detector_steps','zero_click_samples','workers','completed_samples',
       'component_evaluations','compressions','peak_transient_components',
       'maximum_absolute_weight_sum')


class NativeRollingClickSampler(NativeDirectClickSampler):
    def __init__(self,*args,branch_budget=4,**kwargs):
        self._validate_budget(branch_budget)
        super().__init__(*args,**kwargs)
        self.branch_budget=branch_budget
        self.lib.rc80_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,
                                      DP,DP,BP,DP,C.c_char_p,C.c_int]
        self.lib.rc80_sample.restype=C.c_int
        self.lib.rc80_prefix.argtypes=[C.c_void_p,DP,C.c_int,BP,C.c_int,DP,DP,
                                      C.c_char_p,C.c_int]
        self.lib.rc80_prefix.restype=C.c_int

    @staticmethod
    def _validate_budget(value):
        if isinstance(value,bool) or not isinstance(value,int) or not 1<=value<=128 or value&(value-1):
            raise ValueError('branch_budget must be a power of two in [1,128]')

    def sample(self,count,seed,*,workers=None,branch_budget=None):
        if not self.handle:raise RuntimeError('Sampler is closed')
        if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=np.iinfo(np.int32).max:
            raise ValueError('Invalid sample count')
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:
            raise ValueError('seed must be a uint64 integer')
        workers=self.workers if workers is None else workers;self._validate_workers(workers)
        budget=self.branch_budget if branch_budget is None else branch_budget;self._validate_budget(budget)
        clicks=np.zeros((count,self.modes),np.uint8);stats=np.zeros(9);error=C.create_string_buffer(4096)
        start=time.perf_counter();rc=self.lib.rc80_sample(self.handle,count,seed,workers,budget,
            self.mean.ctypes.data_as(DP),self.noise_variance.ctypes.data_as(DP),
            clicks.ctypes.data_as(BP),stats.ctypes.data_as(DP),error,len(error))
        elapsed=time.perf_counter()-start
        if rc:raise ArithmeticError(error.value.decode())
        return dict(clicks=clicks,statistics=dict(zip(STATS,stats.tolist())),wall_seconds=elapsed)

    def prefix_probability(self,bits,*,mean=None,branch_budget=None):
        """Fixed-displacement prefix law; random noise is not integrated here.

        If this model has noise, pass an explicit realized mean. Otherwise the
        method refuses to present a conditional law as the averaged input law.
        """
        if not self.handle:raise RuntimeError('Sampler is closed')
        values=np.asarray(bits)
        if values.ndim!=1 or len(values)>self.modes or not np.isin(values,[0,1]).all():
            raise ValueError('bits must be a binary prefix no longer than modes')
        if mean is None and self.noise_variance.any():
            raise ValueError('An explicit realized mean is required for a noisy model')
        w=self.mean if mean is None else self._mean_vector(mean)
        budget=self.branch_budget if branch_budget is None else branch_budget;self._validate_budget(budget)
        bits=np.ascontiguousarray(values,dtype=np.uint8);p=np.zeros(1);stats=np.zeros(4);error=C.create_string_buffer(4096)
        rc=self.lib.rc80_prefix(self.handle,w.ctypes.data_as(DP),len(bits),bits.ctypes.data_as(BP),budget,
                              p.ctypes.data_as(DP),stats.ctypes.data_as(DP),error,len(error))
        if rc:raise ArithmeticError(error.value.decode())
        return dict(probability=float(p[0]),component_evaluations=float(stats[0]),
                    compressions=float(stats[1]),peak_transient_components=float(stats[2]),
                    maximum_absolute_weight_sum=float(stats[3]))

    def _mean_vector(self,value):
        from .native_gaussian_reset import NativeGaussianResetSampler
        return NativeGaussianResetSampler._vector(value,2*self.inputs,'realized mean')
