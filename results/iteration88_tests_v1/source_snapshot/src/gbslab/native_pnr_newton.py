"""PNR finite-tap signed branches with safeguarded Newton survival inversion.

The flow uses bracketed Newton steps and an explicit bisection fallback.
Residual diagnostics describe arithmetic stopping, not a global error bound.

Finite tap strength introduces unobserved attenuation and multi-photon herald
bias; merging branches discards non-Gaussian information. This is approximate,
including with a very large branch budget. No photon cutoff is applied.
"""
import ctypes as C
import time
import numpy as np
from .native_direct_click import NativeDirectClickSampler, DP
from .native_click_branches import NativeBranchClickSampler

UP=C.POINTER(C.c_uint32)
STATS=('photons','completed_samples','maximum_photons','peak_transient_components',
       'maximum_absolute_weight_sum','compressions','component_evaluations','survival_flow_calls','root_iterations',
       'bisection_fallback_roots','maximum_survival_residual')


class NativeNewtonPNRSampler(NativeDirectClickSampler):
    def __init__(self,*args,branch_budget=2,tap_strength=.001,condition_trigger=1e6,**kwargs):
        NativeBranchClickSampler._validate_budget(branch_budget)
        if isinstance(tap_strength,bool) or not np.isfinite(tap_strength) or not 0<tap_strength<1:
            raise ValueError('tap_strength must be finite and in (0,1)')
        if isinstance(condition_trigger,bool) or not np.isfinite(condition_trigger) or condition_trigger<1:
            raise ValueError('condition_trigger must be finite and at least one')
        super().__init__(*args,**kwargs)
        self.branch_budget=branch_budget;self.tap_strength=float(tap_strength);self.condition_trigger=float(condition_trigger)
        self.lib.pnr73_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,C.c_double,C.c_double,
                                      DP,DP,UP,DP,C.c_char_p,C.c_int]
        self.lib.pnr73_sample.restype=C.c_int

    def sample(self,count,seed,*,workers=None):
        if not self.handle:raise RuntimeError('Sampler is closed')
        if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=np.iinfo(np.int32).max:
            raise ValueError('Invalid sample count')
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:
            raise ValueError('seed must be a uint64 integer')
        workers=self.workers if workers is None else workers;self._validate_workers(workers)
        counts=np.zeros((count,self.modes),np.uint32);stats=np.zeros(11);error=C.create_string_buffer(4096)
        start=time.perf_counter();rc=self.lib.pnr73_sample(self.handle,count,seed,workers,self.branch_budget,
            self.tap_strength,self.condition_trigger,self.mean.ctypes.data_as(DP),self.noise_variance.ctypes.data_as(DP),
            counts.ctypes.data_as(UP),stats.ctypes.data_as(DP),error,len(error))
        elapsed=time.perf_counter()-start
        if rc:raise ArithmeticError(error.value.decode())
        return dict(counts=counts,statistics=dict(zip(STATS,stats.tolist())),wall_seconds=elapsed)
