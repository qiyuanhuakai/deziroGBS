"""FP64 CUDA threshold trajectories with a bounded signed Gaussian expansion.

Uses the same approximate branch compression as NativeBranchClickSampler.
Explicitly supplied libraries; no compilation or GPU initialization on import.
Random displacement and detector draws use the native CPU replay library.
Conditional probabilities are those along each realized trajectory, not
noise-averaged prefix probabilities. Allocation and transfer costs are included.
"""
import ctypes as C
from pathlib import Path
import time
import numpy as np
from .native_direct_click import NativeDirectClickSampler, DP, BP

IP=C.POINTER(C.c_int32)


class NativeCudaBranchClickSampler(NativeDirectClickSampler):
    def __init__(self,isometry,excess_covariance,library,*,gpu_library,
                 branch_budget=4,**kwargs):
        self._validate_budget(branch_budget)
        v=np.array(isometry,dtype=np.complex128,order='C',copy=True)
        if np.iscomplexobj(excess_covariance):raise ValueError('Covariance must be real')
        x=np.array(excess_covariance,dtype=np.float64,order='C',copy=True)
        if v.ndim!=2 or not 1<=v.shape[1]<=128 or not 1<=v.shape[0]<=65536:
            raise ValueError('CUDA supports 1..128 inputs and 1..65536 outputs')
        super().__init__(v,x,library,**kwargs)
        self.branch_budget=branch_budget
        self._x=x;self._a=np.ascontiguousarray(np.c_[v.real,-v.imag])
        self._b=np.ascontiguousarray(np.c_[v.imag,v.real])
        for array in (self._x,self._a,self._b):array.flags.writeable=False
        self.lib.tc72_draws.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint64,DP,DP,DP,DP,C.c_char_p,C.c_int]
        self.lib.tc72_draws.restype=C.c_int
        self.gpu=C.CDLL(str(Path(gpu_library).resolve()))
        self.gpu.gc72_sample.argtypes=[C.c_int,C.c_int,C.c_int,C.c_int,DP,DP,DP,DP,DP,BP,DP,IP,DP,DP,C.c_char_p,C.c_int]
        self.gpu.gc72_sample.restype=C.c_int

    @staticmethod
    def _validate_budget(value):
        if isinstance(value,bool) or not isinstance(value,int) or not 1<=value<=128:
            raise ValueError('branch_budget must be an integer in [1,128]')

    def sample(self,count,seed,*,branch_budget=None):
        if not self.handle:raise RuntimeError('Sampler is closed')
        if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=1048576:
            raise ValueError('count must be an integer in [0,1048576]')
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:
            raise ValueError('seed must be a uint64 integer')
        budget=self.branch_budget if branch_budget is None else branch_budget
        self._validate_budget(budget)
        start=time.perf_counter()
        means=np.zeros((count,2*self.inputs));draws=np.zeros((count,self.modes))
        error=C.create_string_buffer(4096)
        rc=self.lib.tc72_draws(self.modes,2*self.inputs,count,seed,
            self.mean.ctypes.data_as(DP),self.noise_variance.ctypes.data_as(DP),
            means.ctypes.data_as(DP),draws.ctypes.data_as(DP),error,len(error))
        if rc:raise ArithmeticError(error.value.decode())
        random_seconds=time.perf_counter()-start
        result=self.replay(means,draws,branch_budget=budget)
        result['random_seconds']=random_seconds
        result['wall_seconds']=time.perf_counter()-start
        return result

    def replay(self,means,draws,*,branch_budget=None):
        """Sample with explicit displacements and uniforms; no marginal-law claim."""
        if not self.handle:raise RuntimeError('Sampler is closed')
        budget=self.branch_budget if branch_budget is None else branch_budget
        self._validate_budget(budget);start=time.perf_counter()
        if np.iscomplexobj(means) or np.iscomplexobj(draws):raise ValueError('Replay inputs must be real')
        means=np.ascontiguousarray(means,dtype=np.float64)
        draws=np.ascontiguousarray(draws,dtype=np.float64)
        if means.ndim!=2 or means.shape[1]!=2*self.inputs or len(means)>1048576:
            raise ValueError('Invalid realized means shape')
        count=len(means)
        if draws.shape!=(count,self.modes) or not np.isfinite(means).all() or not np.isfinite(draws).all():
            raise ValueError('Invalid replay arrays')
        if np.any(draws<0) or np.any(draws>=1):raise ValueError('Uniform draws must lie in [0,1)')
        clicks=np.zeros((count,self.modes),np.uint8);prob=np.zeros((count,self.modes))
        errors=np.zeros((count,2),np.int32);diag=np.zeros((count,3));timings=np.zeros(6)
        error=C.create_string_buffer(4096);gpu_start=time.perf_counter()
        rc=self.gpu.gc72_sample(self.modes,2*self.inputs,count,budget,
            self._x.ctypes.data_as(DP),self._a.ctypes.data_as(DP),self._b.ctypes.data_as(DP),
            means.ctypes.data_as(DP),draws.ctypes.data_as(DP),clicks.ctypes.data_as(BP),
            prob.ctypes.data_as(DP),errors.ctypes.data_as(IP),diag.ctypes.data_as(DP),
            timings.ctypes.data_as(DP),error,len(error))
        gpu_seconds=time.perf_counter()-gpu_start
        if rc:raise ArithmeticError(error.value.decode())
        return dict(clicks=clicks,conditional_click_probabilities=prob,
            trajectory_diagnostics=diag,statistics=dict(completed_samples=count,
            clicks=int(clicks.sum()),compressions=int(diag[:,0].sum()),
            peak_transient_components=int(diag[:,1].max(initial=0)),
            maximum_absolute_weight_sum=float(diag[:,2].max(initial=0))),
            cuda_timings=dict(zip(('allocation_seconds','host_to_device_seconds',
                'kernel_seconds','device_to_host_seconds','workspace_bytes','failed_rows'),timings.tolist())),
            gpu_seconds=gpu_seconds,wall_seconds=time.perf_counter()-start)
