"""CUDA branch sampling with device Philox RNG and stable row subsequences.

Same bounded-branch approximation as iteration72; a different random stream.
row_offset continues a stream without changing records at batch boundaries.
Diagnostic random-input export enables replay; its copy cost is included when
enabled. Performance experiments disable this additional export.
"""
import ctypes as C
import time
import numpy as np
from .native_cuda_click_branches import NativeCudaBranchClickSampler, DP, BP, IP


class NativePhiloxClickSampler(NativeCudaBranchClickSampler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gpu.gc73_sample.argtypes=[C.c_int,C.c_int,C.c_int,C.c_int,C.c_uint64,C.c_uint64,
            DP,DP,DP,DP,DP,BP,DP,IP,DP,DP,DP,DP,C.c_char_p,C.c_int]
        self.gpu.gc73_sample.restype=C.c_int

    def sample(self,count,seed,*,branch_budget=None,row_offset=0,export_random=False):
        if not self.handle:raise RuntimeError('Sampler is closed')
        if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=1048576:
            raise ValueError('count must be an integer in [0,1048576]')
        for name,value in (('seed',seed),('row_offset',row_offset)):
            if isinstance(value,bool) or not isinstance(value,int) or not 0<=value<2**64:
                raise ValueError(name+' must be a uint64 integer')
        if count and row_offset+count-1>=2**64:raise ValueError('Philox row offset overflow')
        if not isinstance(export_random,bool):raise ValueError('export_random must be boolean')
        budget=self.branch_budget if branch_budget is None else branch_budget;self._validate_budget(budget)
        start=time.perf_counter();clicks=np.zeros((count,self.modes),np.uint8)
        probabilities=np.zeros((count,self.modes));errors=np.zeros((count,2),np.int32)
        diag=np.zeros((count,3));timings=np.zeros(7);error=C.create_string_buffer(4096)
        means=np.zeros((count,2*self.inputs)) if export_random else None
        draws=np.zeros((count,self.modes)) if export_random else None
        rc=self.gpu.gc73_sample(self.modes,2*self.inputs,count,budget,seed,row_offset,
            self._x.ctypes.data_as(DP),self._a.ctypes.data_as(DP),self._b.ctypes.data_as(DP),
            self.mean.ctypes.data_as(DP),self.noise_variance.ctypes.data_as(DP),
            clicks.ctypes.data_as(BP),probabilities.ctypes.data_as(DP),errors.ctypes.data_as(IP),
            diag.ctypes.data_as(DP),timings.ctypes.data_as(DP),
            means.ctypes.data_as(DP) if export_random else None,
            draws.ctypes.data_as(DP) if export_random else None,error,len(error))
        if rc:raise ArithmeticError(error.value.decode())
        result=dict(clicks=clicks,conditional_click_probabilities=probabilities,
            trajectory_diagnostics=diag,statistics=dict(completed_samples=count,clicks=int(clicks.sum()),
            compressions=int(diag[:,0].sum()),peak_transient_components=int(diag[:,1].max(initial=0)),
            maximum_absolute_weight_sum=float(diag[:,2].max(initial=0))),
            cuda_timings=dict(zip(('allocation_seconds','host_to_device_seconds','random_seconds',
                'kernel_seconds','device_to_host_seconds','workspace_bytes','failed_rows'),timings.tolist())),
            wall_seconds=time.perf_counter()-start)
        if export_random:result.update(realized_means=means,uniform_draws=draws)
        return result
