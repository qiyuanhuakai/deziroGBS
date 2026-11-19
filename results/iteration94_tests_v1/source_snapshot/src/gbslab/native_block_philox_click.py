"""GPU direct clicks with compression only at fixed block boundaries.
Block size 1 replays iteration74. Wider blocks retain intermediate branches;
workspace capacity is branch_budget * 2**(block_size-1), at most 128.
Approximate moment compression remains; no full high-dimensional TV guarantee.
"""
import ctypes as C
import time
import numpy as np
from .native_cuda_click_branches import NativeCudaBranchClickSampler, DP, BP, IP


class NativeBlockPhiloxClickSampler(NativeCudaBranchClickSampler):
    def __init__(self,*args,block_size=2,**kwargs):
        if isinstance(block_size,bool) or not isinstance(block_size,int) or not 1<=block_size<=4:
            raise ValueError('block_size must be an integer in [1,4]')
        budget=kwargs.get('branch_budget',4)
        self._validate_budget(budget)
        if budget*2**(block_size-1)>128:raise ValueError('Block workspace capacity exceeds 128')
        self.block_size=block_size
        super().__init__(*args,**kwargs)
        self.gpu.gc79_sample.argtypes=[C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_uint64,C.c_uint64,
            DP,DP,DP,DP,DP,BP,DP,IP,DP,DP,DP,DP,C.c_char_p,C.c_int]
        self.gpu.gc79_sample.restype=C.c_int
        self.gpu.gc79_replay.argtypes=[C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,DP,DP,DP,DP,DP,BP,DP,IP,DP,DP,C.c_char_p,C.c_int]
        self.gpu.gc79_replay.restype=C.c_int

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
        if budget*2**(self.block_size-1)>128:raise ValueError('Block workspace capacity exceeds 128')
        start=time.perf_counter();clicks=np.zeros((count,self.modes),np.uint8)
        probabilities=np.zeros((count,self.modes));errors=np.zeros((count,2),np.int32)
        diag=np.zeros((count,3));timings=np.zeros(7);error=C.create_string_buffer(4096)
        means=np.zeros((count,2*self.inputs)) if export_random else None
        draws=np.zeros((count,self.modes)) if export_random else None
        rc=self.gpu.gc79_sample(self.modes,2*self.inputs,count,budget,self.block_size,seed,row_offset,
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

    def replay(self,means,draws,*,branch_budget=None):
        """Sample with explicit displacements and uniforms; no marginal-law claim."""
        if not self.handle:raise RuntimeError('Sampler is closed')
        budget=self.branch_budget if branch_budget is None else branch_budget
        self._validate_budget(budget)
        if budget*2**(self.block_size-1)>128:raise ValueError('Block workspace capacity exceeds 128')
        start=time.perf_counter()
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
        rc=self.gpu.gc79_replay(self.modes,2*self.inputs,count,budget,self.block_size,
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
