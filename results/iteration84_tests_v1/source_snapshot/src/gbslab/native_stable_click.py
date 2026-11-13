"""Zero-mean extended-state/binary128-weight signed branch sampler.
Finite-budget compression remains approximate. B128 still has explicitly
preserved rare normalization failures; no clipping or silent recovery.
"""
import ctypes as C
import time
import numpy as np
from .native_click_branches import NativeBranchClickSampler
from .native_direct_click import DP,BP

class NativeStableClickSampler(NativeBranchClickSampler):
 @staticmethod
 def _validate_budget(value):
  if isinstance(value,bool) or not isinstance(value,int) or not 1<=value<=128:raise ValueError('Stable branch budget must be in [1,128]')
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  if np.any(self.mean) or np.any(self.noise_variance):self.close();raise ValueError('Stable sampler only supports zero mean and no displacement mixture')
  self.lib.tc82_prefix.argtypes=[C.c_void_p,C.c_int,BP,C.c_int,DP,DP,C.c_char_p,C.c_int];self.lib.tc82_prefix.restype=C.c_int
  self.lib.tc82_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,BP,DP,C.c_char_p,C.c_int];self.lib.tc82_sample.restype=C.c_int
 def _ready(self):
  if not self.handle:raise RuntimeError('Sampler closed')
  if np.any(self.mean) or np.any(self.noise_variance):raise ValueError('Stable sampler requires zero mean/noise')
 def prefix_probability(self,bits,*,branch_budget=None,mean=None):
  self._ready();bits=np.asarray(bits)
  if bits.ndim!=1 or len(bits)>min(16,self.modes) or not np.isin(bits,[0,1]).all():raise ValueError('Invalid stable prefix')
  if mean is not None and (np.asarray(mean).shape!=(2*self.inputs,) or np.any(np.asarray(mean)!=0)):raise ValueError('Stable prefix requires zero mean')
  budget=self.branch_budget if branch_budget is None else branch_budget;self._validate_budget(budget);bits=np.ascontiguousarray(bits,np.uint8);prob=np.zeros(1);stats=np.zeros(6);error=C.create_string_buffer(4096)
  rc=self.lib.tc82_prefix(self.handle,len(bits),bits.ctypes.data_as(BP),budget,prob.ctypes.data_as(DP),stats.ctypes.data_as(DP),error,len(error))
  if rc:raise ArithmeticError(error.value.decode())
  return dict(probability=float(prob[0]),component_evaluations=stats[0],compressions=stats[1],peak_transient_components=stats[2],maximum_absolute_weight_sum=stats[3],maximum_weight_mass_residual=stats[4])
 def sample(self,count,seed,*,workers=None,branch_budget=None):
  self._ready()
  if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=1048576:raise ValueError('Invalid stable sample count')
  if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:raise ValueError('Invalid stable seed')
  workers=self.workers if workers is None else workers;self._validate_workers(workers)
  if workers>64:raise ValueError('Stable worker limit is 64')
  budget=self.branch_budget if branch_budget is None else branch_budget;self._validate_budget(budget);start=time.perf_counter();bits=np.zeros((count,self.modes),np.uint8);stats=np.zeros(7);error=C.create_string_buffer(4096)
  rc=self.lib.tc82_sample(self.handle,count,seed,workers,budget,bits.ctypes.data_as(BP),stats.ctypes.data_as(DP),error,len(error))
  if rc:raise ArithmeticError(error.value.decode())
  return dict(clicks=bits,wall_seconds=time.perf_counter()-start,statistics=dict(zip(('clicks','completed_samples','component_evaluations','compressions','peak_transient_components','maximum_absolute_weight_sum','maximum_weight_mass_residual'),stats.tolist())))
