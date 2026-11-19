"""Native low-memory factorized photon pairs, exact low-total branch unchanged.
Counts are generated before thresholding. This remains the calibrated-pair
approximation to GBS, not an exact all-total sampler. Integer rejection law is
auditable; a one-million-attempt failure guard never silently clips probabilities.
"""
import ctypes as C
import time
import numpy as np
from .native_structured_pnr import NativeStructuredPNRSampler
from .native_cached_transport_pnr import STATS77
from .native_polynomial_pnr import DP,UP

class NativeFactorPNRSampler(NativeStructuredPNRSampler):
 def __init__(self,v,squeezing,library,*,kernel,left,right,latent_weights,exact_total_budget=8,**kwargs):
  self._plan=None
  if isinstance(exact_total_budget,bool) or not isinstance(exact_total_budget,int) or exact_total_budget<0 or exact_total_budget>498 or exact_total_budget%2:raise ValueError('Invalid exact total budget')
  self.exact_total_budget=exact_total_budget
  optics=np.asarray(v);m,n=optics.shape;t=np.asarray(squeezing,dtype=float)
  if t.ndim==0:t=np.full(n,float(t))
  if t.shape!=(n,) or not np.isfinite(t).all() or np.any(t<0) or np.any(t>=1) or not np.all(t==t[0]):raise ValueError('Factor PNR requires equal nonnegative squeezing')
  arrays=[np.ascontiguousarray(a,dtype=float) for a in (latent_weights,kernel,left,right)];w,k,a,b=arrays
  if w.ndim!=1 or not 1<=len(w)<=131072 or m>1024 or len(w)*m>64000000 or k.shape!=(m,m) or a.shape!=(len(w),m) or b.shape!=a.shape:raise ValueError('Invalid factor dimensions or resource bound')
  if any(not x.size or not np.isfinite(x).all() or np.any(x<0) for x in arrays):raise ValueError('Factors must be finite and nonnegative')
  self.latent=len(w);super().__init__(v,t,library,retained=0,**kwargs)
  if np.any(self.mean) or np.any(self.noise_variance):self.close();raise ValueError('Factor PNR requires zero displacement and noise')
  self.lib.factor82_create.argtypes=[C.c_void_p,C.c_int,DP,DP,DP,DP,C.c_int,DP,C.c_char_p,C.c_int];self.lib.factor82_create.restype=C.c_void_p
  self.lib.factor82_destroy.argtypes=[C.c_void_p];self.lib.factor82_destroy.restype=None
  self.lib.factor82_sample.argtypes=[C.c_void_p,C.c_void_p,C.c_double,C.c_int,C.c_uint64,C.c_int,C.c_int,C.c_int,UP,C.POINTER(C.c_uint64),DP,C.c_char_p,C.c_int];self.lib.factor82_sample.restype=C.c_int
  self.lib.factor82_law.argtypes=[C.c_void_p,DP,DP,C.c_char_p,C.c_int];self.lib.factor82_law.restype=C.c_int
  self.lib.factor82_integer.argtypes=[C.c_void_p,C.c_int,C.POINTER(C.c_uint64),C.POINTER(C.c_int),C.POINTER(C.c_uint64),C.POINTER(C.c_int),C.c_char_p,C.c_int];self.lib.factor82_integer.restype=C.c_int
  stats=np.zeros(8);error=C.create_string_buffer(4096);start=time.perf_counter();self._plan=self.lib.factor82_create(self.handle,self.latent,*(x.ctypes.data_as(DP) for x in arrays),self.workers,stats.ctypes.data_as(DP),error,len(error));self.plan_seconds=time.perf_counter()-start
  if not self._plan:self.close();raise ArithmeticError(error.value.decode())
  self.plan_statistics=dict(zip(('payload_bytes','minimum_acceptance','pair_constraint_error','cross_pair_constraint_error','latent_alias_numeric_tv','proposal_alias_numeric_tv','root_alias_numeric_tv','maximum_attempts'),stats.tolist()))
 def sample(self,count,seed,*,workers=None,fixed_total=None):
  if not self.handle:raise RuntimeError('Sampler closed')
  if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=np.iinfo(np.int32).max:raise ValueError('Invalid sample count')
  if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:raise ValueError('Invalid seed')
  if fixed_total is not None and (isinstance(fixed_total,bool) or not isinstance(fixed_total,int) or fixed_total<0 or fixed_total%2 or fixed_total>np.iinfo(np.int32).max):raise ValueError('Invalid even fixed total')
  if np.any(self.mean) or np.any(self.noise_variance) or not np.isfinite(self.squeezing).all() or not np.all(self.squeezing==self.squeezing[0]):raise ValueError('Factor PNR requires equal squeezing and zero mean/noise')
  workers=self.workers if workers is None else workers;self._validate_workers(workers);start=time.perf_counter();counts=np.zeros((count,self.modes),np.uint32);totals=np.zeros(count,np.uint64);stats=np.zeros(12);error=C.create_string_buffer(4096)
  rc=self.lib.factor82_sample(self.handle,self._plan,float(self.squeezing[0]),count,seed,workers,-1 if fixed_total is None else fixed_total,self.exact_total_budget,counts.ctypes.data_as(UP),totals.ctypes.data_as(C.POINTER(C.c_uint64)),stats.ctypes.data_as(DP),error,len(error))
  if rc:raise ArithmeticError(error.value.decode())
  return dict(counts=counts,total_photons=totals,wall_seconds=time.perf_counter()-start,statistics=dict(zip((*STATS77,'pair_proposals','pair_rejections'),stats.tolist())))
 def law(self):
  if not self.handle:raise RuntimeError('Sampler closed')
  if self.modes>16 or self.latent>512:raise ValueError('Full law audit bound')
  weights=np.zeros(self.latent);pairs=np.zeros((self.latent,self.modes,self.modes));error=C.create_string_buffer(4096);rc=self.lib.factor82_law(self._plan,weights.ctypes.data_as(DP),pairs.ctypes.data_as(DP),error,len(error))
  if rc:raise ArithmeticError(error.value.decode())
  return dict(weights=weights,oriented_pairs=pairs)
 def integer_tables(self,label):
  if not self.handle:raise RuntimeError('Sampler closed')
  if self.modes>16 or self.latent>512 or isinstance(label,bool) or not isinstance(label,int) or not 0<=label<self.latent:raise ValueError('Invalid integer audit label')
  slots=max(2,1<<(self.latent-1).bit_length())+(self.modes+1)*max(2,1<<(self.modes-1).bit_length());thresholds=np.zeros(slots,np.uint64);aliases=np.zeros(slots,np.int32);accept=np.zeros(self.modes,np.uint64);sizes=np.zeros(self.modes+2,np.int32);error=C.create_string_buffer(4096)
  rc=self.lib.factor82_integer(self._plan,label,thresholds.ctypes.data_as(C.POINTER(C.c_uint64)),aliases.ctypes.data_as(C.POINTER(C.c_int)),accept.ctypes.data_as(C.POINTER(C.c_uint64)),sizes.ctypes.data_as(C.POINTER(C.c_int)),error,len(error))
  if rc:raise ArithmeticError(error.value.decode())
  return dict(thresholds=thresholds,aliases=aliases,accept=accept,sizes=sizes)
 def close(self):
  if getattr(self,'_plan',None):self.lib.factor82_destroy(self._plan);self._plan=None
  super().close()
