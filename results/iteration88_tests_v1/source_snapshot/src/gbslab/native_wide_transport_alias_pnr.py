"""Fixed-total photon pairs sharing a calibrated latent coupling.
The supplied nonnegative tables must reproduce Q and R within 1e-8.
This restores the ordered-pair marginal, not the complete pure GBS law.
Integer alias choices use one uniform 64-bit word, with the actual table law
exported and its deviation from the calibrated input reported explicitly.
Supports at most 384 outputs and 16 million upper-triangle table entries.
"""
import ctypes as C
import time
import numpy as np
from .native_structured_pnr import NativeStructuredPNRSampler
from .native_polynomial_pnr import DP,UP
STATS77=('photons','completed_samples','maximum_photons','total_draws','detector_steps','maximum_factors','maximum_terms','maximum_rank','tail_records','tail_photons')
class NativeWideTransportAliasPNRSampler(NativeStructuredPNRSampler):
    def __init__(self,v,squeezing,library,*,couplings,latent_weights,exact_total_budget=20,**kwargs):
        if isinstance(exact_total_budget,bool) or not isinstance(exact_total_budget,int) or not 0<=exact_total_budget<=498 or exact_total_budget%2:raise ValueError("exact_total_budget must be an even integer in [0,498]")
        self.exact_total_budget=exact_total_budget
        t=np.asarray(squeezing,dtype=np.float64)
        if t.ndim==0:t=np.full(np.asarray(v).shape[1],float(t))
        if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or np.any(t<0) or np.any(t>=1) or not np.all(t==t[0]):raise ValueError('Number sampler requires equal squeezing in [0,1)')
        optics=np.asarray(v,dtype=complex);m,n=optics.shape
        if m>384:raise ValueError('Transport tables support at most 384 outputs')
        full=np.array(couplings,dtype=float,order='C',copy=True);weights=np.array(latent_weights,dtype=float,order='C',copy=True)
        if weights.ndim!=1 or not 1<=len(weights)<=8192 or full.shape!=(len(weights),m,m):raise ValueError('Invalid transport table shape')
        if len(weights)*m*(m+1)//2>16000000:raise ValueError('Transport table resource bound')
        if not np.isfinite(full).all() or not np.isfinite(weights).all() or np.any(full<0) or np.any(weights<0):raise ValueError('Transport probabilities must be finite and nonnegative')
        if abs(weights.sum()-1)>1e-8 or np.max(abs(full.sum(axis=(1,2))-1))>1e-8 or np.max(abs(full-full.transpose(0,2,1)))>1e-8:raise ValueError('Transport tables must be symmetric and normalized')
        marginal=full.sum(axis=2);ell=(abs(optics)**2).sum(axis=1);Q=abs(optics@optics.T)**2/n;R=(np.outer(ell,ell)+abs(optics@optics.conj().T)**2+abs(optics@optics.T)**2)/(n*(n+2))
        self.pair_constraint_error=float(np.max(abs(np.einsum('l,lij->ij',weights,full)-Q)))
        self.cross_pair_constraint_error=float(np.max(abs(marginal.T@(marginal*weights[:,None])-R)))
        if max(self.pair_constraint_error,self.cross_pair_constraint_error)>1e-8:raise ValueError('Transport tables do not match target pair constraints')
        self.latent_weights=np.ascontiguousarray(weights)
        self.pair_masses=np.ascontiguousarray(np.array([full[:,i,j]*(1 if i==j else 2) for i in range(m) for j in range(i,m)]).T)
        self.latent_weights.flags.writeable=False;self.pair_masses.flags.writeable=False
        super().__init__(v,t,library,retained=0,**kwargs)
        if np.any(self.mean) or np.any(self.noise_variance):self.close();raise ValueError('Number sampler requires zero displacement and no noise')
        self.lib.transport80a_sample.argtypes=[C.c_void_p,C.c_double,C.c_int,C.c_uint64,C.c_int,C.c_int,C.c_int,C.c_int,DP,DP,UP,C.POINTER(C.c_uint64),DP,C.c_char_p,C.c_int];self.lib.transport80a_sample.restype=C.c_int
        self.lib.transport80_alias_law.argtypes=[C.c_int,C.c_int,DP,DP,DP,DP,DP,C.c_char_p,C.c_int];self.lib.transport80_alias_law.restype=C.c_int
        self.actual_latent_weights=np.zeros_like(self.latent_weights);self.actual_pair_masses=np.zeros_like(self.pair_masses);bounds=np.zeros(2);message=C.create_string_buffer(4096)
        rc=self.lib.transport80_alias_law(m,len(weights),self.latent_weights.ctypes.data_as(DP),self.pair_masses.ctypes.data_as(DP),self.actual_latent_weights.ctypes.data_as(DP),self.actual_pair_masses.ctypes.data_as(DP),bounds.ctypes.data_as(DP),message,len(message))
        if rc:self.close();raise ArithmeticError(message.value.decode())
        self.alias_latent_tv=float(bounds[0]);self.alias_maximum_pair_tv=float(bounds[1])
        self.actual_latent_weights.flags.writeable=False;self.actual_pair_masses.flags.writeable=False
    def sample(self,count,seed,*,workers=None,fixed_total=None):
        if not self.handle:raise RuntimeError('Sampler is closed')
        if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=np.iinfo(np.int32).max:raise ValueError('Invalid sample count')
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:raise ValueError('seed must be uint64')
        if fixed_total is not None and (isinstance(fixed_total,bool) or not isinstance(fixed_total,int) or fixed_total<0 or fixed_total%2 or fixed_total>np.iinfo(np.int32).max):raise ValueError('fixed_total must be a nonnegative even integer')
        if np.any(self.mean) or np.any(self.noise_variance):raise ValueError('Number sampler requires zero displacement and no noise')
        if not np.isfinite(self.squeezing).all() or not np.all(self.squeezing==self.squeezing[0]):raise ValueError('Number sampler requires equal finite squeezing')
        workers=self.workers if workers is None else workers;self._validate_workers(workers);start=time.perf_counter();counts=np.zeros((count,self.modes),np.uint32);totals=np.zeros(count,np.uint64);stats=np.zeros(10);error=C.create_string_buffer(4096)
        rc=self.lib.transport80a_sample(self.handle,float(self.squeezing[0]),count,seed,workers,-1 if fixed_total is None else fixed_total,self.exact_total_budget,len(self.latent_weights),self.latent_weights.ctypes.data_as(DP),self.pair_masses.ctypes.data_as(DP),counts.ctypes.data_as(UP),totals.ctypes.data_as(C.POINTER(C.c_uint64)),stats.ctypes.data_as(DP),error,len(error))
        if rc:raise ArithmeticError(error.value.decode())
        return dict(exact_total_budget=self.exact_total_budget,counts=counts,total_photons=totals,statistics=dict(zip(STATS77,stats.tolist())),wall_seconds=time.perf_counter()-start)
