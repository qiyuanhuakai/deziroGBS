"""Experimental native approximate PNR sampling by Gaussian moment resets.

The C++ implementation follows Lim arXiv:2610.06526v1, including the displaced
variant, and adds detector rejection envelopes. It does NOT sample the original
Gaussian target exactly and provides no full-distribution accuracy certificate.
All trajectory evolution, random draws and detector selection run in C++.

Conventions: input covariance is I/2 + X in grouped q,p coordinates, and V is a
complex M-by-N isometry. Mean/noise coordinates have length 2N in that subspace.
The model applies independent real Gaussian displacement noise per trajectory.
Use an explicitly built trusted library, currently Windows x86-64 GNU C++.
Optional OpenBLAS must have NumPy's ILP64 interface. Its process-wide thread
count is restored after each sample call; concurrent thread-setting calls are
unsupported. There is no photon or rejection-attempt cutoff.
"""
import ctypes as C
import os
from pathlib import Path
import time
import numpy as np

DP=C.POINTER(C.c_double)
UP=C.POINTER(C.c_uint32)
SELECTORS={'scalar':0,'eigen':1,'leverage':2,'diagonal':3,'openblas':4,'anchor':5}
STAT_NAMES=('photons','proposals','flow_seconds','selection_seconds',
            'reset_seconds','vacuum_samples','maximum_photons','anchor_seconds')


def lossy_squeezed_components(inputs, squeezing, transmission):
    """Return X_core and diagonal noise variance; vacuum covariance is I/2.

    This is the exact input-state decomposition S62, not an exact sampling
    algorithm after subsequent Gaussian resets. Equal parameters only.
    """
    if isinstance(inputs,bool) or not isinstance(inputs,int) or inputs<1:
        raise ValueError('inputs must be a positive integer')
    if not np.isfinite(squeezing) or squeezing<0:
        raise ValueError('squeezing must be finite and nonnegative')
    if not np.isfinite(transmission) or not 0<=transmission<=1:
        raise ValueError('transmission must be in [0,1]')
    # Exact zero-loss/vacuum limits avoid a negative rounded noise variance.
    if transmission==0 or squeezing==0:
        return np.zeros((2*inputs,2*inputs)),np.zeros(2*inputs)
    with np.errstate(over='raise',invalid='raise'):
        vp=.5*(transmission*np.exp(2*squeezing)+1-transmission)
        vm=.5*(transmission*np.exp(-2*squeezing)+1-transmission)
        core=-.5*np.log(2*vm)
        x=np.diag(np.r_[np.full(inputs,.5*np.expm1(2*core)),
                        np.full(inputs,.5*np.expm1(-2*core))])
        # vp - 1/(4 vm), algebraically rearranged to avoid cancellation.
        noise=(transmission*(1-transmission)*np.sinh(squeezing)**2)/vm
    if not np.isfinite(x).all() or not np.isfinite(noise) or noise<0:
        raise ArithmeticError('Input decomposition overflow or invalid variance')
    return x,np.r_[np.full(inputs,noise),np.zeros(inputs)]


class NativeGaussianResetSampler:
    def __init__(self,isometry,excess_covariance,library,*,mean=None,
                 noise_variance=None,runtime_dirs=(),blas_library=None,threads=1):
        self.handle=None
        if isinstance(threads,bool) or not isinstance(threads,int) or not 1<=threads<=64:
            raise ValueError('Invalid BLAS thread count')
        if np.iscomplexobj(excess_covariance):
            raise ValueError('Covariance must be real')
        v=np.ascontiguousarray(isometry,dtype=np.complex128)
        x=np.ascontiguousarray(excess_covariance,dtype=np.float64)
        if v.ndim!=2 or not 1<=v.shape[1]<=v.shape[0] or max(v.shape)>np.iinfo(np.int32).max//2:
            raise ValueError('Invalid isometry dimensions')
        self.modes,self.inputs=v.shape;size=2*self.inputs
        if x.shape!=(size,size) or not np.isfinite(x).all() or not np.isfinite(v).all():
            raise ValueError('Invalid covariance or isometry')
        self.mean=self._vector(mean,size,'mean')
        self.noise_variance=self._vector(noise_variance,size,'noise variance')
        if np.any(self.noise_variance<0):raise ValueError('Negative noise variance')
        self.runtime=[os.add_dll_directory(str(Path(p).resolve())) for p in runtime_dirs] if os.name=='nt' else []
        self.lib=C.CDLL(str(Path(library).resolve()));self.threads=threads;self.blas=None
        self.lib.gr_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];self.lib.gr_create.restype=C.c_void_p
        self.lib.gr_destroy.argtypes=[C.c_void_p];self.lib.gr_destroy.restype=None
        self.lib.ds_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int];self.lib.ds_sample.restype=C.c_int
        self.lib.ds_sample_anchor.argtypes=[C.c_void_p,C.c_int,C.c_uint64,DP,DP,UP,DP,C.c_char_p,C.c_int];self.lib.ds_sample_anchor.restype=C.c_int
        error=C.create_string_buffer(4096);start=time.perf_counter()
        self.handle=self.lib.gr_create(self.modes,self.inputs,v.ctypes.data_as(DP),x.ctypes.data_as(DP),error,len(error));self.model_seconds=time.perf_counter()-start
        if not self.handle:raise ValueError(error.value.decode())
        if blas_library is not None:
            try:
                self.blas=C.CDLL(str(Path(blas_library).resolve()))
                self.setter=self.blas.scipy_openblas_set_num_threads64_;self.setter.argtypes=[C.c_int];self.setter.restype=None
                self.getter=self.blas.scipy_openblas_get_num_threads64_;self.getter.argtypes=[];self.getter.restype=C.c_int
                self.lib.gr_set_gemm.argtypes=[C.c_void_p,C.c_void_p];self.lib.gr_set_gemm.restype=None
                self.lib.gr_set_gemm(self.handle,C.cast(self.blas.scipy_cblas_zgemm64_,C.c_void_p))
            except Exception:
                self.close();raise

    @staticmethod
    def _vector(value,size,name):
        if value is None:return np.zeros(size)
        if np.iscomplexobj(value):raise ValueError(name+' must be real')
        a=np.array(value,dtype=np.float64,copy=True,order='C')
        if a.shape!=(size,) or not np.isfinite(a).all():raise ValueError('Invalid '+name)
        return a

    def sample(self,count,seed,*,selector='diagonal'):
        if not self.handle:raise RuntimeError('Sampler is closed')
        if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=np.iinfo(np.int32).max:
            raise ValueError('Invalid sample count')
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:
            raise ValueError('seed must be a uint64 integer')
        if selector not in SELECTORS:raise ValueError('Invalid detector selector')
        if selector=='openblas' and self.blas is None:raise ValueError('OpenBLAS library was not supplied')
        output=np.zeros((count,self.modes),np.uint32);stats=np.zeros(8);error=C.create_string_buffer(4096)
        if count:
            args=[self.handle,count,seed]
            if selector!='anchor':args.append(SELECTORS[selector])
            args.extend([self.mean.ctypes.data_as(DP),self.noise_variance.ctypes.data_as(DP),output.ctypes.data_as(UP),stats.ctypes.data_as(DP),error,len(error)])
            old=self.getter() if self.blas is not None else None
            try:
                if old is not None:self.setter(self.threads)
                call=self.lib.ds_sample_anchor if selector=='anchor' else self.lib.ds_sample
                rc=call(*args)
            finally:
                if old is not None:self.setter(old)
            if rc:raise ArithmeticError(error.value.decode())
        return {'counts':output,'statistics':dict(zip(STAT_NAMES,stats.tolist()))}

    def close(self):
        if self.handle:self.lib.gr_destroy(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
    def __del__(self):
        if getattr(self,'handle',None):self.close()
