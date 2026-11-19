"""FP64 C++/CUDA trajectories for a normalized finite-input MPS.

Use as a context manager, with an explicitly built and trusted library. The
default performs the same exact QR normalization as DisplacedThresholdSampler.
``canonical=True`` skips QR only for already normalized right-canonical tensors;
it does not certify that precondition. CUDA currently requires local physical
dimensions <=32. Neither backend certifies the input Gaussian approximation.

The optional CPU BLAS library must expose NumPy's ILP64 scipy_cblas_zgemm64_
interface. Its process-wide thread count is restored after each call; concurrent
calls that change that thread count are unsupported. Libraries are never built
or downloaded on import, and numerical rejections never fall back silently.
"""
import ctypes as C
import os
from pathlib import Path
import time
import numpy as np
from gbslab.displaced_sampling import coherent_direction
from gbslab.mps_overlap import normalized_right_mps, validate_mps

DP=C.POINTER(C.c_double); IP=C.POINTER(C.c_int); PP=C.POINTER(DP)

class NativeDisplacedThresholdSampler:
    def __init__(self,tensors,library,*,backend,runtime_dirs=(),blas_library=None,canonical=False,threads=1,tile=16):
        if backend not in ('cpu','cuda'): raise ValueError('backend must be cpu or cuda')
        if not isinstance(threads,int) or not 1<=threads<=64: raise ValueError('Invalid thread count')
        if not isinstance(tile,int) or not 1<=tile<=128: raise ValueError('Invalid tile size')
        if not isinstance(canonical,bool): raise ValueError('canonical must be bool')
        validate_mps(tensors)
        self.backend=backend;self.threads=threads;self.tile=tile;self.handle=None
        self.runtime=[os.add_dll_directory(str(Path(p).resolve())) for p in runtime_dirs] if os.name=='nt' else []
        self.lib=C.CDLL(str(Path(library).resolve()))
        start=time.perf_counter()
        prepared,self.input_log_norm=(tensors,0.) if canonical else normalized_right_mps(tensors)
        self.tensors=[np.ascontiguousarray(a,dtype=np.complex128) for a in prepared]
        self.normalization_seconds=time.perf_counter()-start
        self.modes=len(self.tensors)
        if backend=='cuda' and any(a.shape[1]>32 for a in self.tensors): raise ValueError('CUDA v2 prototype supports physical dimension <=32')
        prefix='nds' if backend=='cpu' else 'gds'
        self.create=getattr(self.lib,prefix+'_create');self.create.argtypes=[C.c_int,IP,PP,C.c_char_p,C.c_int];self.create.restype=C.c_void_p
        self.destroy=getattr(self.lib,prefix+'_destroy');self.destroy.argtypes=[C.c_void_p];self.destroy.restype=None
        if backend=='cpu':
            self.call=self.lib.nds_sample;self.call.argtypes=[C.c_void_p,C.c_int,PP,PP,DP,C.c_int,C.c_int,IP,DP,DP,DP,C.c_char_p,C.c_int]
        else:
            self.call=self.lib.gds_sample_beta;self.call.argtypes=[C.c_void_p,C.c_int,DP,DP,IP,DP,DP,DP,C.c_char_p,C.c_int]
        self.call.restype=C.c_int
        if self.modes>np.iinfo(np.int32).max or any(max(a.shape)>np.iinfo(np.int32).max for a in self.tensors): raise ValueError('Native dimension overflow')
        dims=np.array([a.shape for a in self.tensors],np.int32);ptrs=(DP*self.modes)(*(a.ctypes.data_as(DP) for a in self.tensors));error=C.create_string_buffer(1024)
        start=time.perf_counter();self.handle=self.create(self.modes,dims.ctypes.data_as(IP),ptrs,error,1024);self.model_seconds=time.perf_counter()-start
        if not self.handle: raise ValueError(error.value.decode())
        self.blas=None
        if backend=='cpu' and blas_library is not None:
            try:
                self.blas=C.CDLL(str(Path(blas_library).resolve()))
                self.setter=self.blas.scipy_openblas_set_num_threads64_;self.setter.argtypes=[C.c_int];self.setter.restype=None
                self.getter=self.blas.scipy_openblas_get_num_threads64_;self.getter.restype=C.c_int
                self.lib.nds_set_gemm.argtypes=[C.c_void_p,C.c_void_p];self.lib.nds_set_gemm.restype=None
                self.lib.nds_set_gemm(self.handle,C.cast(self.blas.scipy_cblas_zgemm64_,C.c_void_p))
            except Exception:
                self.close();raise

    def sample(self,beta,uniforms):
        if not self.handle: raise RuntimeError('Sampler is closed')
        if np.iscomplexobj(uniforms): raise ValueError('Uniform coordinates must be real')
        beta=np.ascontiguousarray(beta,dtype=np.complex128);uniforms=np.ascontiguousarray(uniforms,dtype=np.float64)
        if beta.ndim!=2 or beta.shape[1]!=self.modes or uniforms.shape!=beta.shape or not np.all(np.isfinite(beta)) or not np.all(np.isfinite(uniforms)) or np.any((uniforms<0)|(uniforms>=1)): raise ValueError('Invalid displacement or uniform array')
        if len(beta)>np.iinfo(np.int32).max: raise ValueError('Native batch overflow')
        labels=np.empty(beta.shape,np.int32);selected=np.empty(beta.shape);vacuum=np.empty(beta.shape);residual=C.c_double();error=C.create_string_buffer(1024)
        if len(beta):
            args=[self.handle,len(beta)]
            if self.backend=='cpu':
                cs=[];ps=[]
                for site,a in enumerate(self.tensors):
                    c,w,t=coherent_direction(beta[:,site],a.shape[1]);cs.append(np.ascontiguousarray(c));ps.append(np.ascontiguousarray(np.column_stack((w,t))))
                cp=(DP*self.modes)(*(a.ctypes.data_as(DP) for a in cs));pp=(DP*self.modes)(*(a.ctypes.data_as(DP) for a in ps))
                args += [cp,pp,uniforms.ctypes.data_as(DP),self.threads,self.tile]
            else: args += [beta.ctypes.data_as(DP),uniforms.ctypes.data_as(DP)]
            args += [labels.ctypes.data_as(IP),selected.ctypes.data_as(DP),vacuum.ctypes.data_as(DP),C.byref(residual),error,1024]
            old=self.getter() if self.blas is not None else None
            try:
                if old is not None: self.setter(self.threads)
                rc=self.call(*args)
            finally:
                if old is not None: self.setter(old)
            if rc: raise ArithmeticError(error.value.decode())
        return dict(clicks=(labels!=0).astype(np.uint8),refined_labels=labels,selected_refined_probabilities=selected,vacuum_given_hidden_history=vacuum,maximum_normalization_residual=residual.value)

    def close(self):
        if self.handle:
            self.destroy(self.handle);self.handle=None
    def __enter__(self): return self
    def __exit__(self,*exc): self.close()
    def __del__(self):
        if getattr(self,'handle',None): self.close()
