"""Exponential small-output Gaussian click table. Zero mean, covariance vacuum=I.
Negative floating Mobius masses raise errors; no silent clipping or fallback.
The 53-bit categorical grid and counter mixer are not a universal RNG certificate.
"""
import ctypes as C
import os,time
import numpy as np
DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8)
class NativeTableClickSampler:
    def __init__(self,covariance,library,*,workers=16,runtime_dirs=()):
        self.handle=None;self._dirs=[];self.workers=workers;self._workers(workers)
        cov=np.ascontiguousarray(covariance,dtype=np.float64)
        if cov.ndim!=2 or cov.shape[0]!=cov.shape[1] or cov.shape[0]%2 or not 2<=len(cov)<=32 or not np.isfinite(cov).all():raise ValueError('covariance must be finite 2m by 2m, 1<=m<=16')
        self.modes=len(cov)//2;self.covariance=cov
        try:
            if hasattr(os,'add_dll_directory'):self._dirs=[os.add_dll_directory(str(p)) for p in runtime_dirs]
            self.lib=C.CDLL(str(library));self.lib.table78_create.argtypes=[C.c_int,DP,C.c_char_p,C.c_int];self.lib.table78_create.restype=C.c_void_p;self.lib.table78_destroy.argtypes=[C.c_void_p];self.lib.table78_law.argtypes=[C.c_void_p,DP,DP,C.c_char_p,C.c_int];self.lib.table78_law.restype=C.c_int;self.lib.table78_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_uint64,C.c_int,BP,C.c_char_p,C.c_int];self.lib.table78_sample.restype=C.c_int
            error=C.create_string_buffer(4096);self.handle=self.lib.table78_create(self.modes,cov.ctypes.data_as(DP),error,len(error))
            if not self.handle:raise ArithmeticError(error.value.decode())
        except Exception:self.close();raise
    @staticmethod
    def _workers(workers):
        if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=256:raise ValueError('workers must be an integer in [1,256]')
    def law(self):
        if not self.handle:raise RuntimeError('Sampler is closed')
        p=np.zeros(2**self.modes);diag=np.zeros(2);error=C.create_string_buffer(4096);rc=self.lib.table78_law(self.handle,p.ctypes.data_as(DP),diag.ctypes.data_as(DP),error,len(error))
        if rc:raise ArithmeticError(error.value.decode())
        return dict(probabilities=p,raw_total=diag[0],states=int(diag[1]))
    def sample(self,count,seed,*,row_offset=0,workers=None):
        if not self.handle:raise RuntimeError('Sampler is closed')
        if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=np.iinfo(np.int32).max:raise ValueError('Invalid sample count')
        for value in (seed,row_offset):
            if isinstance(value,bool) or not isinstance(value,int) or not 0<=value<2**64:raise ValueError('seed and offset must be uint64')
        if count and row_offset+count-1>=2**64:raise ValueError('row offset overflow')
        workers=self.workers if workers is None else workers;self._workers(workers);start=time.perf_counter();bits=np.zeros((count,self.modes),np.uint8);error=C.create_string_buffer(4096);rc=self.lib.table78_sample(self.handle,count,seed,row_offset,workers,bits.ctypes.data_as(BP),error,len(error))
        if rc:raise ArithmeticError(error.value.decode())
        return dict(clicks=bits,wall_seconds=time.perf_counter()-start)
    def close(self):
        if self.handle:self.lib.table78_destroy(self.handle);self.handle=None
        for handle in self._dirs:handle.close()
        self._dirs=[]
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
