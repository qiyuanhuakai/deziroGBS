"""Recorded Frugal fp64 CPU adapter; pad two vacuum modes to its four-mode ABI."""
from iteration92_common_v2 import *
BP=C.POINTER(C.c_uint8)
def frugal_covariance(r):
    v=(1+r*r)/(1-r*r);c=2*r/(1-r*r);cov=np.eye(8);cov[0,0]=cov[1,1]=cov[4,4]=cov[5,5]=v;cov[0,1]=cov[1,0]=c;cov[4,5]=cov[5,4]=-c;return cov
def load_frugal():return C.CDLL(str(ROOT/'results/iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll'))
def run_frugal(lib,cov,N,seed,threads=1):
    cov=np.ascontiguousarray(cov);bits=np.zeros((N,4),np.uint8);stats=np.zeros(7);msg=C.create_string_buffer(4096);fn=lib.fc73_sample;fn.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint,C.c_int,DP,BP,DP,C.c_char_p,C.c_int];start=time.perf_counter();rc=fn(4,4,N,seed,threads,ptr(cov),bits.ctypes.data_as(BP),ptr(stats),msg,len(msg));return bits,dict(returncode=rc,error=msg.value.decode(),seconds=time.perf_counter()-start,components=stats.tolist())
