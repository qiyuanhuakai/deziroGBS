"""Bindings for finite-occupation core layers and unresolved-tail sampling."""
from iteration91_common_v1 import *
import math
U64P=C.POINTER(C.c_uint64)
def core_law(lib,u,k,r,cutoff):
 u=np.ascontiguousarray(u,dtype=np.complex128);m=len(u);capacity=math.comb(cutoff+m,m);keys=np.zeros(capacity,np.uint64);probs=np.zeros(capacity);written=C.c_uint64();stats=np.zeros(4);msg=C.create_string_buffer(4096);fn=lib.fock91_law;fn.argtypes=[C.c_int,C.c_int,C.c_double,C.c_int,DP,C.c_uint64,U64P,DP,C.POINTER(C.c_uint64),DP,C.c_char_p,C.c_int];start=time.perf_counter();rc=fn(m,k,r,cutoff,ptr(u.view(float)),capacity,keys.ctypes.data_as(U64P),ptr(probs),C.byref(written),ptr(stats),msg,len(msg));meta=dict(returncode=rc,error=msg.value.decode(),seconds=time.perf_counter()-start,stats=stats.tolist());return keys[:written.value],probs[:written.value],meta
def core_draw(lib,keys,probs,draws):
 keys=np.ascontiguousarray(keys,dtype=np.uint64);probs=np.ascontiguousarray(probs);draws=np.ascontiguousarray(draws);out=np.zeros(len(draws),np.uint64);fails=np.zeros(len(draws),np.int32);msg=C.create_string_buffer(4096);fn=lib.fock91_draw;fn.argtypes=[C.c_uint64,U64P,DP,C.c_int,DP,U64P,IP,C.c_char_p,C.c_int];start=time.perf_counter();rc=fn(len(keys),keys.ctypes.data_as(U64P),ptr(probs),len(draws),ptr(draws),out.ctypes.data_as(U64P),fails.ctypes.data_as(IP),msg,len(msg));return out,fails,dict(returncode=rc,error=msg.value.decode(),seconds=time.perf_counter()-start)
