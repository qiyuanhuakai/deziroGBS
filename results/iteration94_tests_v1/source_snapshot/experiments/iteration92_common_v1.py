"""Explicit rank-two coherent-core bindings; experiment/spectrum/record/purpose RNG."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import ctypes as C,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32);IP=C.POINTER(C.c_int)
def ptr(x):return x.ctypes.data_as(DP)
def H(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(out,name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def manifest(out):save(out,'manifest.json',{p.name:H(p) for p in out.iterdir() if p.is_file()})
def load():
    runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));lib=C.CDLL(str(ROOT/'results/iteration92_build_v1/core.dll'));return lib,runtime
def fock_core(k):
    c=np.zeros((k+1,k+1),complex);c[k,0]=1;return c
def flow(lib,c,r,eta):
    c=np.ascontiguousarray(c,dtype=np.complex128);out=np.zeros_like(c);survival=C.c_double();msg=C.create_string_buffer(4096);fn=lib.coherent92_flow;fn.argtypes=[C.c_int,DP,C.c_double,C.c_double,DP,C.POINTER(C.c_double),C.c_char_p,C.c_int];rc=fn(len(c)-1,ptr(c.view(float)),r,eta,ptr(out.view(float)),C.byref(survival),msg,len(msg));assert rc==0,msg.value;return out,survival.value
def sample(lib,u,c,r,count,experiment,sector=0,offset=0,threads=16,cap=128):
    u=np.ascontiguousarray(u,dtype=np.complex128);c=np.ascontiguousarray(c,dtype=np.complex128);m=len(u);counts=np.zeros((count,m),np.uint32);stats=np.zeros((count,8));errors=np.zeros(count,np.int32);msg=C.create_string_buffer(4096);fn=lib.coherent92_sample;fn.argtypes=[C.c_int,DP,C.c_int,DP,C.c_double,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,UP,DP,IP,C.c_char_p,C.c_int];start=time.perf_counter();rc=fn(m,ptr(u.view(float)),len(c)-1,ptr(c.view(float)),r,count,experiment,sector,offset,threads,cap,counts.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),msg,len(msg));return dict(counts=counts,statistics=stats,errors=errors),dict(returncode=rc,error=msg.value.decode(),seconds=time.perf_counter()-start,failures=int(np.count_nonzero(errors)))
