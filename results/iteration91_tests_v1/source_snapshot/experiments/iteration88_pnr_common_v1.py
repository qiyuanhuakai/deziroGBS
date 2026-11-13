"""Two-mode physical inputs and explicit bindings for the frozen87 PNR kernel."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,ctypes as C,shutil,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_direct_click import NativeDirectClickSampler
DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32);IP=C.POINTER(C.c_int)
def ptr(a):return a.ctypes.data_as(DP)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(out,name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def manifest(out):save(out,'manifest.json',dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}))
def dirs():return [Path(shutil.which('g++')).parent]
def small_input(case):
 if case['name']=='thermal_control':
  n=np.array(case['mean_photons']);t=np.zeros(2);v=np.eye(2,dtype=complex);x=np.zeros((4,4));L=np.diag(np.sqrt(np.r_[n,n]));target_x=L@L.T
 else:
  r,eta=case['squeezing'],case['eta'];core=eta*r/(1+r-eta*r);t=np.array([core,-core]);original=np.array([r,-r]);v=np.array([[1,1],[1,-1]],complex)/np.sqrt(2);x=np.diag(np.r_[-t/(1+t),t/(1-t)]);target_x=eta*np.diag(np.r_[-original/(1+original),original/(1-original)]);noise=np.diag(target_x-x);assert noise.min()>-1e-14;L=np.diag(np.sqrt(np.maximum(noise,0.)))
 return dict(v=v,t=t,x=x,L=L,mean=np.zeros(4),target_x=target_x)
def tree(model,block=16):
 fn=model.lib.pnr87_tree_create;fn.argtypes=[C.c_void_p,C.c_int,C.c_char_p,C.c_int];fn.restype=C.c_void_p;destroy=model.lib.pnr87_tree_destroy;destroy.argtypes=[C.c_void_p];err=C.create_string_buffer(4096);handle=fn(model.handle,block,err,len(err));assert handle,err.value;return handle,destroy
def pnr(model,handle,q,retained,count,experiment,offset=0,threads=16,sector=0):
 m=len(q['v']);counts=np.zeros((count,m),np.uint32);stats=np.zeros((count,20));errors=np.zeros((count,2),np.int32);err=C.create_string_buffer(4096);fn=model.lib.pnr87_correlated_sample;fn.argtypes=[C.c_void_p,C.c_void_p,DP,DP,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,C.c_int,DP,DP,UP,DP,IP,C.c_char_p,C.c_int];start=time.perf_counter();rc=fn(model.handle,handle,ptr(q['t']),ptr(q['L']),count,experiment,sector,offset,threads,retained,0,ptr(q['mean']),None,counts.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),err,len(err));seconds=time.perf_counter()-start;valid=(errors[:,0]==0)&(stats[:,1]==1);return dict(counts=counts,statistics=stats,errors=errors,valid=valid),dict(returncode=rc,error=err.value.decode(),failed=int((~valid).sum()),seconds=seconds)
