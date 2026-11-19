"""Explicit-array bindings shared by iteration87 guarded-click experiments."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,ctypes as C,shutil,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);IP=C.POINTER(C.c_int);U64P=C.POINTER(C.c_uint64)
def ptr(a):return a.ctypes.data_as(DP)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(out,name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def manifest(out):save(out,'manifest.json',dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}))
def runtime_dirs():return [Path(shutil.which('g++')).parent]
def sample(model,draws,budget,guarded=True,threads=16):
 draws=np.ascontiguousarray(draws);count,m=draws.shape;bits=np.zeros((count,m),np.uint8);prob=np.zeros((count,m));errors=np.zeros((count,2),np.int32);stats=np.zeros((count,12 if guarded else 6));err=C.create_string_buffer(4096);bounds=np.zeros((count,m,2));kind=np.zeros((count,m),np.uint8)
 fn=getattr(model.lib,'tc87_guarded_sample' if guarded else 'tc86_shared_sample');fn.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,DP,BP,DP,*([DP,BP] if guarded else []),IP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;start=time.perf_counter();rc=fn(model.handle,count,threads,budget,ptr(draws),bits.ctypes.data_as(BP),ptr(prob),*([ptr(bounds),kind.ctypes.data_as(BP)] if guarded else []),errors.ctypes.data_as(IP),ptr(stats),err,len(err));seconds=time.perf_counter()-start
 result=dict(clicks=bits,conditional=prob,errors=errors,statistics=stats)
 if guarded:result.update(bounds=bounds,kind=kind)
 return result,dict(returncode=rc,error=err.value.decode(),seconds=seconds)
def uniforms(experiment,sector,offset,count,m):
 runtime=os.add_dll_directory(str(runtime_dirs()[0]));lib=C.CDLL(str(ROOT/'results/iteration84_rng_gates_v1/rng.dll'));fn=lib.rng84_stream;fn.argtypes=[C.c_uint64,C.c_uint,C.c_uint,C.c_uint64,C.c_uint64,C.c_int,U64P,C.c_char_p,C.c_int];words=np.zeros((count,m),np.uint64);err=C.create_string_buffer(4096)
 for row in range(count):assert fn(experiment,sector,5,offset+row,0,m,words[row].ctypes.data_as(U64P),err,len(err))==0,err.value
 runtime.close();return np.ascontiguousarray((words>>11).astype(float)*2**-53)
def compare(old,new):
 b=new['bounds'];p=old['conditional'];s=new['statistics'];k=new['kind'];complete=(new['errors'][:,0]==0)&(s[:,5]==p.shape[1]);good=np.isfinite(p)&np.isfinite(b[:,:,0]);identity=bool(np.array_equal(old['clicks'],new['clicks']) and np.array_equal(old['errors'],new['errors']) and np.array_equal(old['statistics'][:,[0,1,2,5]],s[:,[0,1,2,5]]));enclosed=bool(np.all((b[:,:,0][good]<=p[good])&(p[good]<=b[:,:,1][good])));counter=bool(np.all(s[:,4]==s[:,8]+s[:,9]+s[:,10]) and np.all(s[:,6]==s[:,7]+s[:,8]+s[:,10]) and np.all((k==1).sum(1)==s[:,7]-s[:,9]) and np.all((k==2).sum(1)==s[:,4]));return dict(equal_decisions_errors_state_counters=identity,oracle_probabilities_enclosed=enclosed,counter_identities=counter,completed=int(complete.sum()),maximum_midpoint_difference=float(np.max(abs(p[good]-new['conditional'][good]),initial=0)),maximum_interval_width=float(np.max(b[:,:,1][good]-b[:,:,0][good],initial=0)),old_wide_calls=int(old['statistics'][:,4].sum()),new_wide_calls=int(s[:,4].sum()),ball_attempts=int(s[:,6].sum()),certified_decisions=int(s[:,7].sum()),ball_failures=int(s[:,10].sum()),ambiguous=int(s[:,8].sum()))
