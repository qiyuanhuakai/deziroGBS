"""Fresh full-output quality records on the same lossy Gaussian input.
Separate complete datasets; all raw failed/completed rows retained, no refills.
Exploratory wall times exclude QR/core decomposition and are not formal timings.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse,shutil,ctypes as C,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
from gbslab.native_direct_click import NativeDirectClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';clicklib=ROOT/'results/iteration84_exactsum_pilot_v1/exactsum.dll';pnrbase=ROOT/'results/iteration84_correlated_pnr_pilot_v1';pnrlib=pnrbase/'correlated.dll';panels=[('pnr0',2048,841000),('exact32',512,842000),('pnr1',256,843000),('exact128',256,844000)];files=[Path(__file__),inp,clicklib,pnrlib,*[pnrbase/f'input_{s}.npz' for s in range(2)],ROOT/'native/gaussian_click_exactsum84_v1.cpp',ROOT/'native/gaussian_pnr_correlated84_v1.cpp',ROOT/'src/gbslab/native_direct_click.py',ROOT/'src/gbslab/native_stable_click.py'];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},panels=panels,threads=16,retained_photon_factors=3));data=np.load(inp);v,x=data['v'],data['x'];m,n=v.shape;dirs=[Path(shutil.which('g++')).parent];DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);UP=C.POINTER(C.c_uint32);IP=C.POINTER(C.c_int);ptr=lambda x:x.ctypes.data_as(DP);rows=[]
for method,count,seed in panels:
 print('START',method,count,flush=True);start=time.perf_counter();parts=[];sectors=[]
 for s in range(2):
  err=C.create_string_buffer(4096);errors=np.zeros((count,2),np.int32)
  if method.startswith('exact'):
   b=int(method[5:]);bits=np.zeros((count,m),np.uint8);prob=np.zeros((count,m));stats=np.zeros((count,8))
   with NativeStableClickSampler(v,x[s],clicklib,branch_budget=b,runtime_dirs=dirs) as model:
    fn=model.lib.tc84e_sample;fn.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,BP,DP,IP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;rc=fn(model.handle,count,seed+s,16,b,None,bits.ctypes.data_as(BP),ptr(prob),errors.ctypes.data_as(IP),ptr(stats),err,len(err))
   completed=int(np.count_nonzero(stats[:,5]==m));parts.append(bits);np.savez_compressed(OUT/f'{method}_{s}.npz',clicks=bits,conditional=prob,errors=errors,statistics=stats)
  else:
   d=np.load(pnrbase/f'input_{s}.npz');t=np.ascontiguousarray(d['t']);L=np.ascontiguousarray(d['noise_factor']);mean=np.zeros(2*n);xp=np.diag(np.r_[-t/(1+t),t/(1-t)]);counts=np.zeros((count,m),np.uint32);stats=np.zeros((count,12))
   with NativeDirectClickSampler(d['v'],xp,pnrlib,runtime_dirs=dirs) as model:
    fn=model.lib.pnr84_correlated_sample;fn.argtypes=[C.c_void_p,DP,DP,C.c_int,C.c_uint64,C.c_int,C.c_int,C.c_int,DP,DP,UP,DP,IP,C.c_char_p,C.c_int];fn.restype=C.c_int;rc=fn(model.handle,ptr(t),ptr(L),count,seed+s,16,3,int(method[-1]),ptr(mean),None,counts.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),err,len(err))
   completed=int(np.count_nonzero(stats[:,1]==1));parts.append(counts);np.savez_compressed(OUT/f'{method}_{s}.npz',counts=counts,errors=errors,statistics=stats)
  row=dict(sector=s,success=rc==0 and not errors[:,0].any() and completed==count,returncode=rc,error=err.value.decode(),completed=completed,failed_rows=int(np.count_nonzero(errors[:,0])));sectors.append(row);save(f'{method}_{s}.json',row);print(method,row,flush=True)
 row=dict(method=method,records=count,success=all(r['success'] for r in sectors),seconds=time.perf_counter()-start,sectors=sectors)
 if row['success']:
  if method.startswith('exact'):bits=np.bitwise_or(*parts);np.savez_compressed(OUT/f'{method}_combined.npz',clicks=bits,total_histogram=np.bincount(bits.sum(1).astype(int),minlength=m+1))
  else:
   counts=parts[0].astype(np.uint64)+parts[1];bits=counts>0;row['mean_photons']=float(counts.sum(1).mean());np.savez_compressed(OUT/f'{method}_combined.npz',counts=counts,clicks=bits,total_histogram=np.bincount(bits.sum(1).astype(int),minlength=m+1),photon_histogram=np.bincount(counts.sum(1).astype(int)))
  row['mean_clicks']=float(bits.sum(1).mean())
 rows.append(row);save(f'row_{method}.json',row);print('END',row,flush=True)
save('summary.json',dict(rows=rows,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
