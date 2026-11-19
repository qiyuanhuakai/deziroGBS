"""New independent complete records for the two routes on one lossy input.
256 B32 two-binary128 click records; 1024 retained-three photon-first records.
One heavy method at a time, no refills or failed-row deletion. Raw native outputs
and completion/error counters are archived even when a batch fails. Exploratory
wall times are not a replicated performance comparison.
"""
from pathlib import Path
import os,sys,argparse,json,hashlib,shutil,ctypes as C,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
from gbslab.native_polynomial_pnr import NativePolynomialPNRSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';weightlib=ROOT/'results/iteration83_weights_pilot_v1/weights.dll';pnrlib=ROOT/'results/iteration73_polynomial_pilot_v2/polynomial.dll';files=[Path(__file__),inp,weightlib,pnrlib,ROOT/'native/gaussian_click_weights83_v1.cpp',ROOT/'native/gaussian_pnr_polynomial73_v1.cpp',ROOT/'src/gbslab/native_stable_click.py',ROOT/'src/gbslab/native_polynomial_pnr.py'];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},panels=[dict(method='weights32',records=256,seed=832200),dict(method='pnr3',records=1024,seed=832100)],workers=16));d=np.load(inp);v,x=d['v'],d['x'];m,n=v.shape;dirs=[Path(shutil.which('g++')).parent];DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);IP=C.POINTER(C.c_int);UP=C.POINTER(C.c_uint32);ptr=lambda x:x.ctypes.data_as(DP);rows=[]
for method,count,seed in (('weights32',256,832200),('pnr3',1024,832100)):
 print('START',method,count,flush=True);start=time.perf_counter();outputs=[];sectors=[]
 for s in range(2):
  if method=='weights32':
   with NativeStableClickSampler(v,x[s],weightlib,branch_budget=32,runtime_dirs=dirs) as model:
    fn=model.lib.tc83w_sample;fn.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,BP,DP,IP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;bits=np.zeros((count,m),np.uint8);prob=np.zeros((count,m));errors=np.zeros((count,2),np.int32);stats=np.zeros((count,6));err=C.create_string_buffer(4096);rc=fn(model.handle,count,seed+s,16,32,None,bits.ctypes.data_as(BP),ptr(prob),errors.ctypes.data_as(IP),ptr(stats),err,len(err))
   success=rc==0 and not errors[:,0].any() and np.all(stats[:,5]==m);outputs.append(bits);np.savez_compressed(OUT/f'{method}_{s}.npz',clicks=bits,conditional=prob,errors=errors,statistics=stats);row=dict(sector=s,returncode=rc,error=err.value.decode(),success=bool(success),failed_rows=int(np.count_nonzero(errors[:,0])),completed_rows=int(np.count_nonzero(stats[:,5]==m)))
  else:
   with NativePolynomialPNRSampler(v,x[s],pnrlib,retained=3,workers=16,runtime_dirs=dirs) as model:
    counts=np.zeros((count,m),np.uint32);stats=np.zeros(9);err=C.create_string_buffer(4096);rc=model.lib.pnr74_sample(model.handle,count,seed+s,16,3,ptr(model.mean),ptr(model.noise_variance),counts.ctypes.data_as(UP),ptr(stats),err,len(err))
   success=rc==0 and stats[1]==count;outputs.append(counts);np.savez_compressed(OUT/f'{method}_{s}.npz',counts=counts,statistics=stats);row=dict(sector=s,returncode=rc,error=err.value.decode(),success=bool(success),completed_rows=int(stats[1]),statistics=stats.tolist())
  sectors.append(row);save(f'{method}_{s}.json',row);print(method,row,flush=True)
 row=dict(method=method,records=count,success=all(s['success'] for s in sectors),seconds=time.perf_counter()-start,sectors=sectors)
 if row['success']:
  if method=='weights32':bits=np.bitwise_or(*outputs);np.savez_compressed(OUT/f'{method}_combined.npz',clicks=bits,total_histogram=np.bincount(bits.sum(1).astype(int),minlength=m+1))
  else:
   counts=outputs[0].astype(np.uint64)+outputs[1];bits=counts>0;row['mean_total_photons']=float(counts.sum(1).mean());np.savez_compressed(OUT/f'{method}_combined.npz',counts=counts,clicks=bits,total_histogram=np.bincount(bits.sum(1).astype(int),minlength=m+1),photon_histogram=np.bincount(counts.sum(1).astype(int)))
  row['mean_clicks']=float(bits.sum(1).mean())
 rows.append(row);save(f'row_{method}.json',row);print('END',row,flush=True)
save('summary.json',dict(rows=rows,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
