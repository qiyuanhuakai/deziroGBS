"""Test input-only optical grouping against complete-output bias, fixed G0 prefix.
Order uses only occupied transmission rows. 4096 independent row subsequences
are generated for each budget, including all physical detectors and both sectors.
No failed-path deletion; this pilot is not a formal speed measurement.
"""
from pathlib import Path
import os,sys,json,hashlib,shutil,subprocess,ctypes as C,time,argparse,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_philox_click import NativePhiloxClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));dirs=[gcc.parent,Path(shutil.which('nvcc')).parent];handles=[os.add_dll_directory(str(d)) for d in dirs];src=ROOT/'native/gaussian_optical_order83_v1.cpp';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';d=np.load(inp);T,v,x,rr=d['T'],d['v'],d['x'],d['squeezing'];m,n=v.shape;weighted=np.ascontiguousarray(T*np.repeat(np.sqrt(np.sum(np.sinh(rr)**2,axis=1)),2));save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),src,ROOT/'native/gaussian_reset_cpu.cpp',inp]},count=4096,budgets=[4,8],pinned_prefix=8,seed=831000))
command=[str(gcc),'-O3','-std=c++20','-shared','-fopenmp','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(src),'-o',str(OUT/'order.dll')];r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);IP=C.POINTER(C.c_int);ptr=lambda a:a.ctypes.data_as(DP);lib=C.CDLL(str(OUT/'order.dll'));fn=lib.order83_optical;fn.argtypes=[C.c_int,C.c_int,C.c_int,DP,C.c_int,IP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;order=np.zeros(m,np.int32);stats=np.zeros(4);err=C.create_string_buffer(4096);start=time.perf_counter();assert fn(m,n,8,ptr(weighted),16,order.ctypes.data_as(IP),ptr(stats),err,len(err))==0,err.value;prepare=time.perf_counter()-start;assert np.array_equal(np.sort(order),np.arange(m)) and np.array_equal(order[:8],np.arange(8));np.savez_compressed(OUT/'order.npz',order=order,physical_order=d['order'][order],weighted_transmission=weighted);save('order.json',dict(seconds=prepare,near_collinear_pairs=stats[0],original_adjacent_similarity_sum=stats[1],reordered_adjacent_similarity_sum=stats[2],reordered_near_collinear_adjacencies=stats[3]));print('optical order',stats.tolist(),flush=True);rows=[];count=4096
for budget in (4,8):
 outputs=[];failed=0;start=time.perf_counter()
 for s in range(2):
  with NativePhiloxClickSampler(np.ascontiguousarray(v[order]),x[s],ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll',gpu_library=ROOT/'results/iteration74_fused_pilot_v2/philox.dll',branch_budget=budget,runtime_dirs=dirs) as model:
   bits=np.zeros((count,m),np.uint8);prob=np.zeros((count,m));errors=np.zeros((count,2),np.int32);diag=np.zeros((count,3));timings=np.zeros(7);err=C.create_string_buffer(4096);rc=model.gpu.gc73_sample(m,2*n,count,budget,831000+s,0,ptr(model._x),ptr(model._a),ptr(model._b),ptr(model.mean),ptr(model.noise_variance),bits.ctypes.data_as(BP),ptr(prob),errors.ctypes.data_as(IP),ptr(diag),ptr(timings),None,None,err,len(err))
  failed+=int(np.count_nonzero(errors[:,0]));outputs.append(bits);np.savez_compressed(OUT/f'sector_{budget}_{s}.npz',clicks=bits,errors=errors);save(f'sector_{budget}_{s}.json',dict(returncode=rc,error=err.value.decode(),failed_rows=int(np.count_nonzero(errors[:,0]))));assert rc==0 or np.count_nonzero(errors[:,0])>0,err.value
 row=dict(budget=budget,count=count,success=failed==0,failed_sector_rows=failed,seconds=time.perf_counter()-start)
 if not failed:
  bits=np.bitwise_or(*outputs);hist=np.bincount(bits.sum(1).astype(int),minlength=m+1);row.update(mean_clicks=float(bits.sum(1).mean()),variance=float(bits.sum(1).var(ddof=1)));restored=np.zeros_like(bits);restored[:,order]=bits;np.savez_compressed(OUT/f'combined_{budget}.npz',clicks=restored,total_histogram=hist)
 rows.append(row);save(f'row_{budget}.json',row);print(row,flush=True)
save('summary.json',dict(rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
