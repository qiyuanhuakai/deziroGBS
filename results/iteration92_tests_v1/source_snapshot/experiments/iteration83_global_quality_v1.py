"""Full-output quality: exact Gaussian-OR click moments and 4096 GPU records.
Keep all failed rows and forbid distributional TV claims for a failed batch.
Use simultaneous DKW-Cantelli witnesses from complete click counts, separately
from first-eight marginal TV. Finite precision is checked, not interval-certified.
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
gcc=Path(shutil.which('g++'));dirs=[gcc.parent,Path(shutil.which('nvcc')).parent];handles=[os.add_dll_directory(str(d)) for d in dirs];src=ROOT/'native/gaussian_click_moments83_v1.cpp';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';data=np.load(inp);v,x=data['v'],data['x'];m,n=v.shape;covs=np.ascontiguousarray([np.load(ROOT/f'results/iteration83_full_quality_v1/covariance_{s}.npz')['cov'] for s in range(2)]);save('plan.json',dict(scope=__doc__,records=4096,budgets=[4,8],seed=830300,delta_per_method=.005,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),src,ROOT/'native/gaussian_reset_cpu.cpp',inp]}))
command=[str(gcc),'-O3','-std=c++20','-shared','-fopenmp','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(src),'-o',str(OUT/'moments.dll')];r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);IP=C.POINTER(C.c_int);ptr=lambda x:x.ctypes.data_as(DP);lib=C.CDLL(str(OUT/'moments.dll'));fn=lib.click83_moments;fn.argtypes=[C.c_int,C.c_int,DP,C.c_int,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;single=np.zeros(m);stats=np.zeros(4);err=C.create_string_buffer(4096);start=time.perf_counter();assert fn(m,2,ptr(covs),16,ptr(single),ptr(stats),err,len(err))==0,err.value;mean,var=stats[:2];save('reference_moments.json',dict(mean=mean,variance=var,statistics=stats.tolist(),seconds=time.perf_counter()-start));np.savez_compressed(OUT/'reference_singles.npz',click=single);print('reference moments',stats.tolist(),flush=True);del covs
rows=[];count=4096
for budget in (4,8):
 outputs=[];failed=0;start=time.perf_counter()
 for s in range(2):
  with NativePhiloxClickSampler(v,x[s],ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll',gpu_library=ROOT/'results/iteration74_fused_pilot_v2/philox.dll',branch_budget=budget,runtime_dirs=dirs) as model:
   bits=np.zeros((count,m),np.uint8);prob=np.zeros((count,m));errors=np.zeros((count,2),np.int32);diag=np.zeros((count,3));timings=np.zeros(7);err=C.create_string_buffer(4096);rc=model.gpu.gc73_sample(m,2*n,count,budget,830300+s,0,ptr(model._x),ptr(model._a),ptr(model._b),ptr(model.mean),ptr(model.noise_variance),bits.ctypes.data_as(BP),ptr(prob),errors.ctypes.data_as(IP),ptr(diag),ptr(timings),None,None,err,len(err))
  failed+=int(np.count_nonzero(errors[:,0]));outputs.append(bits);np.savez_compressed(OUT/f'sector_{budget}_{s}.npz',clicks=bits,errors=errors,diagnostics=diag);save(f'sector_{budget}_{s}.json',dict(returncode=rc,error=err.value.decode(),failed_rows=int(np.count_nonzero(errors[:,0])),native_timings=timings.tolist()));assert rc==0 or np.count_nonzero(errors[:,0])>0,err.value
 row=dict(budget=budget,records=count,failed_sector_rows=failed,success=failed==0,seconds=time.perf_counter()-start)
 if not failed:
  bits=np.bitwise_or(*outputs);totals=bits.sum(1);hist=np.bincount(totals.astype(int),minlength=m+1);cdf=np.cumsum(hist)/count;radius=math.sqrt(math.log(2/.005)/(2*count));lower=np.zeros(m+1)
  for k in range(m+1):
   if k<mean:lower[k]=max(0,cdf[k]-radius-var/(var+(mean-k)**2))
   elif k+1>mean:lower[k]=max(0,1-cdf[k]-radius-var/(var+(k+1-mean)**2))
  j=int(np.argmax(lower));row.update(mean_clicks=float(totals.mean()),sample_variance=float(totals.var(ddof=1)),mean_error=float(totals.mean()-mean),dkw_radius=radius,nominal_joint99_full_tv_lower_bound=float(lower[j]),witness_threshold=j,witness_direction='at_most' if j<mean else 'greater_than',witness_empirical_probability=float(cdf[j] if j<mean else 1-cdf[j]));np.savez_compressed(OUT/f'combined_{budget}.npz',clicks=bits,total_histogram=hist,witness_lower=lower)
 rows.append(row);save(f'row_{budget}.json',row);print(row,flush=True)
save('summary.json',dict(rows=rows,reference_mean=float(mean),reference_variance=float(var),confidence_scope='joint 99% over two methods and all thresholds, conditional on iid generated records and fixed numerical reference moments'));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
