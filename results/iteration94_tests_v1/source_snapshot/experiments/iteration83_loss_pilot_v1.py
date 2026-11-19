"""Native lossy-map reduction and first complete 1152-detector two-sector pilot.
G0 detectors are placed first for an exactly enumerable same-order marginal.
This is an explicit physical model, not a claim of full experimental accuracy.
"""
from pathlib import Path
import os,sys,json,hashlib,shutil,subprocess,ctypes as C,time,argparse
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
from gbslab.native_philox_click import NativePhiloxClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));dirs=[gcc.parent,Path(shutil.which('nvcc')).parent];runtime=[os.add_dll_directory(str(d)) for d in dirs]
source=ROOT/'native/gaussian_loss_map83_v1.cpp';mtx=ROOT/'results/iteration81_jiuzhang_extract_v1/extracted_0/jiuzhang3/matrix/high power.mtx';prior=ROOT/'results/iteration81_jiuzhang_accuracy_v3/input_0.npz';base=np.load(prior);rr=base['squeezing'];raw=np.loadtxt(mtx,skiprows=3);T=(raw[:,0]+1j*raw[:,1]).reshape(1152,50,order='F');front=base['ports'].tolist();order=np.array(front+[i for i in range(1152) if i not in front]);T=np.ascontiguousarray(T[order]);n=50;m=1152
xs=[]
for s in range(2):
 nn=np.diag(np.repeat(np.sinh(rr[:,s])**2,2));mm=np.zeros((n,n));mm[np.arange(n),np.arange(n)^1]=np.repeat(np.sinh(rr[:,s])*np.cosh(rr[:,s]),2);xs.append(np.block([[nn+mm,np.zeros((n,n))],[np.zeros((n,n)),nn-mm]]))
xs=np.ascontiguousarray(xs);save('plan.json',dict(scope=__doc__,hashes={str(p.relative_to(ROOT)):H(p) for p in [Path(__file__),source,ROOT/'native/gaussian_reset_cpu.cpp',mtx,prior]},threads=16,count=16,order=order.tolist(),budgets=[4,8],timing='exploratory; includes prepare separately; not formal speed claim'))
command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(source),'-o',str(OUT/'loss.dll')];r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
lib=C.CDLL(str(OUT/'loss.dll'));DP=C.POINTER(C.c_double);ptr=lambda x:x.ctypes.data_as(DP);fn=lib.loss83_prepare;fn.argtypes=[C.c_int,C.c_int,C.c_int,DP,DP,DP,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;v=np.zeros_like(T);ys=np.zeros_like(xs);stats=np.zeros(5);err=C.create_string_buffer(4096);start=time.perf_counter();rc=fn(m,n,2,ptr(T),ptr(xs),ptr(v),ptr(ys),ptr(stats),err,len(err));seconds=time.perf_counter()-start;save('prepare.json',dict(returncode=rc,error=err.value.decode(),seconds=seconds,stats=stats.tolist()));assert rc==0,err.value
reconstruction=[]
for s in range(2):
 a=np.block([[v[:8].real,-v[:8].imag],[v[:8].imag,v[:8].real]]);cov=np.eye(16)+2*a@ys[s]@a.T;reconstruction.append(float(np.max(abs(cov-base['covariances'][s]))));assert reconstruction[-1]<1e-12
np.savez_compressed(OUT/'input.npz',T=T,v=v,x=ys,input_x=xs,order=order,squeezing=rr);save('reconstruction.json',dict(maximum_covariance_differences=reconstruction));print('native loss reduction',seconds,stats.tolist(),reconstruction,flush=True)
rows=[]
for method,budget in [('gpu',4),('stable',4),('gpu',8),('stable',8)]:
 outputs=[];row=dict(method=method,budget=budget,count=16,success=False);start=time.perf_counter()
 try:
  for s in range(2):
   if method=='stable':model=NativeStableClickSampler(v,ys[s],ROOT/'results/iteration82_stable_pilot_v1/stable.dll',branch_budget=budget,workers=16,runtime_dirs=dirs)
   else:model=NativePhiloxClickSampler(v,ys[s],ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll',gpu_library=ROOT/'results/iteration74_fused_pilot_v2/philox.dll',branch_budget=budget,runtime_dirs=dirs)
   with model:result=model.sample(16,830001+s)
   outputs.append(result['clicks']);save(f'sector_{method}_{budget}_{s}.json',dict(statistics=result['statistics'],wall_seconds=result['wall_seconds']));np.savez_compressed(OUT/f'sector_{method}_{budget}_{s}.npz',clicks=result['clicks'])
  bits=np.bitwise_or(*outputs);np.savez_compressed(OUT/f'samples_{method}_{budget}.npz',clicks=bits);row.update(success=True,mean_clicks=float(bits.sum(1).mean()),shape=list(bits.shape))
 except Exception as e:row['error']=str(e)
 row['seconds']=time.perf_counter()-start;rows.append(row);save(f'row_{method}_{budget}.json',row);print(row,flush=True)
save('summary.json',dict(rows=rows,prepare_seconds=seconds,reconstruction=reconstruction));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
