"""Locate higher-order PNR approximation error without Monte Carlo histograms.
Factorial moments of the ideal normalized latent-pair model versus Wick moments;
bound the B8 low-total replacement contribution separately for every energy.
"""
from pathlib import Path
import os,sys,json,hashlib,shutil,subprocess,ctypes as C,time,argparse,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));src=ROOT/'native/gaussian_pair_moments83_v1.cpp';save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),src,ROOT/'native/gaussian_reset_cpu.cpp']},labels=['strong4','wide2','large128'],squeezing=[.3,.7,1.1],orders=list(range(1,9)),threads=16,ports='first min(8,m)'))
command=[str(gcc),'-O3','-std=c++20','-shared','-fopenmp','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(src),'-o',str(OUT/'moments.dll')];r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(OUT/'moments.dll'));DP=C.POINTER(C.c_double);ptr=lambda x:x.ctypes.data_as(DP);fn=lib.pair83_moments;fn.argtypes=[C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_double,DP,DP,DP,DP,C.c_int,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;rows=[]
for label in ('strong4','wide2','large128'):
 base=ROOT/'results/iteration81_factor_pilot_v2';d=np.load(base/f'input_{label}.npz');v,w,k=d['v'],d['weights'],d['K'];m,n=v.shape;L=len(w);depth=min(8,m);parts=[np.load(base/f'factors_{label}_{i}.npz') for i in range(0,L,256)];aa=np.concatenate([x['left'] for x in parts]);bb=np.concatenate([x['right'] for x in parts]);w=np.ascontiguousarray(w);k=np.ascontiguousarray(k)
 for squeeze in (.3,.7,1.1):
  t=float(np.tanh(squeeze));z=t*t;out=np.zeros((depth,8));stats=np.zeros(2);err=C.create_string_buffer(4096);start=time.perf_counter();assert fn(m,n,L,depth,8,z,ptr(w),ptr(k),ptr(aa),ptr(bb),16,ptr(out),ptr(stats),err,len(err))==0,err.value;elapsed=time.perf_counter()-start
  occupation=z/(1-z)*np.sum(abs(v[:depth])**2,axis=1);anomalous=t/(1-z)*np.sum(v[:depth]**2,axis=1);target=np.zeros_like(out)
  for i in range(depth):
   for q in range(1,9):target[i,q-1]=sum(math.factorial(q)**2/(math.factorial(q-2*j)*(2**j*math.factorial(j))**2)*occupation[i]**(q-2*j)*abs(anomalous[i])**(2*j) for j in range(q//2+1))
  # For 2J<=8, each factorial moment lies between zero and (2J)_q;
  # replacing either distribution can change it by at most that range.
  bound=np.zeros(8);mass=(1-z)**(n/2)
  for j in range(5):
   if j:mass*=z*(n/2+j-1)/j
   for q in range(1,9):
    if 2*j>=q:bound[q-1]+=mass*math.factorial(2*j)/math.factorial(2*j-q)
  assert np.max(abs(out[:,:2]/target[:,:2]-1))<1e-10
  relative=out/target-1;lower=np.maximum(0,abs(out-target)-bound[None,:]);row=dict(label=label,squeezing=squeeze,analysis_seconds=elapsed,maximum_absolute_relative_moment_errors=np.max(abs(relative),axis=0).tolist(),low_total_replacement_absolute_bounds=bound.tolist(),positive_error_lower_bounds_after_B8_correction=np.max(lower,axis=0).tolist(),scope='ideal factor model moments, not integer-plan TV');rows.append(row);save(f'row_{label}_{squeeze}.json',row);np.savez_compressed(OUT/f'moments_{label}_{squeeze}.npz',factor=out,quantum=target,relative=relative,low_bound=bound,positive_error_lower=lower);print(row,flush=True)
save('summary.json',dict(rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
