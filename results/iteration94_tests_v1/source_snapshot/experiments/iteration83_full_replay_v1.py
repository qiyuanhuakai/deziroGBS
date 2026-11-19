"""CPU/GPU full 1152-output replay with identical native Philox uniforms.
Diagnose numerical drift separately from approximation bias and keep failures.
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
gcc=Path(shutil.which('g++'));dirs=[gcc.parent,Path(shutil.which('nvcc')).parent];handles=[os.add_dll_directory(str(d)) for d in dirs];src=ROOT/'native/gaussian_click_replay83_v1.cpp';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';data=np.load(inp);v,x=data['v'],data['x'];m,n=v.shape;save('plan.json',dict(scope=__doc__,count=16,budgets=[4,8],seed=830300,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),src,ROOT/'native/gaussian_click_stable82_v1.cpp',inp]}))
command=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(src),'-lquadmath','-o',str(OUT/'replay.dll')];r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);IP=C.POINTER(C.c_int);ptr=lambda x:x.ctypes.data_as(DP);rows=[]
for budget in (4,8):
 for s in range(2):
  with NativePhiloxClickSampler(v,x[s],ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll',gpu_library=ROOT/'results/iteration74_fused_pilot_v2/philox.dll',branch_budget=budget,runtime_dirs=dirs) as gpu:result=gpu.sample(16,830300+s,export_random=True)
  with NativeStableClickSampler(v,x[s],OUT/'replay.dll',branch_budget=budget,runtime_dirs=dirs) as cpu:
   fn=cpu.lib.tc83_replay;fn.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,DP,BP,DP,IP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;bits=np.zeros((16,m),np.uint8);prob=np.zeros((16,m));errors=np.zeros((16,2),np.int32);stats=np.zeros(2);err=C.create_string_buffer(4096);rc=fn(cpu.handle,16,16,budget,ptr(result['uniform_draws']),bits.ctypes.data_as(BP),ptr(prob),errors.ctypes.data_as(IP),ptr(stats),err,len(err))
  equal_rows=[];diffs=[]
  for i in range(16):
   if errors[i,0]:continue
   differing=np.flatnonzero(bits[i]!=result['clicks'][i]);end=int(differing[0])+1 if len(differing) else m;diffs.append(float(np.max(abs(prob[i,:end]-result['conditional_click_probabilities'][i,:end]))));equal_rows.append(not len(differing))
  row=dict(budget=budget,sector=s,returncode=rc,error=err.value.decode(),complete_rows=int(stats[0]),failed_rows=int(stats[1]),identical_complete_rows=sum(equal_rows),maximum_conditional_difference_through_first_divergence=max(diffs,default=None));rows.append(row);save(f'row_{budget}_{s}.json',row);np.savez_compressed(OUT/f'replay_{budget}_{s}.npz',cpu=bits,gpu=result['clicks'],cpu_conditional=prob,gpu_conditional=result['conditional_click_probabilities'],uniforms=result['uniform_draws'],errors=errors);print(row,flush=True)
save('summary.json',dict(rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
