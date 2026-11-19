"""Native phase profile and replay of four frozen full-output B32 records.
Not a new sampling algorithm, nor a replicated performance comparison.
All heavy work is serial. Per-row phase durations are worker wall seconds.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse,shutil,subprocess,ctypes as C,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';src=ROOT/'native/gaussian_click_profile84_v1.cpp';gcc=Path(shutil.which('g++'));old=ROOT/'results/iteration83_weights_pilot_v1';files=[Path(__file__),inp,ROOT/'src/gbslab/native_stable_click.py',*[old/f'sector_0_32_{s}.npz' for s in range(2)],*[p for p in (ROOT/'native').iterdir() if p.suffix in ('.cpp','.cu','.hpp','.h')]];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},records=4,budget=32,seeds=[830200,830201],threads=4,phases=['conditional','branch_covariance','weight_normalization','moment_compression']))
command=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(src),'-lquadmath','-o',str(OUT/'profile.dll')];r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
data=np.load(inp);v,x=data['v'],data['x'];m,n=v.shape;DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);IP=C.POINTER(C.c_int);ptr=lambda x:x.ctypes.data_as(DP);rows=[]
for s in range(2):
 start=time.perf_counter()
 with NativeStableClickSampler(v,x[s],OUT/'profile.dll',branch_budget=32,runtime_dirs=[gcc.parent]) as model:
  fn=model.lib.tc84p_sample;fn.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,BP,DP,IP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;bits=np.zeros((4,m),np.uint8);prob=np.zeros((4,m));errors=np.zeros((4,2),np.int32);stats=np.zeros((4,10));err=C.create_string_buffer(4096);rc=fn(model.handle,4,830200+s,4,32,None,bits.ctypes.data_as(BP),ptr(prob),errors.ctypes.data_as(IP),ptr(stats),err,len(err))
 wall=time.perf_counter()-start;ref=np.load(old/f'sector_0_32_{s}.npz');np.savez_compressed(OUT/f'sector_{s}.npz',clicks=bits,conditional=prob,errors=errors,statistics=stats);phase=stats[:,6:].sum(0);row=dict(sector=s,success=rc==0 and not errors[:,0].any() and bool(np.all(stats[:,5]==m)),returncode=rc,error=err.value.decode(),seconds=wall,bits_equal=bool(np.array_equal(bits,ref['clicks'])),probabilities_equal=bool(np.array_equal(prob,ref['conditional'])),original_statistics_equal=bool(np.array_equal(stats[:,:6],ref['statistics'])),phase_worker_seconds=phase.tolist(),phase_fractions=(phase/phase.sum()).tolist());rows.append(row);save(f'row_{s}.json',row);print(row,flush=True)
save('summary.json',dict(rows=rows,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));assert all(r['success'] and r['bits_equal'] and r['probabilities_equal'] and r['original_statistics_equal'] for r in rows)
