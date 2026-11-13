"""Native fixed-passive-basis convex core optimization, independent reference join.
Preserves the Gaussian covariance with full correlated classical displacement.
Energy and duality are numerical checks, not directed-rounding certification.
"""
from pathlib import Path
import os,argparse,json,hashlib,shutil,ctypes as C,subprocess,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
src=ROOT/'native/gaussian_core_energy83_v1.cpp';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';gcc=Path(shutil.which('g++'));save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),src,ROOT/'native/gaussian_reset_cpu.cpp',inp]},reference='iteration83_core_energy_probe_v2'))
command=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(src),'-o',str(OUT/'core.dll')];r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(OUT/'core.dll'));DP=C.POINTER(C.c_double);ptr=lambda x:x.ctypes.data_as(DP);fn=lib.core83_optimize;fn.argtypes=[C.c_int,DP,DP,DP,DP,DP,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;rows=[];xs=np.load(inp)['x']
for sector,x in enumerate(xs):
 old=np.load(ROOT/f'results/iteration83_core_budget_v2/split_{sector}.npz');R=np.ascontiguousarray(old['rotation']);n=len(old['t']);t=np.zeros(n);noise=np.zeros_like(x);dual=np.zeros_like(x);stats=np.zeros(8);err=C.create_string_buffer(4096);start=time.perf_counter();rc=fn(n,ptr(x),ptr(R),ptr(old['t']),ptr(t),ptr(noise),ptr(dual),ptr(stats),err,len(err));elapsed=time.perf_counter()-start;assert rc==0,err.value
 pure=R@np.diag(np.r_[(1-t)/(1+t),(1+t)/(1-t)])@R.T;reference=np.load(ROOT/f'results/iteration83_core_energy_probe_v2/split_{sector}.npz');energy=float((np.trace(pure)-2*n)/4);refenergy=float((np.trace(reference['pure'])-2*n)/4);res=float(abs(pure+2*noise-(np.eye(2*n)+2*x)).max());mineig=float(np.linalg.eigvalsh(noise).min());check=bool(abs(energy-refenergy)<1e-6 and res<1e-11 and mineig>0 and stats[3]<1e-5);row=dict(sector=sector,passed=check,seconds=elapsed,original_core_mean=stats[0],optimized_core_mean=stats[1],dual_core_lower=stats[2],numerical_duality_gap=stats[3],minimum_noise_eigenvalue=mineig,reference_energy_difference=abs(energy-refenergy),covariance_reconstruction_error=res,statistics=stats.tolist());rows.append(row);save(f'row_{sector}.json',row);np.savez_compressed(OUT/f'split_{sector}.npz',rotation=R,t=t,pure=pure,noise=noise,dual=dual);print(row,flush=True);assert check
save('summary.json',dict(passed=all(r['passed'] for r in rows),rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
