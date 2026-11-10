from pathlib import Path
import os,sys,ctypes as C,subprocess,hashlib,json,shutil,time,traceback
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
from scipy.optimize import brentq
OUT=ROOT/'.git/iteration69_reset_blas_v5_20261110';OUT.mkdir(exist_ok=False)
def H(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(n,v):
 p=OUT/n;assert not p.exists();p.write_bytes((json.dumps(v,indent=2)+'\n').encode())
source=ROOT/'.git/iteration69_reset_blas_v5.cpp';compiler=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'
sources=[Path(__file__),source];hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources}
for p in sources:shutil.copy2(p,OUT/p.name)
header_hashes={p.relative_to(eigen).as_posix():H(p) for p in (eigen/'Eigen').rglob('*') if p.is_file()};save('eigen_headers.json',header_hashes)
command=[str(compiler),'-O3','-march=native','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(source),'-o',str(OUT/'native.dll')]
save('plan.json',dict(source_hashes=hashes,compiler_sha256=H(compiler),command=command,seed=690101,scope='First native centered Gaussian reset prototype. Three detector selectors target the same approximate PNR law; no exact Gaussian-target sampling claim. Full source and all failures retained.'))
t=time.perf_counter();build=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=build.returncode,stdout=build.stdout,stderr=build.stderr,seconds=time.perf_counter()-t));print('build',build.returncode,flush=True);assert build.returncode==0
runtime=os.add_dll_directory(str(compiler.parent));lib=C.CDLL(str(OUT/'native.dll'));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32)
lib.gr_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];lib.gr_create.restype=C.c_void_p;lib.gr_destroy.argtypes=[C.c_void_p]
lib.gr_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,UP,DP,C.c_char_p,C.c_int];lib.gr_probe.argtypes=[C.c_void_p,DP,C.c_double,C.c_int,DP,DP,DP,C.c_char_p,C.c_int]
precision=np.zeros(5);lib.gr_precision.argtypes=[DP];lib.gr_precision(precision.ctypes.data_as(DP));save('precision.json',dict(control_before=int(precision[0]),residual_before=precision[1],control_inside=int(precision[2]),residual_inside=precision[3],control_after=int(precision[4])));print('precision',precision,flush=True)
blas_path=next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*.dll'));blas=C.CDLL(str(blas_path));lib.gr_set_gemm.argtypes=[C.c_void_p,C.c_void_p];save('blas.json',dict(path=str(blas_path),sha256=H(blas_path),threads=1))
def isometry(m,n,rng):
 q,r=np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)),mode='reduced');return np.ascontiguousarray(q*(np.diag(r)/abs(np.diag(r)))[None,:])
def initial(n,r,eta):
 number=np.sinh(r)**2;pair=np.sinh(r)*np.cosh(r);return np.diag(np.r_[np.repeat(eta*(number+pair),n),np.repeat(eta*(number-pair),n)])
rows=[];rng=np.random.default_rng(690101)
for case,(m,n,r,eta,count) in enumerate([(4,2,.7,.1,20000),(4,2,.7,.65,20000),(1024,8,.7,.65,128),(8176,8,.7,.65,128),(8176,32,.7,.65,64),(8176,64,.7,.65,32)]):
 v=isometry(m,n,rng);x0=np.ascontiguousarray(initial(n,r,eta));error=C.create_string_buffer(1024);handle=lib.gr_create(m,n,v.ctypes.data_as(DP),x0.ctypes.data_as(DP),error,1024);assert handle,error.value;lib.gr_set_gemm(handle,C.cast(blas.scipy_cblas_zgemm64_,C.c_void_p));np.savez_compressed(OUT/f'input_{case}.npz',v=v,x0=x0)
 # Three forced transitions independently reconstruct full physical K rates.
 x=x0.copy();probes=[]
 for step in range(3):
  eigenvalues,basis=np.linalg.eigh(x);p0=float(np.exp(-.5*np.log1p(eigenvalues).sum()));waiting=.5*(1+p0);s=brentq(lambda s:-.5*np.log1p(s*eigenvalues).sum()-np.log(waiting),0.,1.,xtol=1e-14);damped=(basis*((1-s)*eigenvalues/(1+s*eigenvalues)))@basis.T;damped=(damped+damped.T)/2
  o=np.block([[v.real,-v.imag],[v.imag,v.real]]);ox=o[:m];op=o[m:];expected_rates=.5*(np.einsum('ij,ij->i',ox@damped,ox)+np.einsum('ij,ij->i',op@damped,op));forced=int(np.argmax(expected_rates));a=damped@ox[forced];b=damped@op[forced];expected=damped+(np.outer(a,a)+np.outer(b,b))/expected_rates[forced]
  actual=np.empty_like(x);rates=np.empty(m);info=np.zeros(5);rc=lib.gr_probe(handle,x.ctypes.data_as(DP),waiting,forced,actual.ctypes.data_as(DP),rates.ctypes.data_as(DP),info.ctypes.data_as(DP),error,1024)
  probe=dict(step=step,returncode=rc,error=error.value.decode(),time_error=float(abs(info[2]-s)),rate_error=float(np.max(abs(rates-expected_rates))) if rc==0 else None,state_error=float(np.max(abs(actual-expected))) if rc==0 else None)
  probe['passed']=rc==0 and max(probe['time_error'],probe['rate_error'],probe['state_error'])<1e-10;probes.append(probe)
  np.savez_compressed(OUT/f'probe_{case}_{step}.npz',before=x,after=actual,expected=expected,rates=rates,expected_rates=expected_rates,info=info,waiting=waiting,forced=forced)
  if not probe['passed']:break
  x=np.ascontiguousarray(actual)
 save(f'probes_{case}.json',probes);print('probes',case,all(p['passed'] for p in probes),flush=True)
 for method in (1,3,4):
  if m>100 and method==0:continue
  output=np.zeros((count,m),np.uint32);stats=np.zeros(7);start=time.perf_counter();rc=lib.gr_sample(handle,count,690200+case,method,output.ctypes.data_as(UP),stats.ctypes.data_as(DP),error,1024);seconds=time.perf_counter()-start
  row=dict(case=case,modes=m,inputs=n,r=r,eta=eta,samples=count,method=method,seconds=seconds,returncode=rc,error=error.value.decode(),photons=stats[0],trials=stats[1],flow_seconds=stats[2],selection_seconds=stats[3],reset_seconds=stats[4],zero_samples=stats[5],maximum_photons=stats[6]);row['passed']=rc==0 and all(p['passed'] for p in probes)
  rows.append(row);save(f'case_{case}_method_{method}.json',row);np.savez_compressed(OUT/f'case_{case}_method_{method}.npz',counts=output,stats=stats);print(row,flush=True)
 lib.gr_destroy(handle)
save('summary.json',dict(passed=all(r['passed'] for r in rows),rows=rows,source_hashes=hashes,dll_sha256=H(OUT/'native.dll')))
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
