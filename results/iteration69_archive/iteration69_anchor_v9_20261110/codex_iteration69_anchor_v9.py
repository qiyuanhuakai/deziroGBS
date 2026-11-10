from pathlib import Path
import os, sys, ctypes as C, subprocess, hashlib, json, shutil, time, math
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
from scipy.optimize import brentq
OUT=ROOT/'.git/iteration69_anchor_v9_20261110'; OUT.mkdir(exist_ok=False)
OLD=ROOT/'.git/iteration69_quality_pilot_20261110'
def H(p):
    with p.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def save(name,value):
    p=OUT/name; assert not p.exists(); p.write_bytes((json.dumps(value,indent=2)+'\n').encode())
sources=[Path(__file__),ROOT/'.git/iteration69_anchor_v9.cpp',ROOT/'.git/iteration69_displaced_v6.cpp',ROOT/'.git/iteration69_reset_blas_v5.cpp']
for p in sources: shutil.copy2(p,OUT/p.name)
gcc=Path(shutil.which('g++')); eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'
count=50000; radius=math.sqrt((16*math.log(2)+math.log(4/.01))/(2*count))
command=[str(gcc),'-O3','-march=native','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(sources[1]),'-o',str(OUT/'native.dll')]
save('plan.json',dict(source_hashes={p.name:H(p) for p in sources},compiler_sha256=H(gcc),command=command,seed=690701,samples_per_method=count,methods=[5],radius=radius,dependencies={p.relative_to(ROOT).as_posix():H(p) for p in [OLD/'summary.json']+[OLD/f'input_{i}.npz' for i in range(4)]},scope='Coherent-reference rejection envelope on prior-art displaced Gaussian reset S62--S65 plus occupied-subspace diagonal rejection. Compare full threshold coarsening on the SAME four inputs as centered and official Frugal. Fixed seeded PRNG/normal-generator experiments; nominal joint 99% ideal-IID bands, not exact-law certification. No photon/proposal cap.'))
save('eigen_headers.json',{p.relative_to(eigen).as_posix():H(p) for p in (eigen/'Eigen').rglob('*') if p.is_file()})
start=time.perf_counter(); build=subprocess.run(command,capture_output=True,text=True,errors='replace'); save('build.json',dict(returncode=build.returncode,stdout=build.stdout,stderr=build.stderr,seconds=time.perf_counter()-start)); print('build',build.returncode,flush=True); assert build.returncode==0
runtime=os.add_dll_directory(str(gcc.parent)); lib=C.CDLL(str(OUT/'native.dll')); DP=C.POINTER(C.c_double); UP=C.POINTER(C.c_uint32)
def ptr(a): return a.ctypes.data_as(DP)
lib.gr_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int]; lib.gr_create.restype=C.c_void_p
lib.gr_destroy.argtypes=[C.c_void_p]; lib.gr_precision.argtypes=[DP]
lib.ds_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int]
lib.ds_probe.argtypes=[C.c_void_p,DP,DP,C.c_double,C.c_int,DP,DP,DP,DP,C.c_char_p,C.c_int]
lib.ds_sample_anchor.argtypes=[C.c_void_p,C.c_int,C.c_uint64,DP,DP,UP,DP,C.c_char_p,C.c_int]
error=C.create_string_buffer(4096); checks=[]; rows=[]
def precision():
    data=np.zeros(5);lib.gr_precision(ptr(data));return data
before=precision(); save('precision.json',before.tolist())
for case,(r,eta) in enumerate([(.7,.1),(.7,.3),(.7,.65),(1.1,.65)]):
    source=np.load(OLD/f'input_{case}.npz'); v=np.ascontiguousarray(source['v']); target=source['target']; original=source['x0']; m,n=v.shape
    vp=.5*(eta*np.exp(2*r)+1-eta); vm=.5*(eta*np.exp(-2*r)+1-eta); core_r=-.5*np.log(2*vm); variance=np.r_[np.repeat(vp-.5*np.exp(2*core_r),n),np.zeros(n)]; mean=np.zeros(2*n)
    x0=np.ascontiguousarray(np.diag(np.r_[np.repeat(.5*np.expm1(2*core_r),n),np.repeat(.5*np.expm1(-2*core_r),n)])); decomposition_error=float(np.max(abs(x0+np.diag(variance)-original)));assert decomposition_error<1e-14 and min(variance)>=0
    handle=lib.gr_create(m,n,ptr(v),ptr(x0),error,len(error)); assert handle,error.value
    np.savez_compressed(OUT/f'input_{case}.npz',v=v,x0=x0,noise_variance=variance,mean=mean,target=target,core_r=core_r)
    # Independent reference works in the complete physical quadrature space.
    o=np.block([[v.real,-v.imag],[v.imag,v.real]]); x=x0.copy();w=np.random.default_rng(690701+case).normal(size=2*n)*np.sqrt(variance);probes=[]
    for step in range(5):
        k=o@x@o.T; physical_mean=o@w; eye=np.eye(2*m)
        def logsurvival(s):
            a=eye+s*k;return -.5*np.linalg.slogdet(a)[1]-.5*s*physical_mean@np.linalg.solve(a,physical_mean)
        p0=np.exp(logsurvival(1.));waiting=.5*(1+p0);s=brentq(lambda t:logsurvival(t)-np.log(waiting),0.,1.,xtol=1e-14)
        a=eye+s*k;kt=(1-s)*np.linalg.solve(a,k);mt=np.sqrt(1-s)*np.linalg.solve(a,physical_mean);rate=.5*(np.diag(kt)[:m]+np.diag(kt)[m:]+mt[:m]**2+mt[m:]**2);forced=int(np.argmax(rate));columns=kt[:,[forced,m+forced]];increment=columns@mt[[forced,m+forced]]/rate[forced];expected_k=kt+columns@columns.T/rate[forced]-np.outer(increment,increment);expected_m=mt+increment
        actual=np.zeros_like(x);actual_w=np.zeros_like(w);rates=np.zeros(m);info=np.zeros(3);rc=lib.ds_probe(handle,ptr(x),ptr(w),waiting,forced,ptr(actual),ptr(actual_w),ptr(rates),ptr(info),error,len(error))
        check=dict(case=case,step=step,returncode=rc,error=error.value.decode(),time_error=float(abs(info[2]-s)),rate_error=float(np.max(abs(rates-rate))),covariance_error=float(np.max(abs(o@actual@o.T-expected_k))),mean_error=float(np.max(abs(o@actual_w-expected_m))))
        check['passed']=rc==0 and max(check[k] for k in ['time_error','rate_error','covariance_error','mean_error'])<1e-10;probes.append(check);assert check['passed'],check
        np.savez_compressed(OUT/f'probe_{case}_{step}.npz',before_x=x,before_w=w,after_x=actual,after_w=actual_w,expected_physical_k=expected_k,expected_physical_mean=expected_m,rates=rates,expected_rates=rate,waiting=waiting,forced=forced)
        x=actual;w=actual_w
    save(f'probes_{case}.json',probes);checks+=probes
    for method in (5,):
        counts=np.zeros((count,m),np.uint32);stats=np.zeros(8);start=time.perf_counter();rc=lib.ds_sample_anchor(handle,count,690801+case,ptr(mean),ptr(variance),counts.ctypes.data_as(UP),ptr(stats),error,len(error));seconds=time.perf_counter()-start
        row=dict(case=case,r=r,eta=eta,method=method,returncode=rc,error=error.value.decode(),seconds=seconds,decomposition_error=decomposition_error)
        np.savez_compressed(OUT/f'case_{case}_method_{method}.npz',counts=counts,stats=stats)
        if rc==0:
            reference=np.load(ROOT/'.git/iteration69_anchor_v8_20261110'/f'case_{case}_method_{method}.npz')['counts'][:count]
            row['counts_equal_v8']=bool(np.array_equal(counts,reference));assert row['counts_equal_v8']
            mask=np.sum((counts>0)*(1<<np.arange(m)),axis=1);hist=np.bincount(mask,minlength=1<<m);tv=float(.5*np.sum(abs(hist/count-target)));row.update(empirical_threshold_tv=tv,nominal_tv_interval=[max(0,tv-radius),min(1,tv+radius)],mean_photons=float(counts.sum(axis=1).mean()),target_mean_photons=float(n*eta*np.sinh(r)**2),maximum_photons=int(counts.sum(axis=1).max()),proposals_per_photon=float(stats[1]/stats[0]),vacuum_frequency=float(hist[0]/count));np.savez_compressed(OUT/f'histogram_{case}_{method}.npz',histogram=hist,target=target)
        rows.append(row);save(f'case_{case}_method_{method}.json',row);print(row,flush=True)
    lib.gr_destroy(handle)
# Exact special cases: vacuum and coherent light. No Gaussian reset error here.
for label,mean in [('vacuum',np.zeros(4)),('coherent',np.array([1.,.7,-.2,.4]))]:
    v=np.eye(2,dtype=np.complex128);x0=np.zeros((4,4));variance=np.zeros(4);handle=lib.gr_create(2,2,ptr(v),ptr(x0),error,len(error));assert handle,error.value
    counts=np.zeros((100000,2),np.uint32);stats=np.zeros(8);rc=lib.ds_sample_anchor(handle,len(counts),690901,ptr(mean),ptr(variance),counts.ctypes.data_as(UP),ptr(stats),error,len(error));assert rc==0,error.value
    expected=(mean[:2]**2+mean[2:]**2)/2;observed=counts.mean(axis=0);tolerance=7*np.sqrt(expected/len(counts))+1e-12;passed=bool(np.all(abs(observed-expected)<=tolerance));assert passed
    row=dict(case=label,expected_means=expected.tolist(),observed_means=observed.tolist(),seven_sigma_tolerance=tolerance.tolist(),passed=passed);save(f'{label}.json',row);np.savez_compressed(OUT/f'{label}.npz',counts=counts,stats=stats);checks.append(row);lib.gr_destroy(handle)
after=precision();assert before[0]==after[0]==before[4]==after[4];save('precision_after.json',after.tolist())
save('summary.json',dict(passed=all(r['returncode']==0 for r in rows) and all(c['passed'] for c in checks),rows=rows,checks=checks,nominal_radius=radius,dll_sha256=H(OUT/'native.dll')))
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
