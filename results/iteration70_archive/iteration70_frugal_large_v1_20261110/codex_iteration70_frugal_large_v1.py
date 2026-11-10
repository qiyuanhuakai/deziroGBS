from pathlib import Path
import os,ctypes as C,hashlib,json,shutil,subprocess,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
OUT=ROOT/'.git/iteration70_frugal_large_v1_20261110';OUT.mkdir(exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    p=OUT/name;assert not p.exists();p.write_bytes((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';flags=['-O3','-std=c++20','-ffp-contract=off','-fno-fast-math']
meta=json.loads((ROOT/'results/iteration39_build/build.json').read_text());fc=meta['command'][:];fc[1:1]=['-ffp-contract=off','-fno-fast-math'];fc[-1]=str(OUT/'frugal.exe')
nc=[str(gcc)]+flags+['-shared','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(ROOT/'.git/iteration70_parallel_v1.cpp'),'-o',str(OUT/'native.dll')]
sources=[Path(__file__),ROOT/'.git/iteration70_parallel_v1.cpp',ROOT/'.git/iteration70_flow_v2.cpp',ROOT/'native/gaussian_reset_cpu.cpp',ROOT/'experiments/frugal_extended.cpp']
for p in sources:shutil.copy2(p,OUT/p.name)
save('plan.json',dict(native_command=nc,frugal_command=fc,source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},upstream_commit=meta['upstream_commit'],common_flags=flags,modes=144,inputs=50,r=1.4,eta=.43,matrix_seed=705001,samples=512,repetitions=3,threads=[1,16],methods=['native_newton','frugal_K2','frugal_K3'],scope='Fresh shared synthetic covariance. Same host and CPU worker budget, generic x86-64 compilation settings; native full detector scan calls separately identified single-thread OpenBLAS. Native PNR records thresholded before comparing clicked output. All output records retained. Preparation, sampling, output summaries and process I/O reported separately. Neither large-system law is assumed accurate or equivalent; throughput ratios do not establish equal-quality superiority.'))
for name,command in [('frugal',fc),('native',nc)]:
    start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save(f'{name}_build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start));print(name,'build',r.returncode,flush=True);assert r.returncode==0
rng=np.random.default_rng(705001);a=rng.normal(size=(144,50))+1j*rng.normal(size=(144,50));v,rr=np.linalg.qr(a);v=np.ascontiguousarray(v*(np.diag(rr)/abs(np.diag(rr))));r=1.4;eta=.43;vp=.5*(eta*np.exp(2*r)+1-eta);vm=.5*(eta*np.exp(-2*r)+1-eta);core=-.5*np.log(2*vm);x=np.diag([.5*np.expm1(2*core)]*50+[.5*np.expm1(-2*core)]*50);variance=np.r_[np.full(50,vp-.5*np.exp(2*core)),np.zeros(50)];mean=np.zeros(100);o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.eye(288)+2*o@(x+np.diag(variance))@o.T
np.savez_compressed(OUT/'input.npz',v=v,x0=x,variance=variance,mean=mean,covariance=cov);np.savetxt(OUT/'covariance.csv',cov,delimiter=',',fmt='%.17g')
runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(OUT/'native.dll'));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32)
def ptr(a):return a.ctypes.data_as(DP)
lib.gr_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];lib.gr_create.restype=C.c_void_p;lib.gr_destroy.argtypes=[C.c_void_p];lib.gr_set_gemm.argtypes=[C.c_void_p,C.c_void_p];lib.g70_sample_parallel.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,C.c_void_p,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int]
bp=next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*.dll'));blas=C.CDLL(str(bp));blas.scipy_openblas_set_num_threads64_.argtypes=[C.c_int];blas.scipy_openblas_set_num_threads64_(1);save('blas.json',dict(path=str(bp),sha256=H(bp),threads=1));error=C.create_string_buffer(4096);rows=[]
for rep in range(3):
    configs=[('native',1),(2,1),(3,1),('native',16),(2,16),(3,16)];configs=configs[rep*2:]+configs[:rep*2];native_records=[]
    for method,threads in configs:
        seed=705100+rep;count=512
        if method=='native':
            start=time.perf_counter();handle=lib.gr_create(144,50,ptr(v),ptr(x),error,len(error));setup=time.perf_counter()-start;assert handle,error.value;lib.gr_set_gemm(handle,C.cast(blas.scipy_cblas_zgemm64_,C.c_void_p));counts=np.zeros((count,144),np.uint32);stats=np.zeros(20);start=time.perf_counter();rc=lib.g70_sample_parallel(handle,count,seed,4,1,None,threads,ptr(mean),ptr(variance),counts.ctypes.data_as(UP),ptr(stats),error,len(error));sampling=time.perf_counter()-start;lib.gr_destroy(handle);assert rc==0,error.value;start=time.perf_counter();clicks=counts>0;click_hist=np.bincount(clicks.sum(axis=1),minlength=145);output_time=time.perf_counter()-start;native_records.append(counts);np.savez_compressed(OUT/f'native_{rep}_{threads}.npz',counts=counts,stats=stats,clicks=clicks)
            row=dict(rep=rep,method='native_newton',threads=threads,seed=seed,samples=count,setup_seconds=setup,sampling_seconds=sampling,output_seconds=output_time,compute_seconds=setup+sampling+output_time,mean_clicks=float(clicks.sum(axis=1).mean()),maximum_photons=int(counts.sum(axis=1).max()))
        else:
            command=[str(OUT/'frugal.exe'),'--input',str(OUT/'covariance.csv'),'--modes','144','--cutoff',str(method),'--samples',str(count),'--seed',str(seed),'--threads',str(threads)];start=time.perf_counter();proc=subprocess.run(command,capture_output=True,text=True,errors='replace');wall=time.perf_counter()-start;save(f'process_{rep}_{threads}_{method}.json',dict(command=command,returncode=proc.returncode,stderr=proc.stderr,wall_seconds=wall));assert proc.returncode==0,proc.stderr;result=json.loads(proc.stdout);save(f'frugal_{rep}_{threads}_{method}.json',result);t=result['timings_seconds'];setup=t['layout']+t['cumulants']+t['target'];sampling=t['official_sampling'];strings=result['outputs'];assert len(strings)==count and all(len(s)==144 and set(s)<=set('01') for s in strings);clicks=np.asarray([[c=='1' for c in s] for s in strings]);row=dict(rep=rep,method=f'frugal_K{method}',threads=threads,seed=seed,samples=count,setup_seconds=setup,sampling_seconds=sampling,compute_seconds=setup+sampling,process_wall_seconds=wall,mean_clicks=float(clicks.sum(axis=1).mean()))
        rows.append(row);save(f'row_{rep}_{threads}_{method}.json',row);print(row,flush=True)
    assert len(native_records)==2 and np.array_equal(*native_records)
save('summary.json',dict(passed=True,rows=rows,input_sha256=H(OUT/'input.npz'),native_binary_sha256=H(OUT/'native.dll'),frugal_binary_sha256=H(OUT/'frugal.exe')))
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
