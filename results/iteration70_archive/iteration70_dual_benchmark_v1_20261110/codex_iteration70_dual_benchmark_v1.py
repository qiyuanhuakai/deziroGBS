from pathlib import Path
import os,ctypes as C,hashlib,json,shutil,subprocess,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
OUT=ROOT/'.git/iteration70_dual_benchmark_v1_20261110';OUT.mkdir(exist_ok=False)
LARGE=ROOT/'.git/iteration70_frugal_large_v1_20261110';assert json.loads((LARGE/'summary.json').read_text())['passed']
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    p=OUT/name;assert not p.exists();p.write_bytes((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';source=ROOT/'.git/iteration70_direct_click_v1.cpp'
for p in [Path(__file__),source,ROOT/'native/gaussian_reset_cpu.cpp']:shutil.copy2(p,OUT/p.name)
command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(source),'-o',str(OUT/'direct.dll')]
save('plan.json',dict(command=command,source_sha256=H(source),dependencies={p.relative_to(ROOT).as_posix():H(p) for p in [LARGE/'summary.json',LARGE/'input.npz',LARGE/'native.dll',LARGE/'frugal.exe']},methods=['direct_click','pnr_then_click','frugal_K2','frugal_K3'],samples=512,repetitions=3,threads=16,seed_base=709001,scope='Interleaved same-host/same-input/same-worker-budget generic CPU builds. Compare distinct approximate distributions; no equal-accuracy claim. Preparation and sampling both included, output summary overhead separately recorded where available. Frugal process startup/JSON serialization excluded from compute metric. All produced records retained.'))
start=time.perf_counter();build=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=build.returncode,stdout=build.stdout,stderr=build.stderr,seconds=time.perf_counter()-start));print('direct generic build',build.returncode,flush=True);assert build.returncode==0
runtime=os.add_dll_directory(str(gcc.parent));direct=C.CDLL(str(OUT/'direct.dll'));pnr=C.CDLL(str(LARGE/'native.dll'));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32);BP=C.POINTER(C.c_uint8)
def ptr(a):return a.ctypes.data_as(DP)
direct.tc70_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];direct.tc70_create.restype=C.c_void_p;direct.tc70_destroy.argtypes=[C.c_void_p];direct.tc70_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,DP,DP,BP,DP,C.c_char_p,C.c_int]
pnr.gr_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];pnr.gr_create.restype=C.c_void_p;pnr.gr_destroy.argtypes=[C.c_void_p];pnr.gr_set_gemm.argtypes=[C.c_void_p,C.c_void_p];pnr.g70_sample_parallel.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,C.c_void_p,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int]
bp=next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*.dll'));blas=C.CDLL(str(bp));blas.scipy_openblas_set_num_threads64_.argtypes=[C.c_int];blas.scipy_openblas_set_num_threads64_(1);save('blas.json',dict(path=str(bp),sha256=H(bp),threads=1));d=np.load(LARGE/'input.npz');v=np.ascontiguousarray(d['v']);x=np.ascontiguousarray(d['x0']);mean=np.ascontiguousarray(d['mean']);variance=np.ascontiguousarray(d['variance']);error=C.create_string_buffer(4096);rows=[]
for rep in range(3):
    order=['direct','pnr',2,3];order=order[rep:]+order[:rep]
    for method in order:
        seed=709001+rep;count=512;threads=16
        if method in ('direct','pnr'):
            library=direct if method=='direct' else pnr;create=library.tc70_create if method=='direct' else library.gr_create;destroy=library.tc70_destroy if method=='direct' else library.gr_destroy;start=time.perf_counter();handle=create(144,50,ptr(v),ptr(x),error,len(error));setup=time.perf_counter()-start;assert handle,error.value
            if method=='direct':
                output=np.zeros((count,144),np.uint8);stats=np.zeros(5);start=time.perf_counter();rc=direct.tc70_sample(handle,count,seed,threads,ptr(mean),ptr(variance),output.ctypes.data_as(BP),ptr(stats),error,len(error));sampling=time.perf_counter()-start
            else:
                pnr.gr_set_gemm(handle,C.cast(blas.scipy_cblas_zgemm64_,C.c_void_p));output=np.zeros((count,144),np.uint32);stats=np.zeros(20);start=time.perf_counter();rc=pnr.g70_sample_parallel(handle,count,seed,4,1,None,threads,ptr(mean),ptr(variance),output.ctypes.data_as(UP),ptr(stats),error,len(error));sampling=time.perf_counter()-start
            destroy(handle);save(f'native_status_{rep}_{method}.json',dict(returncode=rc,error=error.value.decode(),setup_seconds=setup,sampling_seconds=sampling));assert rc==0,error.value;start=time.perf_counter();clicks=output>0;hist=np.bincount(clicks.sum(axis=1),minlength=145);output_seconds=time.perf_counter()-start;np.savez_compressed(OUT/f'records_{rep}_{method}.npz',output=output,clicks=clicks,stats=stats);row=dict(rep=rep,method=method,seed=seed,samples=count,threads=threads,setup_seconds=setup,sampling_seconds=sampling,output_seconds=output_seconds,compute_seconds=setup+sampling+output_seconds,mean_clicks=float(clicks.sum(axis=1).mean()))
        else:
            cmd=[str(LARGE/'frugal.exe'),'--input',str(LARGE/'covariance.csv'),'--modes','144','--cutoff',str(method),'--samples',str(count),'--seed',str(seed),'--threads',str(threads)];start=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True,errors='replace');wall=time.perf_counter()-start;save(f'process_{rep}_{method}.json',dict(command=cmd,returncode=r.returncode,stderr=r.stderr,wall_seconds=wall));assert r.returncode==0,r.stderr;result=json.loads(r.stdout);save(f'frugal_{rep}_{method}.json',result);t=result['timings_seconds'];setup=t['layout']+t['cumulants']+t['target'];sampling=t['official_sampling'];strings=result['outputs'];assert len(strings)==count and all(len(s)==144 and set(s)<=set('01') for s in strings);clicks=np.asarray([[c=='1' for c in s] for s in strings]);row=dict(rep=rep,method=f'frugal_K{method}',seed=seed,samples=count,threads=threads,setup_seconds=setup,sampling_seconds=sampling,compute_seconds=setup+sampling,wall_seconds=wall,mean_clicks=float(clicks.sum(axis=1).mean()))
        rows.append(row);save(f'row_{rep}_{method}.json',row);print(row,flush=True)
save('summary.json',dict(passed=True,rows=rows,direct_binary_sha256=H(OUT/'direct.dll')))
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
