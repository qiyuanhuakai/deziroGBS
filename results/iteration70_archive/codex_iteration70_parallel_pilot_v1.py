from pathlib import Path
import os,ctypes as C,hashlib,json,shutil,subprocess,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
OUT=ROOT/'.git/iteration70_parallel_pilot_v1_20261110';OUT.mkdir(exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    p=OUT/name;assert not p.exists();p.write_bytes((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';source=ROOT/'.git/iteration70_parallel_v1.cpp'
sources=[Path(__file__),source,ROOT/'.git/iteration70_flow_v2.cpp',ROOT/'native/gaussian_reset_cpu.cpp']
for p in sources:shutil.copy2(p,OUT/p.name)
command=[str(gcc),'-O3','-march=native','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(source),'-o',str(OUT/'native.dll')]
save('plan.json',dict(command=command,sources={p.relative_to(ROOT).as_posix():H(p) for p in sources},threads=[1,4,8,16],samples=64,repetitions=3,scope='Independent C++ trajectories with per-thread x87 guard and EVD workspace. No photon cutoff. Row seeds fixed regardless of scheduling. Component timings sum across workers; outer wall time is performance metric. Pilot same matrices as iteration69, fresh seeds. Baseline is same new binary one worker.'))
start=time.perf_counter();build=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=build.returncode,stdout=build.stdout,stderr=build.stderr,seconds=time.perf_counter()-start));print('parallel build',build.returncode,flush=True);assert build.returncode==0
runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(OUT/'native.dll'));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32)
def ptr(a):return a.ctypes.data_as(DP)
lib.gr_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];lib.gr_create.restype=C.c_void_p;lib.gr_destroy.argtypes=[C.c_void_p];lib.gr_set_gemm.argtypes=[C.c_void_p,C.c_void_p];lib.gr_precision.argtypes=[DP]
lib.g70_sample_v2.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,C.c_void_p,DP,DP,UP,DP,C.c_char_p,C.c_int]
lib.g70_sample_parallel.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,C.c_void_p,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int]
bp=next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*.dll'));blas=C.CDLL(str(bp));blas.scipy_openblas_set_num_threads64_.argtypes=[C.c_int];blas.scipy_openblas_set_num_threads64_(1);ep=C.cast(blas.scipy_LAPACKE_dsyevd_work64_,C.c_void_p);error=C.create_string_buffer(4096)
save('blas.json',dict(path=str(bp),sha256=H(bp),threads=1))
rows=[];checks=[];deps={}
for case,selector in [(0,4),(1,5),(5,5)]:
    path=ROOT/f'results/iteration69_benchmark/input_{case}.npz';deps[path.relative_to(ROOT).as_posix()]=H(path);data=np.load(path);v=np.ascontiguousarray(data['v']);x=np.ascontiguousarray(data['x0']);mean=np.ascontiguousarray(data['mean']);variance=np.ascontiguousarray(data['variance']);m,n=v.shape;handle=lib.gr_create(m,n,ptr(v),ptr(x),error,len(error));assert handle,error.value;lib.gr_set_gemm(handle,C.cast(blas.scipy_cblas_zgemm64_,C.c_void_p))
    for algorithm in (1,2):
        reference=np.zeros((8,m),np.uint32);stats=np.zeros(18);rc=lib.g70_sample_v2(handle,8,704000+case,selector,algorithm,ep,ptr(mean),ptr(variance),reference.ctypes.data_as(UP),ptr(stats),error,len(error));assert rc==0,error.value
        parallel=np.zeros_like(reference);pstats=np.zeros(20);rc=lib.g70_sample_parallel(handle,8,704000+case,selector,algorithm,ep,4,ptr(mean),ptr(variance),parallel.ctypes.data_as(UP),ptr(pstats),error,len(error));check=dict(case=case,algorithm=algorithm,rc=rc,mismatches=int(np.count_nonzero(reference!=parallel)),completed=int(pstats[19]));checks.append(check);save(f'check_{case}_{algorithm}.json',check);np.savez_compressed(OUT/f'check_{case}_{algorithm}.npz',reference=reference,parallel=parallel,stats=pstats);assert rc==0 and check['mismatches']==0 and check['completed']==8,check
    for rep in range(3):
        order=[1,4,8,16];order=order[rep:]+order[:rep];reference=None;current=[]
        for threads in order:
            counts=np.zeros((64,m),np.uint32);stats=np.zeros(20);before=np.zeros(5);after=np.zeros(5);lib.gr_precision(ptr(before));start=time.perf_counter();rc=lib.g70_sample_parallel(handle,64,704100+case*100+rep,selector,1,None,threads,ptr(mean),ptr(variance),counts.ctypes.data_as(UP),ptr(stats),error,len(error));seconds=time.perf_counter()-start;lib.gr_precision(ptr(after));row=dict(case=case,modes=m,inputs=n,rep=rep,threads=threads,rc=rc,error=error.value.decode(),seconds=seconds,completed=int(stats[19]),workers=int(stats[18]),precision_restored=bool(before[0]==after[0]),photons=int(stats[0]),maximum_photons=int(stats[6]));save(f'case_{case}_{rep}_{threads}.json',row);np.savez_compressed(OUT/f'case_{case}_{rep}_{threads}.npz',counts=counts,stats=stats);current.append((row,counts));print(row,flush=True);assert rc==0 and stats[19]==64 and row['precision_restored'],row
            if threads==1:reference=counts
        for row,counts in current:
            row['mismatches']=int(np.count_nonzero(reference!=counts));assert row['mismatches']==0,row;rows.append(row)
    lib.gr_destroy(handle)
save('summary.json',dict(passed=True,rows=rows,checks=checks,dependencies=deps,dll_sha256=H(OUT/'native.dll')))
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
