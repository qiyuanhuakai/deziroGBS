from pathlib import Path
import os,sys,ctypes as C,hashlib,json,shutil,subprocess,time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
OUT=ROOT/'.git/iteration70_flow_pilot_v1_20261110';OUT.mkdir(exist_ok=False)
def H(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):
    p=OUT/name;assert not p.exists();p.write_bytes((json.dumps(obj,indent=2)+'\n').encode())
source=ROOT/'.git/iteration70_flow_v1.cpp';old_source=ROOT/'native/gaussian_reset_cpu.cpp';gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';old=ROOT/'results/iteration69_benchmark';probedir=ROOT/'results/iteration69_archive/iteration69_displaced_v6_20261110'
sources=[Path(__file__),source,old_source]
for p in sources:shutil.copy2(p,OUT/p.name)
command=[str(gcc),'-O3','-march=native','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(source),'-o',str(OUT/'native.dll')]
save('plan.json',dict(source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},command=command,compiler_sha256=H(gcc),cases=[dict(old_case=case,selector=selector,samples=32) for case,selector in [(0,4),(1,5),(4,3),(5,5)]],algorithms={0:'80 bisections',1:'safeguarded Newton with residual stopping and bisection fallback'},repetitions=2,scope='Profile the same frozen inputs/seeds as iteration69 to separate spectral solve, waiting inversion and reconstruction. Baseline counts must replay iteration69. Extra timing and residual evaluations are included in BOTH methods. Newton changes finite arithmetic; report state and count differences, not an exact program-law equivalence claim. This is a pilot, not the final fresh benchmark.'))
save('eigen_headers.json',json.loads((ROOT/'results/iteration69_build/eigen_headers.json').read_text()))
start=time.perf_counter();build=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=build.returncode,stdout=build.stdout,stderr=build.stderr,seconds=time.perf_counter()-start));print('build',build.returncode,flush=True);assert build.returncode==0
runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(OUT/'native.dll'));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32)
def ptr(a):return a.ctypes.data_as(DP)
lib.gr_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];lib.gr_create.restype=C.c_void_p;lib.gr_destroy.argtypes=[C.c_void_p];lib.gr_set_gemm.argtypes=[C.c_void_p,C.c_void_p]
lib.g70_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int]
lib.g70_probe.argtypes=[C.c_void_p,DP,DP,C.c_double,C.c_int,DP,DP,DP,DP,C.c_char_p,C.c_int]
blas_path=next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*.dll'));blas=C.CDLL(str(blas_path));blas.scipy_openblas_set_num_threads64_.argtypes=[C.c_int];blas.scipy_openblas_set_num_threads64_(1);save('blas.json',dict(path=str(blas_path),sha256=H(blas_path),threads=1))
error=C.create_string_buffer(4096);checks=[];dependencies={}
for case in range(4):
    path=probedir/f'input_{case}.npz';dependencies[path.relative_to(ROOT).as_posix()]=H(path);data=np.load(path);v=np.ascontiguousarray(data['v']);x0=np.ascontiguousarray(data['x0']);m,n=v.shape;handle=lib.gr_create(m,n,ptr(v),ptr(x0),error,len(error));assert handle,error.value
    for step in range(5):
        path=probedir/f'probe_{case}_{step}.npz';dependencies[path.relative_to(ROOT).as_posix()]=H(path);d=np.load(path);x=np.ascontiguousarray(d['before_x']);w=np.ascontiguousarray(d['before_w']);results=[]
        for algorithm in (0,1):
            xx=np.zeros_like(x);ww=np.zeros_like(w);info=np.zeros(3);stats=np.zeros(18);rc=lib.g70_probe(handle,ptr(x),ptr(w),float(d['waiting']),algorithm,ptr(xx),ptr(ww),ptr(info),ptr(stats),error,len(error));assert rc==0,error.value;results.append((xx,ww,info,stats));np.savez_compressed(OUT/f'probe_{case}_{step}_{algorithm}.npz',x=xx,w=ww,info=info,stats=stats)
        a,b=results;row=dict(case=case,step=step,state_error=float(np.max(abs(a[0]-b[0]))),mean_error=float(np.max(abs(a[1]-b[1]))),time_error=float(abs(a[2][2]-b[2][2])),log_residual=float(b[3][13]),iterations=int(b[3][12]));row['passed']=max(row[k] for k in ('state_error','mean_error','time_error'))<1e-11 and row['log_residual']<1e-12;checks.append(row);assert row['passed'],row
    lib.gr_destroy(handle)
save('probes.json',checks);rows=[]
for case,selector in [(0,4),(1,5),(4,3),(5,5)]:
    path=old/f'input_{case}.npz';dependencies[path.relative_to(ROOT).as_posix()]=H(path);data=np.load(path);v=np.ascontiguousarray(data['v']);x0=np.ascontiguousarray(data['x0']);mean=np.ascontiguousarray(data['mean']);variance=np.ascontiguousarray(data['variance']);m,n=v.shape;handle=lib.gr_create(m,n,ptr(v),ptr(x0),error,len(error));assert handle,error.value;lib.gr_set_gemm(handle,C.cast(blas.scipy_cblas_zgemm64_,C.c_void_p))
    for repetition in range(2):
        reference_path=old/f'case_{case}_rep_{repetition}_method_{selector}.npz';dependencies[reference_path.relative_to(ROOT).as_posix()]=H(reference_path);reference=np.load(reference_path)['counts'][:32];seed=692100+case*100+repetition
        for algorithm in ((0,1) if repetition==0 else (1,0)):
            counts=np.zeros((32,m),np.uint32);stats=np.zeros(18);start=time.perf_counter();rc=lib.g70_sample(handle,len(counts),seed,selector,algorithm,ptr(mean),ptr(variance),counts.ctypes.data_as(UP),ptr(stats),error,len(error));seconds=time.perf_counter()-start;row=dict(case=case,modes=m,inputs=n,selector=selector,algorithm=algorithm,repetition=repetition,seed=seed,returncode=rc,error=error.value.decode(),seconds=seconds,stats=stats.tolist(),mismatched_reference_entries=int(np.count_nonzero(counts!=reference)));row['passed']=rc==0 and (algorithm==1 or row['mismatched_reference_entries']==0);rows.append(row);save(f'case_{case}_rep_{repetition}_algorithm_{algorithm}.json',row);np.savez_compressed(OUT/f'case_{case}_rep_{repetition}_algorithm_{algorithm}.npz',counts=counts,stats=stats);print(row,flush=True);assert row['passed'],row
    lib.gr_destroy(handle)
save('summary.json',dict(passed=all(r['passed'] for r in rows+checks),rows=rows,probes=checks,dependencies=dependencies,dll_sha256=H(OUT/'native.dll')))
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
