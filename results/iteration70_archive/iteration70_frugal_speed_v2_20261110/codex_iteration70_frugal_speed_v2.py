from pathlib import Path
import os,sys,ctypes as C,hashlib,json,shutil,subprocess,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from gbslab.frugal import frugal_sample
OUT=ROOT/'.git/iteration70_frugal_speed_v2_20261110';OUT.mkdir(exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    p=OUT/name;assert not p.exists();p.write_bytes((json.dumps(obj,indent=2)+'\n').encode())
shutil.copy2(__file__,OUT/Path(__file__).name)
old=json.loads((ROOT/'results/iteration38_frugal_runtime/build.json').read_text())
command=old['command'][:];command[1:1]=['-ffp-contract=off','-fno-fast-math'];command[-1]=str(OUT/'frugal.exe')
flags=['-O3','-std=c++20','-ffp-contract=off','-fno-fast-math']
save('plan.json',dict(command=command,common_optimization_flags=flags,samples=200000,repetitions=3,seeds=[703101,703102,703103],cases=list(range(4)),methods=['native_newton','frugal_K2','frugal_K3','frugal_K4'],threads=1,scope='Same host, physical input covariance, sample count and single CPU thread. Native PNR is thresholded; Frugal directly emits threshold strings, so the native route performs more work. Both wall output costs and inner sampling costs reported. Numerical distribution quality compared, not assumed equal. Three cyclically rotated orderings; no GPU throughput extrapolation.',upstream_commit=old['upstream_commit']))
start=time.perf_counter();build=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=build.returncode,stdout=build.stdout,stderr=build.stderr,seconds=time.perf_counter()-start));print('Frugal build',build.returncode,flush=True);assert build.returncode==0
gcc=Path(shutil.which('g++'));runtime=os.add_dll_directory(str(gcc.parent));dll=OUT/'native.dll'
native_source=ROOT/'.git/iteration70_flow_v1.cpp';shutil.copy2(native_source,OUT/native_source.name)
native_command=[str(gcc)]+flags+['-shared','-DEIGEN_DONT_PARALLELIZE','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(native_source),'-o',str(dll)]
start=time.perf_counter();nb=subprocess.run(native_command,capture_output=True,text=True,errors='replace');save('native_build.json',dict(command=native_command,returncode=nb.returncode,stdout=nb.stdout,stderr=nb.stderr,seconds=time.perf_counter()-start,source_sha256=H(native_source),included_source_sha256=H(ROOT/'native/gaussian_reset_cpu.cpp')));print('native build',nb.returncode,flush=True);assert nb.returncode==0
lib=C.CDLL(str(dll));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32)
def ptr(a):return a.ctypes.data_as(DP)
lib.gr_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];lib.gr_create.restype=C.c_void_p;lib.gr_destroy.argtypes=[C.c_void_p];lib.g70_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int]
rows=[];count=200000;error=C.create_string_buffer(4096);dependencies={str(dll.relative_to(ROOT)):H(dll),str((ROOT/'src/gbslab/frugal.py').relative_to(ROOT)):H(ROOT/'src/gbslab/frugal.py')}
for case in range(4):
    inp=ROOT/f'results/iteration69_archive/iteration69_displaced_v6_20261110/input_{case}.npz';covp=ROOT/f'results/iteration69_archive/iteration69_quality_pilot_20261110/input_{case}.npz';dependencies[inp.relative_to(ROOT).as_posix()]=H(inp);dependencies[covp.relative_to(ROOT).as_posix()]=H(covp)
    data=np.load(inp);v=np.ascontiguousarray(data['v']);x=np.ascontiguousarray(data['x0']);mean=np.ascontiguousarray(data['mean']);variance=np.ascontiguousarray(data['noise_variance']);target=data['target'];cov=np.load(covp)['covariance']
    for rep in range(3):
        order=['native',2,3,4];order=order[rep:]+order[:rep]
        for method in order:
            seed=703101+100*case+rep
            if method=='native':
                start=time.perf_counter();handle=lib.gr_create(4,2,ptr(v),ptr(x),error,len(error));setup=time.perf_counter()-start;assert handle,error.value
                counts=np.zeros((count,4),np.uint32);stats=np.zeros(18);start=time.perf_counter();rc=lib.g70_sample(handle,count,seed,3,1,ptr(mean),ptr(variance),counts.ctypes.data_as(UP),ptr(stats),error,len(error));sampling=time.perf_counter()-start;lib.gr_destroy(handle);assert rc==0,error.value
                start=time.perf_counter();hist=np.bincount((counts>0)@np.array([1,2,4,8]),minlength=16);coarsen=time.perf_counter()-start
                row=dict(case=case,rep=rep,method='native_newton',seed=seed,samples=count,sampling_seconds=sampling,setup_seconds=setup,output_seconds=coarsen,total_seconds=setup+sampling+coarsen,empirical_threshold_tv=float(.5*abs(hist/count-target).sum()))
                np.savez_compressed(OUT/f'native_{case}_{rep}.npz',counts=counts,stats=stats,histogram=hist)
            else:
                result=frugal_sample(cov,method,count,seed,executable=OUT/'frugal.exe',threads=1);save(f'frugal_{case}_{rep}_{method}.json',result);t=result['timings_seconds'];setup=t['layout']+t['target'];sampling=t['sampling'];coarsen=t['histogram'];hist=np.asarray(result['histogram_counts']);law=np.asarray(result['distribution'])
                row=dict(case=case,rep=rep,method=f'frugal_K{method}',seed=seed,samples=count,sampling_seconds=sampling,setup_seconds=setup,output_seconds=coarsen,total_seconds=setup+sampling+coarsen,wrapper_wall_seconds=result['wrapper_wall_seconds'],empirical_threshold_tv=float(.5*abs(hist/count-target).sum()),induced_threshold_tv=float(.5*abs(law-target).sum()))
            rows.append(row);save(f'row_{case}_{rep}_{method}.json',row);print(row,flush=True)
save('summary.json',dict(passed=True,rows=rows,dependencies=dependencies,frugal_binary_sha256=H(OUT/'frugal.exe')))
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
