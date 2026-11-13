"""Complete resumed run after the user-interrupted v1; no cross-date splicing. Matched small and large click sampling: Philox, CPU RNG, AVX2 Frugal, CUDA Frugal.

Fresh model preparation, RNG, allocation, transfer and host output packing are
included. Library loading for Frugal and common fixture reading are excluded.
Three rotating repetitions; no concurrent benchmark jobs. Large full TV is
not certified. Large comparisons remeasure the best tested AVX2 FP64 CPU; prior GPU timings remain explicitly historical.
"""
from pathlib import Path
import os,ctypes as C,json,hashlib,shutil,time,argparse,sys,subprocess
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_philox_click import NativePhiloxClickSampler
from gbslab.native_cuda_click_branches import NativeCudaBranchClickSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_double))
gcc=Path(shutil.which('g++')).parent;cuda=Path(shutil.which('nvcc')).parent;omp=next((Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Redist').glob('MSVC/*/debug_nonredist/x64/Microsoft.VC143.OpenMP.LLVM/libomp140.x86_64.dll'));dirs=[gcc,cuda,omp.parent];runtimes=[os.add_dll_directory(str(p)) for p in dirs]
cpu=ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll';oldgpu=ROOT/'results/iteration72_cuda_pilot_v1/gpu.dll';philox=ROOT/'results/iteration74_fused_pilot_v2/philox.dll'
frugal_paths={f'cpu{bits}':ROOT/f'results/iteration73_frugal_cpu_library_v1/frugal_cpu_fp{bits}.dll' for bits in (32,64)}|{f'gpu{bits}':ROOT/f'results/iteration72_frugal_gpu_build_v5/frugal_fp{bits}.dll' for bits in (32,64)}
libs={name:C.CDLL(str(path)) for name,path in frugal_paths.items()}
for kind,lib in libs.items():
    fn=lib.fc73_sample if kind.startswith('cpu') else lib.fg72_sample;fn.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint,C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_uint8),C.POINTER(C.c_double),C.c_char_p,C.c_int];fn.restype=C.c_int
large_methods=[('philox',1),('philox',4),('philox',8),('cpu64',2),('cpu64',3)]
paths=[Path(__file__),ROOT/'src/gbslab/native_philox_click.py',ROOT/'src/gbslab/native_cuda_click_branches.py',cpu,oldgpu,philox,*frugal_paths.values()]
save('plan.json',dict(source_hashes={str(p):H(p) for p in paths},scope=__doc__,large_sizes=[4096],large_methods=large_methods,small_cases=[3,4],small_count=16384,repetitions=3,threads=16,seeds={'small_base':745001,'large_base':745501},cpu32_diagnostic_count=0))
q=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,errors='replace');save('device.json',dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr));rows=[]
def run(kind,budget,count,seed,v,x,mean,variance,cov):
    if kind in ('philox','cpu_rng'):
        start=time.perf_counter();cls=NativePhiloxClickSampler if kind=='philox' else NativeCudaBranchClickSampler
        with cls(v,x,cpu,gpu_library=philox if kind=='philox' else oldgpu,mean=mean,noise_variance=variance,branch_budget=budget,runtime_dirs=dirs) as model:
            setup=time.perf_counter()-start;r=model.sample(count,seed)
        return r['clicks'],dict(returncode=0,wall_seconds=time.perf_counter()-start,setup_seconds=setup,sampling_seconds=r['wall_seconds'],cuda_timings=r['cuda_timings'],statistics=r['statistics'])
    m=len(cov)//2;out=np.zeros((count,m),np.uint8);times=np.zeros(7 if kind.startswith('cpu') else 6);error=C.create_string_buffer(4096);fn=libs[kind].fc73_sample if kind.startswith('cpu') else libs[kind].fg72_sample
    start=time.perf_counter();rc=fn(m,budget,count,seed,16,ptr(cov),out.ctypes.data_as(C.POINTER(C.c_uint8)),ptr(times),error,len(error));elapsed=time.perf_counter()-start
    return out,dict(returncode=rc,error=error.value.decode(errors='replace') if rc else '',wall_seconds=elapsed,component_timings=times.tolist())
for case in (3,4):
    data=np.load(ROOT/f'results/iteration71_branch_pilot_v1/input_{case}.npz');v=data['v'];x=data['x'];mean=data['mean'];variance=data['variance'];m,n=v.shape;o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.ascontiguousarray(np.eye(2*m)+2*o@x@o.T);cutoff=4 if case==3 else 5
    methods=([('philox',8),('cpu_rng',8)] if case==3 else [('philox',8),('philox',32),('philox',128),('cpu_rng',8)])+[(kind,cutoff) for kind in ('cpu32','cpu64','gpu32','gpu64')]
    for kind,budget in methods:
        _,r=run(kind,budget,1,745000,v,x,mean,variance,cov);save(f'warm_{case}_{kind}_{budget}.json',r);assert r['returncode']==0
    for rep in range(3):
        for kind,budget in methods[rep:]+methods[:rep]:
            label=f'small_{case}_{rep}_{kind}_{budget}';out,r=run(kind,budget,16384,745001+100*case+rep,v,x,mean,variance,cov);r.update(panel=f'small_{case}',rep=rep,method=kind,budget=budget,samples=16384,seed=745001+100*case+rep);rows.append(r);np.savez_compressed(OUT/f'records_{label}.npz',clicks=out);save(f'row_{label}.json',r);print(label,r['wall_seconds'],r['returncode'],flush=True);assert r['returncode']==0
data=np.load(ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110/input.npz');v=data['v'];x=data['x0'];mean=data['mean'];variance=data['variance'];cov=np.ascontiguousarray(np.loadtxt(ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110/covariance.csv',delimiter=','))
for count in (4096,):
    for rep in range(3):
        order=large_methods[2*rep:]+large_methods[:2*rep]
        for kind,budget in order:
            seed=745501+rep;label=f'large_{count}_{rep}_{kind}_{budget}';save(f'started_{label}.json',dict(kind=kind,budget=budget,count=count,seed=seed));out,r=run(kind,budget,count,seed,v,x,mean,variance,cov);r.update(panel=f'large_{count}',rep=rep,method=kind,budget=budget,samples=count,seed=seed);rows.append(r);np.savez_compressed(OUT/f'records_{label}.npz',clicks=out);save(f'row_{label}.json',r);print(label,r['wall_seconds'],r['returncode'],flush=True);assert r['returncode']==0
save('summary.json',dict(passed=True,rows=rows));shutil.copy2(__file__,OUT/Path(__file__).name);save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
