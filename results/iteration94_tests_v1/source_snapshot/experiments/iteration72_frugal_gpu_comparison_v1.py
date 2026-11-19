"""Pinned Frugal CUDA FP32/FP64 versus public CUDA branch sampler.

Both prepare a fresh target per call, same device, host budget 16 workers.
Library loading and conversion of a common input fixture are excluded; random
generation, allocation, transfers, sampling and output packing are included.
Frugal's host-only layout compatibility patch is recorded by its build driver.
Different approximation laws on the large panel: no equal-quality claim.
"""
from pathlib import Path
import os,ctypes as C,json,hashlib,shutil,time,argparse,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_cuda_click_branches import NativeCudaBranchClickSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--frugal-build',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False);BUILD=args.frugal_build.resolve()

def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_double))

gcc=Path(shutil.which('g++')).parent;cuda=Path(shutil.which('nvcc')).parent
redist=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Redist'
omp=next(redist.glob('MSVC/*/debug_nonredist/x64/Microsoft.VC143.OpenMP.LLVM/libomp140.x86_64.dll'))
runtime_dirs=[gcc,cuda,omp.parent];runtime=[os.add_dll_directory(str(p)) for p in runtime_dirs]
cpu=ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll';gpu=ROOT/'results/iteration72_cuda_pilot_v1/gpu.dll'
paths=[Path(__file__),ROOT/'src/gbslab/native_cuda_click_branches.py',cpu,gpu,BUILD/'frugal_fp32.dll',BUILD/'frugal_fp64.dll',omp]
save('plan.json',dict(hashes={str(p):H(p) for p in paths},frugal_build=str(BUILD),frugal_build_summary=json.loads((BUILD/'summary.json').read_text()),samples=512,repetitions=3,seed_base=715901,host_workers=16,scope=__doc__,small_validation_count=100000,small_validation_tv_limit=.025,large_methods=['branch_1','branch_4','branch_8','frugal32_2','frugal32_3','frugal64_2','frugal64_3']))
shutil.copy2(__file__,OUT/Path(__file__).name);shutil.copy2(ROOT/'src/gbslab/native_cuda_click_branches.py',OUT/'native_cuda_click_branches.py')
libs={bits:C.CDLL(str(BUILD/f'frugal_fp{bits}.dll')) for bits in (32,64)}
for lib in libs.values():
    lib.fg72_sample.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint,C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_uint8),C.POINTER(C.c_double),C.c_char_p,C.c_int]
    lib.fg72_sample.restype=C.c_int

def frugal(cov,cutoff,count,seed,bits):
    m=len(cov)//2;out=np.zeros((count,m),np.uint8);times=np.zeros(6);error=C.create_string_buffer(4096)
    start=time.perf_counter();rc=libs[bits].fg72_sample(m,cutoff,count,seed,16,ptr(cov),out.ctypes.data_as(C.POINTER(C.c_uint8)),ptr(times),error,len(error));wall=time.perf_counter()-start
    return out,dict(returncode=rc,error=error.value.decode(errors='replace') if rc else '',wall_seconds=wall,timings=dict(zip(('layout','cumulant_layout','target','sampling','packing','precision_bits'),times.tolist())))

validation=[]
for bits in (32,64):
    output,r=frugal(np.eye(8),4,128,722000,bits);np.savez_compressed(OUT/f'vacuum_{bits}.npz',clicks=output);r.update(precision=bits,case='vacuum',clicks=int(output.sum()));validation.append(r);save(f'vacuum_{bits}.json',r);assert r['returncode']==0 and not output.any(),r
    for case,cutoff in ((3,4),(4,5)):
        data=np.load(ROOT/f'results/iteration71_branch_pilot_v1/input_{case}.npz');v=data['v'];x=data['x'];m,n=v.shape;o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.ascontiguousarray(np.eye(2*m)+2*o@x@o.T)
        target=data['target'] if case==3 else np.array(json.loads((ROOT/f'results/iteration71_frugal_comparison/frugal_{case}_0_{cutoff}.json').read_text())['distribution'])
        output,r=frugal(cov,cutoff,100000,722000+case,bits);np.savez_compressed(OUT/f'validation_{case}_{bits}.npz',clicks=output,target=target,cov=cov)
        codes=(output@np.left_shift(np.uint64(1),np.arange(m,dtype=np.uint64))).astype(np.int64)
        hist=np.bincount(codes,minlength=2**m)/len(output)
        r.update(case=case,precision=bits,cutoff=cutoff,tv_to_reference=float(abs(hist-target).sum()/2));validation.append(r);save(f'validation_{case}_{bits}.json',r);print('validation',r,flush=True);assert r['returncode']==0 and r['tv_to_reference']<.025,r
save('validation.json',dict(passed=True,rows=validation))

data=np.load(ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110/input.npz');v=data['v'];x=data['x0'];mean=data['mean'];variance=data['variance'];cov=np.ascontiguousarray(np.loadtxt(ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110/covariance.csv',delimiter=','))
with NativeCudaBranchClickSampler(v,x,cpu,gpu_library=gpu,mean=mean,noise_variance=variance,runtime_dirs=runtime_dirs) as warm:
    warm.sample(1,722008)
methods=[('branch',1),('branch',4),('branch',8),('frugal32',2),('frugal32',3),('frugal64',2),('frugal64',3)];rows=[]
for rep in range(3):
    order=methods[2*rep:]+methods[:2*rep]
    for kind,budget in order:
        seed=715901+rep;label=f'{rep}_{kind}_{budget}';save(f'started_{label}.json',dict(rep=rep,kind=kind,budget=budget,seed=seed))
        row=dict(rep=rep,method=kind,budget=budget,samples=512,seed=seed,workers=16)
        if kind=='branch':
            start=time.perf_counter()
            with NativeCudaBranchClickSampler(v,x,cpu,gpu_library=gpu,mean=mean,noise_variance=variance,branch_budget=budget,workers=16,runtime_dirs=runtime_dirs) as model:
                setup=time.perf_counter()-start;r=model.sample(512,seed);output=r['clicks']
            row.update(returncode=0,setup_seconds=setup,wall_seconds=time.perf_counter()-start,sampling_seconds=r['wall_seconds'],random_seconds=r['random_seconds'],gpu_seconds=r['gpu_seconds'],cuda_timings=r['cuda_timings'],statistics=r['statistics'])
            expected=np.load(ROOT/f'results/iteration71_frugal_comparison/large_{rep}_branch_{budget}.npz')['clicks'];row['mismatched_archived_bits']=int(np.count_nonzero(output!=expected));assert row['mismatched_archived_bits']==0
            np.savez_compressed(OUT/f'records_{label}.npz',clicks=output,conditional_probabilities=r['conditional_click_probabilities'],diagnostics=r['trajectory_diagnostics'])
        else:
            output,r=frugal(cov,budget,512,seed,int(kind[-2:]));row.update(r);np.savez_compressed(OUT/f'records_{label}.npz',clicks=output)
        rows.append(row);save(f'row_{label}.json',row);print(row,flush=True);assert row['returncode']==0,row
save('summary.json',dict(passed=True,validation=validation,rows=rows))
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
