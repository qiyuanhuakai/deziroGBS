"""Small full-law accuracy anchors and matched CUDA timing, fresh model per call."""
from pathlib import Path
import os,ctypes as C,json,hashlib,shutil,time,argparse,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_cuda_click_branches import NativeCudaBranchClickSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_double))
gcc=Path(shutil.which('g++')).parent;cuda=Path(shutil.which('nvcc')).parent
omp=next((Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Redist').glob('MSVC/*/debug_nonredist/x64/Microsoft.VC143.OpenMP.LLVM/libomp140.x86_64.dll'))
dirs=[gcc,cuda,omp.parent];runtimes=[os.add_dll_directory(str(p)) for p in dirs]
cpu=ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll';gpu=ROOT/'results/iteration72_cuda_pilot_v1/gpu.dll';frugal_path=ROOT/'results/iteration72_frugal_gpu_build_v5/frugal_fp64.dll'
save('plan.json',dict(source_hashes={str(p):H(p) for p in [Path(__file__),cpu,gpu,frugal_path,ROOT/'src/gbslab/native_cuda_click_branches.py']},cases=[3,4],samples=16384,repetitions=3,seed_base=723001,scope=__doc__,note='Frugal exact small-law references are its previous FP64 CPU enumeration, checked empirically against this CUDA baseline; CUDA RNG uses upstream float-uniform macro even in FP64.'))
lib=C.CDLL(str(frugal_path));lib.fg72_sample.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint,C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_uint8),C.POINTER(C.c_double),C.c_char_p,C.c_int];lib.fg72_sample.restype=C.c_int
def frugal(cov,k,count,seed):
    m=len(cov)//2;out=np.zeros((count,m),np.uint8);times=np.zeros(6);error=C.create_string_buffer(4096);start=time.perf_counter()
    rc=lib.fg72_sample(m,k,count,seed,16,ptr(cov),out.ctypes.data_as(C.POINTER(C.c_uint8)),ptr(times),error,len(error));elapsed=time.perf_counter()-start
    return out,dict(returncode=rc,error=error.value.decode(errors='replace') if rc else '',wall_seconds=elapsed,timings=times.tolist())
rows=[]
for case in (3,4):
    data=np.load(ROOT/f'results/iteration71_branch_pilot_v1/input_{case}.npz');v=data['v'];x=data['x'];m,n=v.shape;o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.ascontiguousarray(np.eye(2*m)+2*o@x@o.T);k=4 if case==3 else 5
    warm,ws=frugal(cov,k,1,723000);save(f'warm_frugal_{case}.json',ws);assert ws['returncode']==0
    with NativeCudaBranchClickSampler(v,x,cpu,gpu_library=gpu,runtime_dirs=dirs) as model:model.sample(1,723000)
    methods=[('branch',8),('frugal',k)] if case==3 else [('branch',8),('branch',32),('branch',128),('frugal',k)]
    for rep in range(3):
        for kind,budget in methods[rep%len(methods):]+methods[:rep%len(methods)]:
            seed=723001+case*100+rep;label=f'{case}_{rep}_{kind}_{budget}'
            if kind=='branch':
                start=time.perf_counter()
                with NativeCudaBranchClickSampler(v,x,cpu,gpu_library=gpu,branch_budget=budget,runtime_dirs=dirs) as model:
                    r=model.sample(16384,seed);out=r['clicks']
                row=dict(returncode=0,wall_seconds=time.perf_counter()-start,sampling_seconds=r['wall_seconds'],cuda_timings=r['cuda_timings'])
                law=np.load(ROOT/f'results/iteration72_cuda_pilot_v1/law_{case}_{budget}.npz')['law']
            else:
                out,row=frugal(cov,budget,16384,seed)
                law=np.array(json.loads((ROOT/f'results/iteration71_frugal_comparison/frugal_{case}_0_{budget}.json').read_text())['distribution'])
            row.update(case=case,rep=rep,method=kind,budget=budget,seed=seed,samples=16384,reference_induced_tv=float(abs(law-data['target']).sum()/2))
            np.savez_compressed(OUT/f'records_{label}.npz',clicks=out,reference_law=law,target=data['target']);save(f'row_{label}.json',row);rows.append(row);print(row,flush=True);assert row['returncode']==0
save('summary.json',dict(passed=True,rows=rows));shutil.copy2(__file__,OUT/Path(__file__).name)
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
