"""Fresh same-input PNR-to-click / direct-click benchmark, three rotated repeats.
Setup and output conversion included; binary loading and common input excluded.
Timing ratios alone do not certify equal distributional quality.
"""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time,ctypes as C,statistics
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_projected_pnr import NativeProjectedPNRSampler
from gbslab.native_number_pnr import NativeNumberPNRSampler
from gbslab.native_budget_number_pnr import NativeBudgetNumberPNRSampler
from gbslab.native_philox_click import NativePhiloxClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_double))
gcc=Path(shutil.which('g++')).parent;cuda=Path(shutil.which('nvcc')).parent;omp=next((Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Redist').glob('MSVC/*/debug_nonredist/x64/Microsoft.VC143.OpenMP.LLVM/libomp140.x86_64.dll'));dirs=[gcc,cuda,omp.parent];handles=[os.add_dll_directory(str(p)) for p in dirs]
pnr={'projected':(NativeProjectedPNRSampler,ROOT/'results/iteration76_projected_pilot_v3/projected.dll'), 'strict':(NativeNumberPNRSampler,ROOT/'results/iteration77_number_pilot_v2/number.dll'), 'number':(NativeBudgetNumberPNRSampler,ROOT/'results/iteration77_budget_pilot_v1/budget.dll')}
cpu=ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll';fused=ROOT/'results/iteration74_fused_pilot_v2/philox.dll';frugal=ROOT/'results/iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll';gpu=ROOT/'results/iteration75_frugal_gpu_build_v3/frugal_fp64.dll'
libs={'cpu':C.CDLL(str(frugal)),'gpu':C.CDLL(str(gpu))}
for kind,lib in libs.items():
    fn=lib.fc73_sample if kind=='cpu' else lib.fg72_sample;fn.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint,C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_uint8),C.POINTER(C.c_double),C.c_char_p,C.c_int];fn.restype=C.c_int
paths=[Path(__file__),cpu,fused,frugal,gpu,*[p for _,p in pnr.values()]]
cases=[('results/iteration75_task_comparison_v1/input.npz',65536,[('projected',31),('strict',0),('number',64),('fused',8),('cpu',4),('gpu',4)]),('results/iteration75_lowrank_v2/input.npz',1024,[('projected',31),('strict',0),('number',64),('fused',4),('cpu',2),('cpu',3)])]
save('plan.json',dict(scope=__doc__,cases=cases,repetitions=3,workers=16,seed_base=0x8a000000000000,seed_stride=2**40,frugal_seed=778001,hashes={p.relative_to(ROOT).as_posix():H(p) for p in paths}));shutil.copy2(__file__,OUT/Path(__file__).name);rows=[];couplings=[]
for case,(source,count,methods) in enumerate(cases):
    d=np.load(ROOT/source);v,t=d['v'],d['t'];m,n=v.shape;x=np.diag(np.r_[-t/(1+t),t/(1-t)]);mean=np.zeros(2*n);o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.ascontiguousarray(np.eye(2*m)+2*o@x@o.T);np.savez_compressed(OUT/f'input_{case}.npz',v=v,t=t,x=x,cov=cov)
    def run(kind,budget,count,rep):
        seed=0x8a000000000000+(case*16+rep)*2**40;start=time.perf_counter();counts=totals=None
        if kind in pnr:
            cls,binary=pnr[kind];kw={'retained':budget} if kind=='projected' else {'exact_total_budget':budget} if kind=='number' else {}
            with cls(v,t,binary,workers=16,runtime_dirs=dirs,**kw) as model:r=model.sample(count,seed)
            counts=r['counts'];bits=(counts>0).astype(np.uint8);totals=r.get('total_photons',counts.sum(axis=1));meta=dict(statistics=r['statistics'],native_sampling_seconds=r['wall_seconds'])
        elif kind=='fused':
            with NativePhiloxClickSampler(v,x,cpu,gpu_library=fused,mean=mean,branch_budget=budget,runtime_dirs=dirs) as model:r=model.sample(count,seed)
            bits=r['clicks'];meta=dict(cuda_timings=r['cuda_timings'],statistics=r['statistics'])
        else:
            bits=np.zeros((count,m),np.uint8);times=np.zeros(7 if kind=='cpu' else 6);error=C.create_string_buffer(4096);fn=libs[kind].fc73_sample if kind=='cpu' else libs[kind].fg72_sample;rc=fn(m,budget,count,778001+case*16+rep,16,ptr(cov),bits.ctypes.data_as(C.POINTER(C.c_uint8)),ptr(times),error,len(error));assert rc==0,error.value;meta=dict(component_timings=times.tolist())
        return counts,bits,totals,dict(wall_seconds=time.perf_counter()-start,**meta)
    for kind,budget in methods:run(kind,budget,1,0)
    for rep in range(3):
        order=methods[2*rep:]+methods[:2*rep]
        for kind,budget in order:
            label=f'{case}_{rep}_{kind}_{budget}';save(f'started_{label}.json',dict(case=case,rep=rep,kind=kind,budget=budget,count=count));counts,bits,totals,row=run(kind,budget,count,rep);row.update(case=case,rep=rep,kind=kind,budget=budget,records=count);np.savez_compressed(OUT/f'records_{label}.npz',clicks=bits,**({} if counts is None else {'counts':counts,'totals':totals}));rows.append(row);save(f'row_{label}.json',row);print(label,row['wall_seconds'],flush=True)
        a=np.load(OUT/f'records_{case}_{rep}_strict_0.npz');b=np.load(OUT/f'records_{case}_{rep}_number_64.npz');mask=a['totals']<=64;assert np.array_equal(a['totals'],b['totals']) and np.array_equal(a['counts'][mask],b['counts'][mask]);couplings.append(dict(case=case,rep=rep,records=count,exact_records=int(mask.sum()),same_exact_records=True))
medians={f'{case}_{kind}_{budget}':statistics.median(r['wall_seconds'] for r in rows if (r['case'],r['kind'],r['budget'])==(case,kind,budget)) for case,(_,_,methods) in enumerate(cases) for kind,budget in methods}
save('summary.json',dict(passed=True,rows=rows,medians=medians,couplings=couplings));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
