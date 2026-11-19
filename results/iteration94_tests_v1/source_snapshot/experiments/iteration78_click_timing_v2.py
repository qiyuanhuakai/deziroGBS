"""Complete setup-inclusive small-output click timings, three rotated repeats.
Table and fused full-branch laws checked against independent 80-digit targets.
Frugal K4 is complete at m4; Frugal K5 at m8 is a different approximation.
"""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,time,ctypes as C,statistics
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_table_click import NativeTableClickSampler
from gbslab.native_alias_click import NativeAliasClickSampler
from gbslab.native_philox_click import NativePhiloxClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
ptr=lambda a:a.ctypes.data_as(C.POINTER(C.c_double));gcc=Path(shutil.which('g++')).parent;cuda=Path(shutil.which('nvcc')).parent;omp=next((Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Redist').glob('MSVC/*/debug_nonredist/x64/Microsoft.VC143.OpenMP.LLVM/libomp140.x86_64.dll'));dirs=[gcc,cuda,omp.parent];handles=[os.add_dll_directory(str(p)) for p in dirs];table=ROOT/'results/iteration78_pilot_v2/table.dll';alias=ROOT/'results/iteration78_alias_pilot_v1/table.dll';cpu=ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll';fused=ROOT/'results/iteration74_fused_pilot_v2/philox.dll';frugal=ROOT/'results/iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll';gpu=ROOT/'results/iteration75_frugal_gpu_build_v3/frugal_fp64.dll';libs={'cpu':C.CDLL(str(frugal)),'gpu':C.CDLL(str(gpu))}
for kind,lib in libs.items():
    fn=lib.fc73_sample if kind=='cpu' else lib.fg72_sample;fn.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint,C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_uint8),C.POINTER(C.c_double),C.c_char_p,C.c_int];fn.restype=C.c_int
save('plan.json',dict(scope=__doc__,repetitions=3,cases=[dict(modes=4,records=65536),dict(modes=8,records=8192)],workers=16,seed_base=0x92000000000000,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),table,alias,cpu,fused,frugal,gpu]}));shutil.copy2(__file__,OUT/Path(__file__).name);rows=[];lawrows=[]
for case in (0,1):
    d=np.load(ROOT/f'results/iteration78_pilot_v2/table_{case+2}.npz');v,t,x,cov,target=d['v'],d['t'],d['x'],d['cov'],d['target'];m,n=v.shape;count=65536 if case==0 else 8192;B=8 if case==0 else 128;K=4 if case==0 else 5;methods=[('table',1),('table',16),('alias',1),('alias',16),('fused',B),('cpu',K),('gpu',K)];np.savez_compressed(OUT/f'input_{case}.npz',v=v,t=t,x=x,cov=cov,target=target);patterns=np.array([[(mask>>i)&1 for i in range(m)] for mask in range(1<<m)],np.uint8);draws=np.where(patterns,0.,np.nextafter(1.,0.))
    with NativePhiloxClickSampler(v,x,cpu,gpu_library=fused,mean=np.zeros(2*n),runtime_dirs=dirs) as model:r=model.replay(np.zeros((1<<m,2*n)),draws,branch_budget=B)
    law=np.prod(np.where(patterns,r['conditional_click_probabilities'],1-r['conditional_click_probabilities']),axis=1);delta=float(abs(law-target).sum()/2);assert np.array_equal(patterns,r['clicks']) and delta<1e-10;np.savez_compressed(OUT/f'fused_law_{case}.npz',patterns=patterns,probabilities=r['conditional_click_probabilities'],law=law,target=target);lawrows.append(dict(case=case,branch_budget=B,full_tv=delta))
    def run(kind,budget,count,rep):
        seed=0x92000000000000+(case*16+rep)*2**40;start=time.perf_counter()
        if kind in ('table','alias'):
            cls,binary=(NativeTableClickSampler,table) if kind=='table' else (NativeAliasClickSampler,alias)
            with cls(cov,binary,workers=budget,runtime_dirs=dirs) as model:setup=time.perf_counter()-start;r=model.sample(count,seed)
            bits=r['clicks'];meta=dict(setup_seconds=setup,native_sampling_seconds=r['wall_seconds'])
        elif kind=='fused':
            with NativePhiloxClickSampler(v,x,cpu,gpu_library=fused,mean=np.zeros(2*n),branch_budget=budget,runtime_dirs=dirs) as model:setup=time.perf_counter()-start;r=model.sample(count,seed)
            bits=r['clicks'];meta=dict(setup_seconds=setup,cuda_timings=r['cuda_timings'])
        else:
            bits=np.zeros((count,m),np.uint8);times=np.zeros(7 if kind=='cpu' else 6);error=C.create_string_buffer(4096);fn=libs[kind].fc73_sample if kind=='cpu' else libs[kind].fg72_sample;rc=fn(m,budget,count,788001+case*16+rep,16,ptr(cov),bits.ctypes.data_as(C.POINTER(C.c_uint8)),ptr(times),error,len(error));assert rc==0,error.value;meta=dict(component_timings=times.tolist())
        return bits,dict(wall_seconds=time.perf_counter()-start,**meta)
    with NativeAliasClickSampler(cov,alias,runtime_dirs=dirs) as model:alias_law=model.law()['probabilities']
    assert abs(alias_law-target).sum()/2<1e-12
    np.savez_compressed(OUT/f'alias_law_{case}.npz',law=alias_law,target=target)
    for kind,budget in methods:run(kind,budget,1,0)
    for rep in range(3):
        for kind,budget in methods[rep:]+methods[:rep]:
            label=f'{case}_{rep}_{kind}_{budget}';save(f'started_{label}.json',dict(case=case,rep=rep,kind=kind,budget=budget,records=count));bits,row=run(kind,budget,count,rep);row.update(case=case,rep=rep,kind=kind,budget=budget,records=count,outputs=m);np.savez_compressed(OUT/f'counts_{label}.npz',clicks=bits);rows.append(row);save(f'row_{label}.json',row);print(label,row['wall_seconds'],flush=True)
        for method in ('table','alias'):
            a=np.load(OUT/f'counts_{case}_{rep}_{method}_1.npz')['clicks'];b=np.load(OUT/f'counts_{case}_{rep}_{method}_16.npz')['clicks'];assert np.array_equal(a,b)
medians={f'{case}_{kind}_{budget}':statistics.median(r['wall_seconds'] for r in rows if (r['case'],r['kind'],r['budget'])==(case,kind,budget)) for case,kind,budget in set((r['case'],r['kind'],r['budget']) for r in rows)};save('summary.json',dict(passed=True,rows=rows,medians=medians,full_fused_laws=lawrows));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
