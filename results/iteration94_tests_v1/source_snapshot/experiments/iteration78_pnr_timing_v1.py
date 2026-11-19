"""Rotated setup-inclusive sphere/iid PNR, direct CUDA and Frugal CPU timings.
Three input panels. Only case1 compares larger-output Frugal on identical input.
"""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,time,ctypes as C,statistics
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_budget_number_pnr import NativeBudgetNumberPNRSampler
from gbslab.native_sphere_number_pnr import NativeSphereNumberPNRSampler
from gbslab.native_philox_click import NativePhiloxClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
ptr=lambda a:a.ctypes.data_as(C.POINTER(C.c_double));gcc=Path(shutil.which('g++')).parent;cuda=Path(shutil.which('nvcc')).parent;dirs=[gcc,cuda];handles=[os.add_dll_directory(str(p)) for p in dirs];models={'iid':(NativeBudgetNumberPNRSampler,ROOT/'results/iteration77_budget_pilot_v1/budget.dll'),'sphere':(NativeSphereNumberPNRSampler,ROOT/'results/iteration78_pilot_v2/sphere.dll')};cpu=ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll';fused=ROOT/'results/iteration74_fused_pilot_v2/philox.dll';frugal=ROOT/'results/iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll';lib=C.CDLL(str(frugal));fn=lib.fc73_sample;fn.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint,C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_uint8),C.POINTER(C.c_double),C.c_char_p,C.c_int];fn.restype=C.c_int
cases=[('results/iteration74_pure_quality_v1/input_1.npz',65536,[('iid',8),('sphere',8),('fused',8),('cpu',4)]),('results/iteration77_budget_pilot_v1/input_3.npz',512,[('iid',12),('sphere',12),('fused',4),('cpu',2),('cpu',3)]),('results/iteration77_budget_pilot_v1/input_4.npz',512,[('iid',12),('sphere',12),('fused',4)])]
save('plan.json',dict(scope=__doc__,cases=cases,workers=16,repetitions=3,seed_base=0x95000000000000,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),cpu,fused,frugal,*[p for _,p in models.values()]]}));shutil.copy2(__file__,OUT/Path(__file__).name);rows=[]
for case,(source,count,methods) in enumerate(cases):
    d=np.load(ROOT/source);v,t=d['v'],d['t'];m,n=v.shape;t=np.full(n,float(t)) if np.ndim(t)==0 else t;x=np.diag(np.r_[-t/(1+t),t/(1-t)]);o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.ascontiguousarray(np.eye(2*m)+2*o@x@o.T);np.savez_compressed(OUT/f'input_{case}.npz',v=v,t=t,x=x,cov=cov)
    def run(kind,budget,count,rep):
        seed=0x95000000000000+(case*16+rep)*2**40;start=time.perf_counter();counts=totals=None
        if kind in models:
            cls,binary=models[kind]
            with cls(v,t,binary,exact_total_budget=budget,workers=16,runtime_dirs=dirs) as model:r=model.sample(count,seed)
            counts=r['counts'];totals=r['total_photons'];bits=(counts>0).astype(np.uint8);meta=dict(statistics=r['statistics'],native_sampling_seconds=r['wall_seconds'])
        elif kind=='fused':
            with NativePhiloxClickSampler(v,x,cpu,gpu_library=fused,mean=np.zeros(2*n),branch_budget=budget,runtime_dirs=dirs) as model:r=model.sample(count,seed)
            bits=r['clicks'];meta=dict(cuda_timings=r['cuda_timings'])
        else:
            bits=np.zeros((count,m),np.uint8);times=np.zeros(7);error=C.create_string_buffer(4096);rc=fn(m,budget,count,789001+case*16+rep,16,ptr(cov),bits.ctypes.data_as(C.POINTER(C.c_uint8)),ptr(times),error,len(error));assert rc==0,error.value;meta=dict(component_timings=times.tolist())
        return bits,counts,totals,dict(wall_seconds=time.perf_counter()-start,**meta)
    for kind,budget in methods:run(kind,budget,1,0)
    for rep in range(3):
        for kind,budget in methods[rep:]+methods[:rep]:
            label=f'{case}_{rep}_{kind}_{budget}';save(f'started_{label}.json',dict(case=case,rep=rep,kind=kind,budget=budget,records=count));bits,counts,totals,row=run(kind,budget,count,rep);row.update(case=case,rep=rep,kind=kind,budget=budget,records=count,inputs=n,outputs=m);np.savez_compressed(OUT/f'counts_{label}.npz',clicks=bits,**({} if counts is None else dict(counts=counts,totals=totals)));rows.append(row);save(f'row_{label}.json',row);print(label,row['wall_seconds'],flush=True)
        k=8 if case==0 else 12;a=np.load(OUT/f'counts_{case}_{rep}_iid_{k}.npz');b=np.load(OUT/f'counts_{case}_{rep}_sphere_{k}.npz');mask=a['totals']<=k;assert np.array_equal(a['totals'],b['totals']) and np.array_equal(a['counts'][mask],b['counts'][mask])
medians={f'{case}_{kind}_{budget}':statistics.median(r['wall_seconds'] for r in rows if (r['case'],r['kind'],r['budget'])==(case,kind,budget)) for case,kind,budget in set((r['case'],r['kind'],r['budget']) for r in rows)};save('summary.json',dict(passed=True,rows=rows,medians=medians));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
