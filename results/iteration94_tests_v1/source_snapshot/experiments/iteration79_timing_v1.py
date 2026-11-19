"""Three rotated, setup-inclusive timing repetitions on identical inputs.
Block width/B and actual GPU capacity are separate: capacity=B*2**(width-1).
Frugal K limits the conditional approximation; no equal-quality inference.
"""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,time,ctypes as C,statistics
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_joint_number_pnr import NativeJointNumberPNRSampler
from gbslab.native_sphere_number_pnr import NativeSphereNumberPNRSampler
from gbslab.native_philox_click import NativePhiloxClickSampler
from gbslab.native_block_philox_click import NativeBlockPhiloxClickSampler
from gbslab.native_block_click import NativeBlockClickSampler
from gbslab.native_alias_click import NativeAliasClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
ptr=lambda a:a.ctypes.data_as(C.POINTER(C.c_double));dirs=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent];handles=[os.add_dll_directory(str(p)) for p in dirs]
bins={k:ROOT/v for k,v in dict(joint='results/iteration79_pilot_v1/joint.dll',sphere='results/iteration78_pilot_v2/sphere.dll',cpu_replay='results/iteration72_cuda_pilot_v1/cpu.dll',old='results/iteration74_fused_pilot_v2/philox.dll',gpu='results/iteration79_gpu_pilot_v1/block.dll',block='results/iteration79_pilot_v1/block.dll',alias='results/iteration78_alias_pilot_v1/table.dll',frugal='results/iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll').items()}
lib=C.CDLL(str(bins['frugal']));fn=lib.fc73_sample;fn.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint,C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_uint8),C.POINTER(C.c_double),C.c_char_p,C.c_int];fn.restype=C.c_int
gpu_methods=[('old',1,4),('old',1,8),('old',1,32),('gpu',2,4),('gpu',4,4)]
cases=[('results/iteration79_pilot_v1/click_input_1.npz',8192,gpu_methods+[('old',1,128),('block',2,4),('block',4,4),('alias',1,0),('frugal',1,5)]),('results/iteration79_pilot_v1/large_input_3.npz',128,gpu_methods+[('block',2,4),('block',4,4),('joint',2,12),('sphere',1,12),('frugal',1,2),('frugal',1,3)]),('results/iteration79_pilot_v1/large_input_4.npz',128,gpu_methods+[('joint',2,12),('sphere',1,12),('frugal',1,2)])]
save('plan.json',dict(scope=__doc__,cases=cases,workers=16,alias_workers=1,repetitions=3,seed_base=0x9e000000000000,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),*bins.values()]}));shutil.copy2(__file__,OUT/Path(__file__).name);rows=[]
for case,(source,count,methods) in enumerate(cases):
    d=np.load(ROOT/source);v,x=d['v'],d['x'];m,n=v.shape;t=d['t'] if 't' in d else None;o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.ascontiguousarray(np.eye(2*m)+2*o@x@o.T);np.savez_compressed(OUT/f'input_{case}.npz',v=v,x=x,cov=cov,**({} if t is None else dict(t=t)))
    def run(kind,width,budget,count,rep):
        seed=0x9e000000000000+(case*16+rep)*2**40;start=time.perf_counter();counts=totals=None
        if kind in ('joint','sphere'):
            cls=NativeJointNumberPNRSampler if kind=='joint' else NativeSphereNumberPNRSampler
            with cls(v,t,bins[kind],exact_total_budget=budget,workers=16,runtime_dirs=dirs) as model:r=model.sample(count,seed)
            counts=r['counts'];totals=r['total_photons'];bits=(counts>0).astype(np.uint8);meta=dict(statistics=r['statistics'],native_sampling_seconds=r['wall_seconds'])
        elif kind in ('old','gpu'):
            cls=NativePhiloxClickSampler if kind=='old' else NativeBlockPhiloxClickSampler;kw={} if kind=='old' else dict(block_size=width)
            with cls(v,x,bins['cpu_replay'],gpu_library=bins[kind],branch_budget=budget,runtime_dirs=dirs,**kw) as model:r=model.sample(count,seed)
            bits=r['clicks'];meta=dict(cuda_timings=r['cuda_timings'],statistics=r['statistics'],workspace_capacity=budget*2**(width-1))
        elif kind=='block':
            with NativeBlockClickSampler(v,x,bins['block'],branch_budget=budget,block_size=width,workers=16,runtime_dirs=dirs) as model:r=model.sample(count,seed)
            bits=r['clicks'];meta=dict(statistics=r['statistics'])
        elif kind=='alias':
            with NativeAliasClickSampler(cov,bins['alias'],workers=1,runtime_dirs=dirs) as model:r=model.sample(count,seed)
            bits=r['clicks'];meta=dict(native_sampling_seconds=r['wall_seconds'])
        else:
            bits=np.zeros((count,m),np.uint8);times=np.zeros(7);error=C.create_string_buffer(4096);rc=fn(m,budget,count,799001+case*16+rep,16,ptr(cov),bits.ctypes.data_as(C.POINTER(C.c_uint8)),ptr(times),error,len(error));assert rc==0,error.value;meta=dict(component_timings=times.tolist())
        return bits,counts,totals,dict(wall_seconds=time.perf_counter()-start,**meta)
    for kind,width,budget in methods:
        try:run(kind,width,budget,1,0)
        except Exception as e:save(f'warmup_failure_{case}_{kind}_{width}_{budget}.json',dict(error=str(e)));print('warmup failure',kind,str(e),flush=True)
    for rep in range(3):
        for kind,width,budget in methods[rep:]+methods[:rep]:
            label=f'{case}_{rep}_{kind}_{width}_{budget}';save(f'started_{label}.json',dict(case=case,rep=rep,kind=kind,width=width,budget=budget,records=count));start=time.perf_counter()
            try:
                bits,counts,totals,row=run(kind,width,budget,count,rep);row['success']=True;np.savez_compressed(OUT/f'counts_{label}.npz',clicks=bits,**({} if counts is None else dict(counts=counts,totals=totals)))
            except Exception as e:row=dict(success=False,error=str(e),wall_seconds=time.perf_counter()-start)
            row.update(case=case,rep=rep,kind=kind,width=width,budget=budget,records=count,inputs=n,outputs=m);rows.append(row);save(f'row_{label}.json',row);print(label,row,flush=True)
        if t is not None:
            a=np.load(OUT/f'counts_{case}_{rep}_joint_2_12.npz');b=np.load(OUT/f'counts_{case}_{rep}_sphere_1_12.npz');mask=a['totals']<=12;assert np.array_equal(a['totals'],b['totals']) and np.array_equal(a['counts'][mask],b['counts'][mask])
medians={f'{case}_{kind}_{width}_{budget}':statistics.median(r['wall_seconds'] for r in rows if (r['case'],r['kind'],r['width'],r['budget'])==(case,kind,width,budget)) for case,kind,width,budget in set((r['case'],r['kind'],r['width'],r['budget']) for r in rows) if all(r['success'] for r in rows if (r['case'],r['kind'],r['width'],r['budget'])==(case,kind,width,budget))};save('summary.json',dict(completed=True,rows=rows,medians=medians));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
