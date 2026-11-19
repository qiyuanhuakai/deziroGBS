"""Fixed two squeezed inputs, 144 outputs: vacuum event law and three timing repetitions.
The prefix coupling bound is ideal arithmetic/RNG, not a floating point certificate.
Frugal event error is a full-TV lower bound for the exported numerical law.
"""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time,ctypes as C,statistics,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
mp.mp.dps=80
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_prefix_pnr import NativePrefixPNRSampler
from gbslab.native_philox_click import NativePhiloxClickSampler
from gbslab.frugal_law import zero_probability
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
DP=C.POINTER(C.c_double)
def ptr(a):return a.ctypes.data_as(DP)
gcc=Path(shutil.which('g++')).parent;cuda=Path(shutil.which('nvcc')).parent;omp=next((Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Redist').glob('MSVC/*/debug_nonredist/x64/Microsoft.VC143.OpenMP.LLVM/libomp140.x86_64.dll'));dirs=[gcc,cuda,omp.parent];handles=[os.add_dll_directory(str(p)) for p in dirs]
paths={key:ROOT/value for key,value in dict(prefix='results/iteration75_prefix_pilot_v1/prefix.dll',zero='results/iteration75_zero_trace_build_v1/zero.dll',frugal='results/iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll',cpu='results/iteration72_cuda_pilot_v1/cpu.dll',fused='results/iteration74_fused_pilot_v2/philox.dll').items()}
lib=C.CDLL(str(paths['frugal']));lib.fc73_sample.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint,C.c_int,DP,C.POINTER(C.c_uint8),DP,C.c_char_p,C.c_int]
trace=C.CDLL(str(paths['zero']));trace.fc75_zero.argtypes=[C.c_int,C.c_int,C.c_int,DP,DP,DP,DP,C.c_char_p,C.c_int]
m=144;n=2;rng=np.random.default_rng(759001);v=np.ascontiguousarray(np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0]);t=np.full(n,np.tanh(1.1));x=np.diag(np.r_[-t/(1+t),t/(1-t)]);o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.ascontiguousarray(np.eye(2*m)+2*o@x@o.T);np.savez_compressed(OUT/'input.npz',v=v,t=t,x=x,cov=cov)
save('plan.json',dict(scope=__doc__,modes=m,inputs=n,records=4096,repetitions=3,input_seed=759001,pnr_seed_base=0x80000000000000,seed_stride=2**40,frugal_seed=759101,workers=16,source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),*paths.values()]}));shutil.copy2(__file__,OUT/Path(__file__).name)
# Sylvester determinant identity reduces the physical numerical vacuum to 4x4.
O=mp.matrix([[mp.mpf(float(z)) for z in row] for row in o]);X=mp.matrix(x.tolist());vacuum=1/mp.sqrt(mp.det(mp.eye(2*n)+X*O.T*O));idealvac=1-mp.mpf(float(t[0]))**2;bound=mp.mpf(float(t[0]))**32
save('target.json',dict(vacuum=str(vacuum),ideal_isometry_vacuum=str(idealvac),numerical_isometry_vacuum_difference=str(abs(vacuum-idealvac)),isometry_error=float(abs(v.conj().T@v-np.eye(n)).max()),ideal_prefix31_tv_bound=str(bound),bound_scope='Exact isometry, real arithmetic and ideal RNG. No global floating/PRNG certificate.'))
events=[]
for k in (1,2,3):
    q=np.zeros(m);previous=np.zeros(m);proposed=np.zeros(m);error=C.create_string_buffer(4096);start=time.perf_counter();rc=trace.fc75_zero(m,k,16,ptr(cov),ptr(q),ptr(previous),ptr(proposed),error,len(error));assert rc==0,error.value
    np.savez_compressed(OUT/f'zero_{k}.npz',threshold=q,previous=previous,proposed=proposed)
    row=dict(cutoff=k,seconds=time.perf_counter()-start,nonfinite=int((~np.isfinite(q)).sum()),saturated=int(((q<0)|(q>1)).sum()),models={})
    for model in ('continuous','uint64'):
        prob=math.prod(zero_probability(float(z),model) for z in q);value=mp.mpf(prob.numerator)/prob.denominator;row['models'][model]=dict(vacuum_probability=str(value),full_tv_lower_bound=str(abs(value-vacuum)),lower_over_ideal_prefix_upper=str(abs(value-vacuum)/bound))
    events.append(row);save(f'event_{k}.json',row);print('event',k,row['models']['uint64'],flush=True)
methods=[('prefix',31),('fused',4),('frugal',2),('frugal',3)]
def run(kind,k,count,rep):
    start=time.perf_counter();counts=None;seed=0x80000000000000+rep*2**40
    if kind=='prefix':
        with NativePrefixPNRSampler(v,t,paths['prefix'],retained=k,workers=16,runtime_dirs=dirs) as model:r=model.sample(count,seed)
        counts=r['counts'];bits=(counts>0).astype(np.uint8);meta=dict(statistics=r['statistics'])
    elif kind=='fused':
        with NativePhiloxClickSampler(v,x,paths['cpu'],gpu_library=paths['fused'],branch_budget=k,runtime_dirs=dirs) as model:r=model.sample(count,seed)
        bits=r['clicks'];meta=dict(statistics=r['statistics'])
    else:
        bits=np.zeros((count,m),np.uint8);times=np.zeros(7);error=C.create_string_buffer(4096);rc=lib.fc73_sample(m,k,count,759101+rep,16,ptr(cov),bits.ctypes.data_as(C.POINTER(C.c_uint8)),ptr(times),error,len(error));assert rc==0,error.value;meta=dict(component_timings=times.tolist())
    return counts,bits,dict(wall_seconds=time.perf_counter()-start,**meta)
for kind,k in methods:run(kind,k,1,0)
rows=[]
for rep in range(3):
    for kind,k in methods[rep:]+methods[:rep]:
        label=f'{rep}_{kind}_{k}';save(f'started_{label}.json',dict(rep=rep,kind=kind,budget=k));counts,bits,row=run(kind,k,4096,rep);row.update(rep=rep,kind=kind,budget=k,records=4096,zero_records=int((bits.sum(axis=1)==0).sum()),mean_clicks=float(bits.sum(axis=1).mean()));np.savez_compressed(OUT/f'records_{label}.npz',clicks=bits,**({} if counts is None else dict(counts=counts)));rows.append(row);save(f'row_{label}.json',row);print(label,row['wall_seconds'],flush=True)
medians={f'{kind}_{k}':statistics.median(r['wall_seconds'] for r in rows if (r['kind'],r['budget'])==(kind,k)) for kind,k in methods}
save('summary.json',dict(passed=True,events=events,rows=rows,medians=medians));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
