"""Beyond 50 occupied inputs: matched fused CUDA, Frugal CPU and Frugal GPU.
Three rotated repetitions, 512 returned bit arrays each, setup included.
These are finite-scale throughput panels, not an equal-full-TV certificate.
"""
from pathlib import Path
import os,ctypes as C,json,hashlib,shutil,time,argparse,sys,subprocess,statistics
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_philox_click import NativePhiloxClickSampler
from gbslab.native_gaussian_reset import lossy_squeezed_components
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_double))
gcc=Path(shutil.which('g++')).parent;cuda=Path(shutil.which('nvcc')).parent;omp=next((Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Redist').glob('MSVC/*/debug_nonredist/x64/Microsoft.VC143.OpenMP.LLVM/libomp140.x86_64.dll'));dirs=[gcc,cuda,omp.parent];handles=[os.add_dll_directory(str(p)) for p in dirs]
cpu=ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll';fused=ROOT/'results/iteration74_fused_pilot_v2/philox.dll';old=ROOT/'results/iteration73_philox_pilot_v1/philox.dll'
paths={'cpu64':ROOT/'results/iteration73_frugal_cpu_library_v1/frugal_cpu_fp64.dll','gpu64':ROOT/'results/iteration72_frugal_gpu_build_v5/frugal_fp64.dll'}
libs={kind:C.CDLL(str(p)) for kind,p in paths.items()}
for kind,lib in libs.items():
    fn=lib.fc73_sample if kind=='cpu64' else lib.fg72_sample;fn.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint,C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_uint8),C.POINTER(C.c_double),C.c_char_p,C.c_int];fn.restype=C.c_int
methods=[('fused',1),('fused',4),('fused',8),('cpu64',2),('cpu64',3),('gpu64',2)];sizes=[(192,64),(288,100),(384,128)];count=512
sourcepaths=[Path(__file__),cpu,fused,old,*paths.values(),ROOT/'src/gbslab/native_philox_click.py',ROOT/'src/gbslab/native_gaussian_reset.py']
save('plan.json',dict(scope=__doc__,source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sourcepaths},sizes=sizes,samples=count,repetitions=3,methods=methods,squeezing=1.4,transmission=.43,fixture_seed=752000,sample_seed=752001,workers=16))
shutil.copy2(__file__,OUT/Path(__file__).name);q=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,errors='replace');save('device.json',dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr));rows=[];replays=[]
for ci,(m,n) in enumerate(sizes):
    rng=np.random.default_rng(752000+ci);v=np.ascontiguousarray(np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0]);x,variance=lossy_squeezed_components(n,1.4,.43);mean=np.zeros(2*n);o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.ascontiguousarray(np.eye(2*m)+2*o@(x+np.diag(variance))@o.T);np.savez_compressed(OUT/f'input_{ci}.npz',v=v,x=x,mean=mean,variance=variance,cov=cov)
    # Reuse precisely the same random stream to test implementation at new sizes.
    for budget in (1,4,8):
        values=[]
        for binary in (old,fused):
            with NativePhiloxClickSampler(v,x,cpu,gpu_library=binary,mean=mean,noise_variance=variance,branch_budget=budget,runtime_dirs=dirs) as model:values.append(model.sample(32,752901+ci))
        a,b=values;row=dict(case=ci,inputs=n,outputs=m,budget=budget,different_bits=int(np.count_nonzero(a['clicks']!=b['clicks'])),maximum_probability_difference=float(abs(a['conditional_click_probabilities']-b['conditional_click_probabilities']).max()));replays.append(row);np.savez_compressed(OUT/f'replay_{ci}_{budget}.npz',old_clicks=a['clicks'],new_clicks=b['clicks'],old_probabilities=a['conditional_click_probabilities'],new_probabilities=b['conditional_click_probabilities']);save(f'replay_{ci}_{budget}.json',row);assert row['different_bits']==0 and row['maximum_probability_difference']<1e-8,row
    for rep in range(3):
        for kind,budget in methods[2*rep:]+methods[:2*rep]:
            label=f'{ci}_{rep}_{kind}_{budget}';seed=752001+10000*ci+100*rep;row=dict(case=ci,inputs=n,outputs=m,rep=rep,method=kind,budget=budget,samples=count,seed=seed);save(f'started_{label}.json',row);start=time.perf_counter()
            try:
                if kind=='fused':
                    with NativePhiloxClickSampler(v,x,cpu,gpu_library=fused,mean=mean,noise_variance=variance,branch_budget=budget,runtime_dirs=dirs) as model:r=model.sample(count,seed)
                    bits=r['clicks'];row.update(returncode=0,wall_seconds=time.perf_counter()-start,cuda_timings=r['cuda_timings'],statistics=r['statistics']);np.savez_compressed(OUT/f'probabilities_{label}.npz',conditional=r['conditional_click_probabilities'],diagnostics=r['trajectory_diagnostics'])
                else:
                    bits=np.zeros((count,m),np.uint8);timings=np.zeros(7 if kind=='cpu64' else 6);error=C.create_string_buffer(4096);fn=libs[kind].fc73_sample if kind=='cpu64' else libs[kind].fg72_sample;rc=fn(m,budget,count,seed,16,ptr(cov),bits.ctypes.data_as(C.POINTER(C.c_uint8)),ptr(timings),error,len(error));row.update(returncode=rc,wall_seconds=time.perf_counter()-start,error=error.value.decode(errors='replace') if rc else '',component_timings=timings.tolist())
                np.savez_compressed(OUT/f'records_{label}.npz',clicks=bits);row['observed_mean_clicks']=float(bits.sum(axis=1).mean())
            except Exception as e:row.update(returncode=-1,wall_seconds=time.perf_counter()-start,error=str(e))
            rows.append(row);save(f'row_{label}.json',row);print(label,row['wall_seconds'],row['returncode'],flush=True)
medians={str(ci):{f'{kind}_{budget}':statistics.median(r['wall_seconds'] for r in rows if (r['case'],r['method'],r['budget'])==(ci,kind,budget)) for kind,budget in methods} for ci in range(len(sizes))}
save('summary.json',dict(passed=all(r['returncode']==0 for r in rows),rows=rows,replays=replays,medians=medians));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
