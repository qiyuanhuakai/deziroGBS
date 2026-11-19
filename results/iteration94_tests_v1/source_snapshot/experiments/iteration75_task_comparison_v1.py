"""One strong pure input, complete click laws, and PNR-to-click/direct timings.
Three rotated repetitions of 65536 records; complete setup and output conversion
included. PNR and threshold tasks are distinguished; the direct samplers do not
provide integer photon records. Common PNR seeds enable deterministic coupling checks.
"""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time,ctypes as C,subprocess,statistics
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
mp.mp.dps=80
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_structured_pnr import NativeStructuredPNRSampler
from gbslab.native_deep_pnr import NativeDeepPNRSampler
from gbslab.native_resplit_pnr import NativeResplitPNRSampler
from gbslab.native_prefix_pnr import NativePrefixPNRSampler
from gbslab.native_philox_click import NativePhiloxClickSampler
from gbslab.frugal_law import full_law
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_double))
gcc=Path(shutil.which('g++')).parent;cuda=Path(shutil.which('nvcc')).parent;omp=next((Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Redist').glob('MSVC/*/debug_nonredist/x64/Microsoft.VC143.OpenMP.LLVM/libomp140.x86_64.dll'));dirs=[gcc,cuda,omp.parent];handles=[os.add_dll_directory(str(p)) for p in dirs]
pnr={'structured':(NativeStructuredPNRSampler,ROOT/'results/iteration74_structured_pilot_v1/structured.dll'),'deep':(NativeDeepPNRSampler,ROOT/'results/iteration75_deep_pilot_v1/deep.dll'),'resplit':(NativeResplitPNRSampler,ROOT/'results/iteration75_resplit_pilot_v1/resplit.dll'),'prefix':(NativePrefixPNRSampler,ROOT/'results/iteration75_prefix_pilot_v1/prefix.dll')};cpu=ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll';fused=ROOT/'results/iteration74_fused_pilot_v2/philox.dll';frugal_path=ROOT/'results/iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll';gpu_path=ROOT/'results/iteration75_frugal_gpu_build_v3/frugal_fp64.dll';exe=ROOT/'results/iteration73_frugal_cpu_build_v5/frugal_avx2_fp64.exe'
libs={'cpu':C.CDLL(str(frugal_path)),'gpu':C.CDLL(str(gpu_path))}
for kind,lib in libs.items():
    fn=lib.fc73_sample if kind=='cpu' else lib.fg72_sample;fn.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint,C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_uint8),C.POINTER(C.c_double),C.c_char_p,C.c_int];fn.restype=C.c_int
data=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');v,t=data['v'],data['t'];m,n=v.shape;x=np.diag(np.r_[-t/(1+t),t/(1-t)]);mean=np.zeros(2*n);o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.ascontiguousarray(np.eye(2*m)+2*o@x@o.T);np.savez_compressed(OUT/'input.npz',v=v,t=t,x=x,mean=mean,cov=cov);np.savetxt(OUT/'covariance.csv',cov,delimiter=',',fmt='%.17g')
methods=[('structured',3),('deep',3),('deep',7),('resplit',7),('prefix',7),('prefix',15),('prefix',23),('prefix',31),('fused',8),('cpu',4),('gpu',4)];paths=[Path(__file__),cpu,fused,frugal_path,gpu_path,exe,*[p for _,p in pnr.values()]];save('plan.json',dict(scope=__doc__,methods=methods,samples=65536,repetitions=3,pnr_seed_base=0x7f000000000000,seed_stride=2**40,frugal_seed=758101,workers=16,source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in paths}));shutil.copy2(__file__,OUT/Path(__file__).name)
V=mp.matrix([[mp.mpc(z) for z in row] for row in v]);O=mp.matrix(2*m,2*n)
for i in range(m):
    for j in range(n):O[i,j]=V[i,j].real;O[i,n+j]=-V[i,j].imag;O[m+i,j]=V[i,j].imag;O[m+i,n+j]=V[i,j].real
Q=mp.eye(2*m)+O*mp.matrix(x.tolist())*O.T;vac=[mp.mpf(1)]
for mask in range(1,2**m):
    ids=[i for i in range(m) if mask>>i&1];ids+=[i+m for i in ids];vac.append(1/mp.sqrt(mp.det(mp.matrix([[Q[i,j] for j in ids] for i in ids]))))
target=[]
for mask in range(2**m):
    zero=(2**m-1)^mask;sub=mask;terms=[]
    while True:
        terms.append((-1)**sub.bit_count()*vac[zero|sub])
        if sub==0:break
        sub=(sub-1)&mask
    target.append(mp.fsum(terms))
save('target.json',dict(precision_digits=80,probabilities=[str(p) for p in target]));target=np.array([float(p) for p in target]);lawrows=[];patterns=np.array([[(mask>>i)&1 for i in range(m)] for mask in range(2**m)],np.uint8);draws=np.where(patterns,0.,np.nextafter(1.,0.))
with NativePhiloxClickSampler(v,x,cpu,gpu_library=fused,mean=mean,runtime_dirs=dirs) as model:
    for budget in (1,2,4,8):
        r=model.replay(np.zeros((2**m,2*n)),draws,branch_budget=budget);law=np.prod(np.where(patterns,r['conditional_click_probabilities'],1-r['conditional_click_probabilities']),axis=1);assert np.array_equal(patterns,r['clicks']);row=dict(kind='fused',budget=budget,tv=float(abs(law-target).sum()/2),mass_error=float(abs(law.sum()-1)));lawrows.append(row);np.savez_compressed(OUT/f'fused_law_{budget}.npz',patterns=patterns,probabilities=r['conditional_click_probabilities'],law=law,target=target);assert row['mass_error']<1e-10
for cutoff in (1,2,3,4):
    cmd=[str(exe),'--input',str(OUT/'covariance.csv'),'--modes','4','--cutoff',str(cutoff),'--samples','128','--seed','758001','--threads','16','--audit','--prefix-tree'];r=subprocess.run(cmd,capture_output=True,text=True,errors='replace');save(f'frugal_process_{cutoff}.json',dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr;result=json.loads(r.stdout);assert result['audit_identical_outputs'];save(f'frugal_export_{cutoff}.json',result)
    laws={model:full_law(4,result['prefix_tree'],model=model) for model in ('continuous','uint64')};saved={model:{key:([str(p) for p in value] if key=='distribution' else value) for key,value in law.items()} for model,law in laws.items()};save(f'frugal_laws_{cutoff}.json',saved)
    for model,law in laws.items():lawrows.append(dict(kind='frugal_'+model,budget=cutoff,tv=float(abs(np.array(list(map(float,law['distribution'])))-target).sum()/2),nonfinite=len(law['reachable_nonfinite_prefixes']),saturated=len(law['reachable_saturated_prefixes'])))
save('law_summary.json',dict(passed=True,rows=lawrows));print('Complete fused and Frugal laws checked',flush=True)
def run(kind,budget,count,rep):
    seed=0x7f000000000000+rep*2**40;start=time.perf_counter();counts=None
    if kind in pnr:
        cls,binary=pnr[kind]
        with cls(v,t,binary,retained=budget,workers=16,runtime_dirs=dirs) as model:r=model.sample(count,seed)
        counts=r['counts'];bits=(counts>0).astype(np.uint8);meta=dict(statistics=r['statistics'],native_sampling_seconds=r['wall_seconds'])
    elif kind=='fused':
        with NativePhiloxClickSampler(v,x,cpu,gpu_library=fused,mean=mean,branch_budget=budget,runtime_dirs=dirs) as model:r=model.sample(count,seed)
        bits=r['clicks'];meta=dict(cuda_timings=r['cuda_timings'],statistics=r['statistics'])
    else:
        bits=np.zeros((count,m),np.uint8);times=np.zeros(7 if kind=='cpu' else 6);error=C.create_string_buffer(4096);fn=libs[kind].fc73_sample if kind=='cpu' else libs[kind].fg72_sample;rc=fn(m,budget,count,758101+rep,16,ptr(cov),bits.ctypes.data_as(C.POINTER(C.c_uint8)),ptr(times),error,len(error));assert rc==0,error.value;meta=dict(component_timings=times.tolist())
    return counts,bits,dict(wall_seconds=time.perf_counter()-start,**meta)
for kind,budget in methods:run(kind,budget,1,0)
rows=[]
for rep in range(3):
    for kind,budget in methods[3*rep:]+methods[:3*rep]:
        label=f'{rep}_{kind}_{budget}';save(f'started_{label}.json',dict(rep=rep,kind=kind,budget=budget));counts,bits,row=run(kind,budget,65536,rep);row.update(rep=rep,kind=kind,budget=budget,records=65536);np.savez_compressed(OUT/f'records_{label}.npz',clicks=bits,**({} if counts is None else {'counts':counts}));rows.append(row);save(f'row_{label}.json',row);print(label,row['wall_seconds'],flush=True)
couplings=[]
for rep in range(3):
    reference=np.load(OUT/f'records_{rep}_prefix_31.npz')['counts'];total=reference.sum(axis=1)
    for k in (7,15,23):
        counts=np.load(OUT/f'records_{rep}_prefix_{k}.npz')['counts'];different=np.any(counts!=reference,axis=1);assert not different[total<=k].any() and np.all(counts[different].sum(axis=1)>=k+1);couplings.append(dict(rep=rep,retained=k,different_records=int(different.sum()),reference_reaching_compression=int(np.count_nonzero(total>=k+1))))
medians={f'{kind}_{budget}':statistics.median(r['wall_seconds'] for r in rows if (r['kind'],r['budget'])==(kind,budget)) for kind,budget in methods};save('summary.json',dict(passed=True,rows=rows,medians=medians,couplings=couplings));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
