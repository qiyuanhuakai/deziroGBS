"""Full direct-threshold trajectories, with all failures retained."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import argparse,ctypes as C,hashlib,json,math,shutil,subprocess,time
from pathlib import Path
import numpy as np
from scipy.special import betaincinv
from iteration93_baseline_common_v1 import Baseline93

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
out=p.parse_args().output.resolve()
out.mkdir(parents=True,exist_ok=False)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(out/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
source=root/'native/gaussian_click_trajectory95_v1.cpp'
small=[root/f'results/iteration93_reference_v2/input_{j}.npz' for j in range(4)]
actual=[root/f'results/iteration94_actual_quality_v1/input_{j}.npz' for j in range(2)]
hardware=root/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz'
refs=[root/f'results/iteration93_reference_v2/law_{j}.npz' for j in range(4)]
files=[source,source.with_name('gaussian_click_proxy95_v1.cpp'),source.with_name('gaussian_rng_domain84_v1.hpp'),Path(__file__),
       root/'experiments/iteration93_baseline_common_v1.py',root/'results/iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll',hardware,*small,*actual,*refs]
save('plan.json',dict(scope=__doc__,hashes={f.relative_to(root).as_posix():digest(f) for f in files},
    actual_records=1024,batch=64,actual_budgets=[[1152,1152],[.1,.01]],family_failure_budget=.01,
    events=['hardware_count_DKW','loose_count_DKW','limited_count_DKW','loose_failure_upper','limited_failure_upper'],
    quality_experiments=[0x95030000,0x95030001],sectors=[0,1],purposes=[3,5],
    small_timing='2048 records, 3 serial rotations, 1 worker; prepared models',
    actual_timing='64 complete attempts, both spectra and OR, three serial repetitions; large budgets',
    small_tree_budgets=[.01,10],small_tree_cases=[0,1,2],witness_scope='Hardware count CDF distinct from full Gaussian click TV.'))
gcc=Path(shutil.which('g++'))
dll=out/'trajectory.dll'
build=subprocess.run([str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE',
    '-I',str(root.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(source),'-o',str(dll)],capture_output=True,text=True)
save('build.json',dict(returncode=build.returncode,stderr=build.stderr));build.check_returncode()
runtime=os.add_dll_directory(str(gcc.parent))
lib=C.CDLL(str(dll))
fp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
bp=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
lib.trajectory95_prepare.restype=C.c_void_p
lib.trajectory95_prepare.argtypes=[C.c_int,C.c_int,fp,fp,fp,fp]
lib.trajectory95_destroy.argtypes=[C.c_void_p]
lib.trajectory95_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_double,bp,fp,ip]
lib.trajectory95_tree.argtypes=[C.c_void_p,C.c_double,fp,fp]

def prepare(path):
    with np.load(path) as archive:
        q={k:np.ascontiguousarray(archive[k]) for k in archive.files}
    m,n=q['u'].shape
    begin=time.perf_counter()
    ptr=lib.trajectory95_prepare(m,n,q['u'].view(float),q['t'],q['alpha'].view(float),q['noise'])
    assert ptr
    return ptr,m,q,time.perf_counter()-begin
def sample(model,N,exp,sector,offset,budget):
    ptr,m,_,_=model
    bits=np.empty((N,m),np.uint8);stats=np.empty((N,6));errors=np.empty(N,np.int32)
    assert lib.trajectory95_sample(ptr,N,exp,sector,offset,budget,bits,stats,errors)==0
    return dict(clicks=bits,statistics=stats,errors=errors)
def audit(raw,m,budget):
    s=raw['statistics'];e=raw['errors']
    assert np.isin(raw['clicks'],[0,1]).all() and np.isin(e,[0,2,4]).all()
    assert np.array_equal(e==0,s[:,5].astype(bool)) and (s[e==0,0]==m).all()
    assert (s[:,1]>=0).all() and (s[:,1]<=budget).all() and (s[:,3]==s[:,0]).all()

models=[prepare(f) for f in small+actual]
save('setup.json',[dict(path=f.relative_to(root).as_posix(),seconds=model[3]) for f,model in zip(small+actual,models)])
trees=[]
for j in (0,1,2):
    model=models[j];m=model[1];target=np.r_[np.load(refs[j])['click'],0.]
    for budget in (.01,10.):
        law=np.empty(2**m+1);weighted=np.zeros(1)
        assert lib.trajectory95_tree(model[0],budget,law,weighted)==0
        distance=float(abs(law-target).sum()/2)
        assert min(law)>=0 and abs(law.sum()-1)<1e-12 and distance<=weighted[0]+law[-1]+1e-10
        np.savez_compressed(out/f'tree_{j}_{budget}.npz',law=law,target=target)
        trees.append(dict(case=j,budget=budget,full_tv=distance,failure=law[-1],weighted_charge=weighted[0]))
save('trees.json',trees)
timings=[]
for j in (0,2):
    model=models[j];ks=(2,4) if j==0 else (2,5)
    frugal={k:Baseline93(dict(family='frugal',case=j,cutoff=k,threads=1),model[2]) for k in ks}
    labels=[0,*ks]
    for rep in (-1,0,1,2):
        N=16 if rep<0 else 2048
        for label in labels[rep%3:]+labels[:rep%3]:
            exp=0x95040000+1024*(rep+1)+j*16+label
            begin=time.perf_counter()
            if label==0:
                raw=sample(model,N,exp,0,0,10.)
            else:raw,_=frugal[label].run(N,0,exp,exp)
            elapsed=time.perf_counter()-begin
            failed=raw['errors']!=0 if label==0 else raw['failed']
            assert not failed.any()
            np.savez_compressed(out/f'time_{j}_{rep}_{label}.npz',clicks=raw['clicks'],failed=failed)
            if rep>=0:timings.append(dict(case=j,worker=1,rep=rep,records=N,method='proxy' if label==0 else f'frugal{label}',seconds=elapsed))
    for item in frugal.values():item.close()
save('timing_small.json',timings)
delta=.01/5
hist=np.load(hardware)['total_click_histogram']
realcdf=np.r_[np.cumsum(hist)/hist.sum(),1.]
hardware_radius=math.sqrt(math.log(2/delta)/(2*hist.sum()))
quality=[]
for index,budgets in enumerate(((1152.,1152.),(.1,.01))):
    all_bits=[];all_failed=[];code_totals=[{},{}]
    for offset in range(0,1024,64):
        raws=[]
        for sector,model in enumerate(models[4:]):
            raw=sample(model,64,0x95030000+index,sector,offset,budgets[sector])
            audit(raw,1152,budgets[sector])
            np.savez_compressed(out/f'actual_{index}_{sector}_{offset}.npz',**raw)
            for code,n in zip(*np.unique(raw['errors'],return_counts=True)):
                key=str(code);code_totals[sector][key]=code_totals[sector].get(key,0)+int(n)
            raws.append(raw)
        all_bits.append(raws[0]['clicks']|raws[1]['clicks'])
        all_failed.append((raws[0]['errors']!=0)|(raws[1]['errors']!=0))
    bits=np.concatenate(all_bits);failed=np.concatenate(all_failed)
    values=bits.sum(1).astype(int);values[failed]=1153
    cdf=np.cumsum(np.bincount(values,minlength=1154))/len(values)
    ks=float(abs(cdf-realcdf).max());radius=hardware_radius+math.sqrt(math.log(2/delta)/(2*len(values)))
    k=int(failed.sum());pf=1. if k==len(values) else float(betaincinv(k+1,len(values)-k,1-delta))
    quality.append(dict(budgets=budgets,records=len(values),failures=k,codes=code_totals,failure_upper=pf,
        empirical_hardware_count_cdf=ks,lower=max(0.,ks-radius),upper=min(1.,ks+radius)))
    np.savez_compressed(out/f'combined_{index}.npz',bits=bits,failed=failed,cdf=cdf)
    print('actual quality',quality[-1],flush=True)
actual_timing=[]
for rep in (-1,0,1,2):
    N=4 if rep<0 else 64
    begin=time.perf_counter()
    pair=[sample(model,N,0x95050000+rep+1,sector,0,1152.) for sector,model in enumerate(models[4:])]
    bits=pair[0]['clicks']|pair[1]['clicks']
    failed=(pair[0]['errors']!=0)|(pair[1]['errors']!=0)
    elapsed=time.perf_counter()-begin
    np.savez_compressed(out/f'actual_time_{rep}.npz',bits=bits,failed=failed)
    if rep>=0:actual_timing.append(dict(rep=rep,records=N,failures=int(failed.sum()),seconds=elapsed))
for model in models:lib.trajectory95_destroy(model[0])
save('summary.json',dict(completed=True,trees=trees,small_timing=timings,actual_quality=quality,actual_timing=actual_timing,hardware_records=int(hist.sum())))
save('manifest.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()})
print('complete',actual_timing,flush=True)
