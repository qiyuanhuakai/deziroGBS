"""Independent 2M-record confirmation, fixed after the 100k pilot."""
from pathlib import Path
import os,ctypes as C,hashlib,json,shutil,time,argparse,sys,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_gaussian_reset_parallel import NativeParallelGaussianResetSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_double))
pilot=ROOT/'results/iteration71_pnr_branch_pilot_v2';dll=pilot/'pnr_branches.dll';count=2000000;batch_size=100000;seed_base=0x74000000000000;stride=1<<40;radius=math.sqrt(math.log(8/.01)/(2*count))
save('plan.json',dict(source_sha256=H(Path(__file__)),library_sha256=H(dll),pilot_manifest_sha256=H(pilot/'manifest.json'),cases=[3,4],methods=['B2_eps0.001','iteration70_reset'],samples_per_method=count,batch_size=batch_size,seed_base=seed_base,stride=stride,condition_trigger=1e6,nominal_joint_failure_probability=.01,radius=radius,scope='Four fixed distributions, 8 nominal one-sided bounds under ideal IID and exact numerical target. Lower event fixed from prior pilot; upper empirical TV plus McDiarmid radius plus target tail. No count truncation, failures abort. Numerical cancellation remains a risk; condition trigger is at jump construction, not a global bound after survival flow.'))
shutil.copy2(__file__,OUT/Path(__file__).name);runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));lib=C.CDLL(str(dll));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32);error=C.create_string_buffer(4096)
lib.tc70_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];lib.tc70_create.restype=C.c_void_p;lib.tc70_destroy.argtypes=[C.c_void_p]
lib.pnr72_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,C.c_double,C.c_double,DP,DP,UP,DP,C.c_char_p,C.c_int]
rows=[]
for ci,case in enumerate((3,4)):
    data=np.load(pilot/f'input_{case}.npz');v=data['v'];x=data['x'];mean=data['mean'];variance=data['variance'];law=data['target'];codes=data['codes'];cutoff=int(data['cutoff']);base=cutoff+1;lookup=np.full(base**4,-1,np.int32);lookup[codes]=np.arange(len(codes));powers=np.array([base**j for j in range(4)],np.uint64)
    handle=lib.tc70_create(4,2,ptr(v),ptr(x),error,len(error));assert handle,error.value
    old=NativeParallelGaussianResetSampler(v,x,ROOT/'results/iteration70_build/gaussian_reset_parallel.dll',noise_variance=variance,workers=16,runtime_dirs=[Path(shutil.which('g++')).parent])
    for mi,method in enumerate(('branches','reset')):
        previous=np.load(pilot/(f'counts_{case}_2_1.npz' if method=='branches' else f'baseline_{case}.npz'))['histogram'];event=previous/previous.sum()>law;hist=np.zeros(len(law),np.int64);elapsed=0;maximum=0;photons=0;extreme=0;cancel=0
        for batch in range(count//batch_size):
            seed=seed_base+(ci*100+mi*30+batch)*stride
            if method=='branches':
                counts=np.zeros((batch_size,4),np.uint32);stats=np.zeros(7);start=time.perf_counter();rc=lib.pnr72_sample(handle,batch_size,seed,16,2,.001,1e6,ptr(mean),ptr(variance),counts.ctypes.data_as(UP),ptr(stats),error,len(error));seconds=time.perf_counter()-start
                save(f'status_{case}_{method}_{batch}.json',dict(returncode=rc,error=error.value.decode() if rc else '',seconds=seconds,seed=seed,stats=stats.tolist()));np.savez_compressed(OUT/f'counts_{case}_{method}_{batch}.npz',counts=counts,stats=stats);assert rc==0,error.value;cancel=max(cancel,stats[4])
            else:
                result=old.sample(batch_size,seed);counts=result['counts'];seconds=result['wall_seconds'];np.savez_compressed(OUT/f'counts_{case}_{method}_{batch}.npz',counts=counts);save(f'status_{case}_{method}_{batch}.json',dict(returncode=0,seconds=seconds,seed=seed,statistics=result['statistics']))
            totals=counts.sum(axis=1);inside=totals<=cutoff;labels=np.full(batch_size,len(codes),np.int32);labels[inside]=lookup[(counts[inside].astype(np.uint64)@powers).astype(np.int64)];assert labels.min()>=0;hist+=np.bincount(labels,minlength=len(law));elapsed+=seconds;maximum=max(maximum,int(totals.max()));photons+=int(totals.sum());extreme+=int(sum(totals>=128))
        empirical=hist/count;tv=float(abs(empirical-law).sum()/2);lower=max(0,float(empirical[event].sum()-law[event].sum())-radius);upper=min(1,tv+radius+law[-1]);row=dict(case=case,method=method,samples=count,seconds=elapsed,empirical_tv=tv,nominal_full_tv_interval=[lower,upper],maximum_photons=maximum,mean_photons=photons/count,events_ge128=extreme,maximum_absolute_weight_sum=cancel);rows.append(row);save(f'row_{case}_{method}.json',row);np.savez_compressed(OUT/f'histogram_{case}_{method}.npz',histogram=hist,target=law,fixed_event=event);print(row,flush=True)
    old.close();lib.tc70_destroy(handle)
save('summary.json',dict(passed=True,rows=rows,radius=radius,nominal_joint_failure_probability=.01))
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
