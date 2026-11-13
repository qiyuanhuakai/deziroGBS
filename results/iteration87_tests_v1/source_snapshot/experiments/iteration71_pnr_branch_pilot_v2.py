"""Finite-tap PNR branch prototype: preserve all failures and uncut counts."""
from pathlib import Path
import os,ctypes as C,hashlib,json,shutil,subprocess,time,argparse,sys,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_gaussian_reset_parallel import NativeParallelGaussianResetSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_double))
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';sources=[Path(__file__)]+[ROOT/'native'/n for n in ['gaussian_pnr_branches_v2.cpp','gaussian_click_branches_v1.cpp','gaussian_direct_click.cpp','gaussian_reset_cpu.cpp']]
for p in sources:shutil.copy2(p,OUT/p.name)
command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(sources[1]),'-o',str(OUT/'pnr_branches.dll')]
save('plan.json',dict(command=command,source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},samples=100000,cases=[3,4],budgets=[1,2,4],epsilon=[.01,.001],condition_limit=1e6,workers=16,seed=713001,scope='Condition-aware analytic moment compression at absolute signed weight limit 1e6. Finite-tap Gaussian difference approximates photon subtraction; signed survival flow and rates. Both finite epsilon bias and branch compression bias remain. No count truncation or retry after failed samples. Parameter pilot only, no finite-sample quality certificate. Independent dense optical beam-splitter conditional-moment checks.'))
start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start));assert r.returncode==0,r.stderr
print('PNR finite-tap branches built',flush=True)
runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(OUT/'pnr_branches.dll'));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32);error=C.create_string_buffer(4096)
lib.tc70_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];lib.tc70_create.restype=C.c_void_p;lib.tc70_destroy.argtypes=[C.c_void_p]
lib.pnr72_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,C.c_double,C.c_double,DP,DP,UP,DP,C.c_char_p,C.c_int]
lib.pnr71_jump_probe.argtypes=[C.c_void_p,DP,DP,C.c_int,C.c_double,DP,DP,DP,C.c_char_p,C.c_int]
rows=[];checks=[];reference=ROOT/'results/iteration70_archive/iteration70_pgf_quad_v1_20261110';desc=json.loads((reference/'summary.json').read_text())['rows']
for case in (3,4):
    data=np.load(reference/f'target_{case}.npz');v=np.ascontiguousarray(data['v']);m,n=v.shape;initial=np.load(ROOT/f'results/iteration70_archive/iteration70_noise_holdout_v1_20261110/case_{case}_theta_000.npz');x=np.ascontiguousarray(initial['x0']);variance=np.ascontiguousarray(initial['variance']);mean=np.zeros(2*n);cutoff=int(data['cutoff']);base=cutoff+1;codes=data['codes'];law=np.r_[data['probabilities'],float(desc[case]['analytic_target_tail'])];lookup=np.full(base**4,-1,np.int32);lookup[codes]=np.arange(len(codes));powers=np.array([base**j for j in range(4)],np.uint64);np.savez_compressed(OUT/f'input_{case}.npz',v=v,x=x,mean=mean,variance=variance,target=law,codes=codes,cutoff=cutoff)
    h=lib.tc70_create(m,n,ptr(v),ptr(x),error,len(error));assert h,error.value
    o=np.block([[v.real,-v.imag],[v.imag,v.real]]);w=np.random.default_rng(712100+case).normal(size=2*n)*np.sqrt(variance);physical_x=o@x@o.T;mu=o@w
    for eps in (.05,.01,.001):
        for mode in range(m):
            xx=np.zeros_like(x);ww=np.zeros_like(w);prob=np.zeros(1);rc=lib.pnr71_jump_probe(h,ptr(x),ptr(w),mode,eps,ptr(xx),ptr(ww),ptr(prob),error,len(error));assert rc==0,error.value
            ids=[mode,mode+m];a=np.eye(2)+eps*physical_x[np.ix_(ids,ids)];tapmu=np.sqrt(eps)*mu[ids];p0=float(np.exp(-.5*tapmu@np.linalg.solve(a,tapmu))/np.sqrt(np.linalg.det(a)));p1=1-p0
            cov=np.eye(2*m)/2+(1-eps)*physical_x;before=np.sqrt(1-eps)*mu;cross=np.sqrt(eps*(1-eps))*physical_x[:,ids];delta=cross@np.linalg.solve(a,tapmu);vac_cov=cov-cross@np.linalg.solve(a,cross.T);expected_mean=before+p0/p1*delta;expected_cov=(cov-p0*vac_cov)/p1-p0/(p1*p1)*np.outer(delta,delta)
            cov_error=float(abs(np.eye(2*m)/2+o@xx@o.T-expected_cov).max());mean_error=float(abs(o@ww-expected_mean).max());row=dict(case=case,epsilon=eps,mode=mode,probability_error=float(abs(prob[0]-p1)),covariance_error=cov_error,mean_error=mean_error);checks.append(row);assert max(cov_error,mean_error)<2e-8,row
    save(f'probes_{case}.json',[r for r in checks if r['case']==case])
    def evaluate(counts):
        totals=counts.sum(axis=1);inside=totals<=cutoff;labels=np.full(len(counts),len(codes),np.int32);labels[inside]=lookup[(counts[inside].astype(np.uint64)@powers).astype(np.int64)];assert labels.min()>=0;hist=np.bincount(labels,minlength=len(law));return hist,dict(empirical_tv=float(abs(hist/len(counts)-law).sum()/2),mean_photons=float(totals.mean()),maximum_photons=int(totals.max()),tail_events=int(sum(~inside)))
    for budget in (1,2,4):
        for ei,eps in enumerate((.01,.001)):
            count=100000;seed=713001+case*100+budget*10+ei;counts=np.zeros((count,m),np.uint32);stats=np.zeros(7);start=time.perf_counter();rc=lib.pnr72_sample(h,count,seed,16,budget,eps,1e6,ptr(mean),ptr(variance),counts.ctypes.data_as(UP),ptr(stats),error,len(error));seconds=time.perf_counter()-start
            row=dict(case=case,budget=budget,epsilon=eps,seed=seed,returncode=rc,error=error.value.decode() if rc else '',seconds=seconds,stats=stats.tolist());payload={}
            if rc==0:hist,quality=evaluate(counts);payload['histogram']=hist;row.update(quality)
            np.savez_compressed(OUT/f'counts_{case}_{budget}_{ei}.npz',counts=counts,stats=stats,**payload);save(f'row_{case}_{budget}_{ei}.json',row);rows.append(row);print(row,flush=True)
    with NativeParallelGaussianResetSampler(v,x,ROOT/'results/iteration70_build/gaussian_reset_parallel.dll',noise_variance=variance,workers=16,runtime_dirs=[gcc.parent]) as old:
        result=old.sample(100000,713901+case);counts=result['counts'];hist,quality=evaluate(counts);np.savez_compressed(OUT/f'baseline_{case}.npz',counts=counts,histogram=hist);row=dict(case=case,method='iteration70_reset',seconds=result['wall_seconds'],**quality);save(f'baseline_{case}.json',row);rows.append(row);print(row,flush=True)
    lib.tc70_destroy(h)
save('summary.json',dict(experiment_completed=True,all_batches_succeeded=all(r.get('returncode',0)==0 for r in rows),rows=rows,checks=checks,binary_sha256=H(OUT/'pnr_branches.dll')))
assert {p.relative_to(ROOT).as_posix():H(p) for p in sources}==json.loads((OUT/'plan.json').read_text())['source_hashes']
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
