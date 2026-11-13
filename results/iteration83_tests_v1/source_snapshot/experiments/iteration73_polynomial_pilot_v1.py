"""Build polynomial photon-jump core; retain every success and failure."""
from pathlib import Path
import os,ctypes as C,json,hashlib,shutil,subprocess,time,argparse,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_double))
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';sources=[Path(__file__)]+[ROOT/'native'/name for name in ('gaussian_pnr_polynomial73_v1.cpp','gaussian_direct_click.cpp','gaussian_reset_cpu.cpp')]
for p in sources:shutil.copy2(p,OUT/p.name)
command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(sources[1]),'-o',str(OUT/'polynomial.dll')]
save('plan.json',dict(command=command,source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},cases=[0,1,2,3,4],retained_photon_factors=[0,1,2,3],samples=20000,workers=16,seed_base=731001,scope='Analytic photon subtraction factors, exact Gaussian Wick moments within each retained block, moment compression after K+1 jumps. No finite tap. Approximate after compression; no photon cutoff. Pilot compares complete PNR pattern histograms and retains failed batches.'))
start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start));assert r.returncode==0,r.stderr;print('polynomial core built',flush=True)
runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(OUT/'polynomial.dll'));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32);error=C.create_string_buffer(4096)
lib.tc70_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];lib.tc70_create.restype=C.c_void_p;lib.tc70_destroy.argtypes=[C.c_void_p]
lib.pnr74_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int]
rows=[]
for case in range(5):
    data=np.load(ROOT/f'results/iteration71_pnr_branch_pilot_v2/input_{case}.npz');v=data['v'];x=data['x'];mean=data['mean'];variance=data['variance'];h=lib.tc70_create(v.shape[0],v.shape[1],ptr(v),ptr(x),error,len(error));assert h,error.value
    for retained in (0,1,2,3):
        output=np.zeros((20000,4),np.uint32);stats=np.zeros(9);seed=731001+100*case;start=time.perf_counter();rc=lib.pnr74_sample(h,len(output),seed,16,retained,ptr(mean),ptr(variance),output.ctypes.data_as(UP),ptr(stats),error,len(error));elapsed=time.perf_counter()-start
        row=dict(case=case,retained=retained,seed=seed,returncode=rc,error=error.value.decode() if rc else '',seconds=elapsed,stats=stats.tolist());np.savez_compressed(OUT/f'counts_{case}_{retained}.npz',counts=output,stats=stats)
        if rc==0:
            cutoff=int(data['cutoff']);codes=data['codes'];base=cutoff+1;lookup=np.full(base**4,-1,np.int32);lookup[codes]=np.arange(len(codes));inside=output.sum(axis=1)<=cutoff;labels=np.full(len(output),len(codes),np.int32);labels[inside]=lookup[(output[inside].astype(np.uint64)@np.array([base**j for j in range(4)],np.uint64)).astype(np.int64)];hist=np.bincount(labels,minlength=len(data['target']));row.update(empirical_tv=float(abs(hist/len(output)-data['target']).sum()/2),mean_photons=float(output.sum()/len(output)),events_ge128=int(np.count_nonzero(output.sum(axis=1)>=128)));np.savez_compressed(OUT/f'hist_{case}_{retained}.npz',histogram=hist,target=data['target'])
        rows.append(row);save(f'row_{case}_{retained}.json',row);print(row,flush=True)
    lib.tc70_destroy(h)
save('summary.json',dict(completed=True,rows=rows,all_batches_succeeded=all(r['returncode']==0 for r in rows),binary_sha256=H(OUT/'polynomial.dll')))
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
