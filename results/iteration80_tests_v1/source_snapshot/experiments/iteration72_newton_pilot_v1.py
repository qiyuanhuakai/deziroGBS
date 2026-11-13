"""PNR safeguarded-Newton replay and interleaved same-build CPU timings."""
from pathlib import Path
import os,ctypes as C,json,hashlib,shutil,subprocess,time,argparse
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_double))
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';sources=[Path(__file__)]+[ROOT/'native'/name for name in ('gaussian_pnr_newton72_v1.cpp','gaussian_click_branches_v1.cpp','gaussian_direct_click.cpp','gaussian_reset_cpu.cpp')]
for p in sources:shutil.copy2(p,OUT/p.name)
oldpath=ROOT/'results/iteration71_pnr_branch_pilot_v2/pnr_branches.dll';command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(sources[1]),'-o',str(OUT/'newton.dll')]
save('plan.json',dict(command=command,source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},old_binary_sha256=H(oldpath),replay_records_per_case=2000000,cases=[3,4],threads=16,budget=2,epsilon=.001,condition_trigger=1e6,timing_samples=100000,timing_repetitions=3,timing_seed_base=721001,scope='Same finite-tap approximation and branch compression; replace 64 bisections with safeguarded Newton and bisection fallback. Common random records and explicit residual diagnostics. Replay tests do not constitute a global equality proof or a new independent quality cohort.'))
start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start));assert r.returncode==0,r.stderr;print('PNR Newton build passed',flush=True)
runtime=os.add_dll_directory(str(gcc.parent));new=C.CDLL(str(OUT/'newton.dll'));old=C.CDLL(str(oldpath));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32);error=C.create_string_buffer(4096)
for lib in (new,old):lib.tc70_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];lib.tc70_create.restype=C.c_void_p;lib.tc70_destroy.argtypes=[C.c_void_p]
signature=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,C.c_double,C.c_double,DP,DP,UP,DP,C.c_char_p,C.c_int];new.pnr73_sample.argtypes=signature;old.pnr72_sample.argtypes=signature
def sample(lib,h,count,seed,mean,variance):
    output=np.zeros((count,4),np.uint32);stats=np.zeros(11 if lib is new else 7);start=time.perf_counter();rc=(lib.pnr73_sample if lib is new else lib.pnr72_sample)(h,count,seed,16,2,.001,1e6,ptr(mean),ptr(variance),output.ctypes.data_as(UP),ptr(stats),error,len(error));seconds=time.perf_counter()-start
    return output,stats,dict(returncode=rc,error=error.value.decode() if rc else '',seconds=seconds)
rows=[];timings=[];failed=[]
for case in (3,4):
    data=np.load(ROOT/f'results/iteration71_pnr_branch_pilot_v2/input_{case}.npz');v=data['v'];x=data['x'];mean=data['mean'];variance=data['variance'];hn=new.tc70_create(4,2,ptr(v),ptr(x),error,len(error));ho=old.tc70_create(4,2,ptr(v),ptr(x),error,len(error));assert hn and ho,error.value
    for batch in range(20):
        status=json.loads((ROOT/f'results/iteration71_pnr_holdout/status_{case}_branches_{batch}.json').read_text());seed=status['seed'];reference=np.load(ROOT/f'results/iteration71_pnr_holdout/counts_{case}_branches_{batch}.npz')['counts'];output,stats,result=sample(new,hn,100000,seed,mean,variance);np.savez_compressed(OUT/f'replay_{case}_{batch}.npz',counts=output,stats=stats);row=dict(case=case,batch=batch,seed=seed,**result,stats=stats.tolist())
        if result['returncode']:failed.append(row)
        else:row.update(mismatched_rows=int(np.count_nonzero(np.any(output!=reference,axis=1))),mismatched_entries=int(np.count_nonzero(output!=reference)))
        rows.append(row);save(f'replay_{case}_{batch}.json',row)
        if batch%5==0 or result['returncode']:print(row,flush=True)
        if result['returncode']:break
    for rep in range(3):
        for method in (['newton','bisection'] if rep%2==0 else ['bisection','newton']):
            lib,h=(new,hn) if method=='newton' else (old,ho);seed=721001+100*case+rep;output,stats,result=sample(lib,h,100000,seed,mean,variance);np.savez_compressed(OUT/f'timing_{case}_{rep}_{method}.npz',counts=output,stats=stats);row=dict(case=case,rep=rep,method=method,seed=seed,**result,stats=stats.tolist());timings.append(row);save(f'timing_{case}_{rep}_{method}.json',row);print(row,flush=True)
    new.tc70_destroy(hn);old.tc70_destroy(ho)
save('summary.json',dict(completed=True,all_batches_succeeded=not failed and all(r['returncode']==0 for r in timings),replay_rows=rows,timing_rows=timings,failed=failed,binary_sha256=H(OUT/'newton.dll')))
assert {p.relative_to(ROOT).as_posix():H(p) for p in sources}==json.loads((OUT/'plan.json').read_text())['source_hashes'];save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
