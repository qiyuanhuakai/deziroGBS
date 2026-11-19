"""Build and validate FP64 GPU branch trajectories against shared CPU draws."""
from pathlib import Path
import os,ctypes as C,json,hashlib,shutil,subprocess,time,uuid,argparse,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_click_branches import NativeBranchClickSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_double))
gcc=Path(shutil.which('g++'));nvcc=Path(shutil.which('nvcc'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'
sources=[Path(__file__)]+[ROOT/'native'/name for name in ('gaussian_click_replay72.cpp','gaussian_click_cuda72_v1.cu','gaussian_click_branches_v1.cpp','gaussian_direct_click.cpp','gaussian_reset_cpu.cpp')]
for p in sources:shutil.copy2(p,OUT/p.name)
save('plan.json',dict(source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},arithmetic='FP64 GPU, FP64 CPU covariance with extended-precision probability accumulation; no clipping, no fast math, no FMA contraction',scope='Identical explicit GCC random inputs, common bounded-branch real-arithmetic model, different floating-point execution. Full small induced-law checks and record/probability replay. Large input has no full-TV certificate.',small_cases=list(range(6)),large_samples=512,large_budgets=[1,4,8],large_repetitions=3,large_seed_base=715901))
command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(sources[1]),'-o',str(OUT/'cpu.dll')];start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('cpu_build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start));assert r.returncode==0,r.stderr;print('CPU replay built',flush=True)
vcvars=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat';assert vcvars.is_file();build=Path(os.environ['LOCALAPPDATA'])/'Temp'/('deziro_gbs_cuda72_'+uuid.uuid4().hex);build.mkdir(exist_ok=False);shutil.copy2(sources[2],build/'kernel.cu');(build/'environment.bat').write_text('@call "'+str(vcvars)+'" >nul\n@set\n',encoding='ascii');values=subprocess.check_output(['cmd','/d','/c','environment.bat'],cwd=build,text=True,errors='replace');env=os.environ.copy()
for line in values.splitlines():
    if '=' in line:key,value=line.split('=',1);env[key]=value
command=[str(nvcc),'-O3','-arch=sm_120','--fmad=false','-std=c++17','-shared','-Xcompiler','/MD,/utf-8','kernel.cu','-o','gpu.dll'];start=time.perf_counter();r=subprocess.run(command,cwd=build,env=env,capture_output=True,text=True,errors='replace');save('cuda_build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start,build_directory=str(build),nvcc_sha256=H(nvcc)));assert r.returncode==0,r.stderr;shutil.copy2(build/'gpu.dll',OUT/'gpu.dll');print('CUDA branch kernel built',flush=True)
runtimes=[os.add_dll_directory(str(p)) for p in [gcc.parent,nvcc.parent,nvcc.parent/'x64'] if p.is_dir()];cpu=C.CDLL(str(OUT/'cpu.dll'));gpu=C.CDLL(str(OUT/'gpu.dll'));DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);IP=C.POINTER(C.c_int32);error=C.create_string_buffer(4096)
cpu.tc70_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];cpu.tc70_create.restype=C.c_void_p;cpu.tc70_destroy.argtypes=[C.c_void_p]
cpu.tc72_draws.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint64,DP,DP,DP,DP,C.c_char_p,C.c_int]
cpu.tc72_cpu.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,DP,DP,BP,DP,C.c_char_p,C.c_int]
gpu.gc72_sample.argtypes=[C.c_int,C.c_int,C.c_int,C.c_int,DP,DP,DP,DP,DP,BP,DP,IP,DP,DP,C.c_char_p,C.c_int]
def run_gpu(v,x,means,draws,budget):
    m,n=v.shape;count=len(means);a=np.ascontiguousarray(np.c_[v.real,-v.imag]);b=np.ascontiguousarray(np.c_[v.imag,v.real]);output=np.zeros((count,m),np.uint8);prob=np.zeros((count,m));errors=np.zeros((count,2),np.int32);diag=np.zeros((count,3));timings=np.zeros(6);start=time.perf_counter();rc=gpu.gc72_sample(m,2*n,count,budget,ptr(x),ptr(a),ptr(b),ptr(means),ptr(draws),output.ctypes.data_as(BP),ptr(prob),errors.ctypes.data_as(IP),ptr(diag),ptr(timings),error,len(error));seconds=time.perf_counter()-start
    return dict(output=output,probabilities=prob,errors=errors,diagnostics=diag,timings=timings),dict(returncode=rc,error=error.value.decode() if rc else '',seconds=seconds)
def run_cpu(h,means,draws,budget):
    count,m=draws.shape;output=np.zeros((count,m),np.uint8);prob=np.zeros((count,m));start=time.perf_counter();rc=cpu.tc72_cpu(h,count,16,budget,ptr(means),ptr(draws),output.ctypes.data_as(BP),ptr(prob),error,len(error));seconds=time.perf_counter()-start
    return dict(output=output,probabilities=prob),dict(returncode=rc,error=error.value.decode() if rc else '',seconds=seconds)
small=[];failures=[]
for case in range(6):
    data=np.load(ROOT/f'results/iteration71_branch_pilot_v1/input_{case}.npz');v=data['v'];x=data['x'];m,n=v.shape;means=np.tile(data['mean'],(2**m,1));patterns=np.array([[(mask>>i)&1 for i in range(m)] for mask in range(2**m)],np.uint8);draws=np.where(patterns==1,0.,np.nextafter(1.,0.));h=cpu.tc70_create(m,n,ptr(v),ptr(x),error,len(error));assert h,error.value
    for budget in ([1,2,4,8] if m==4 else [1,2,4,8,16,32,64,128]):
        g,gs=run_gpu(v,x,means,draws,budget);c,cs=run_cpu(h,means,draws,budget);label=f'{case}_{budget}';np.savez_compressed(OUT/f'small_gpu_{label}.npz',**g);np.savez_compressed(OUT/f'small_cpu_{label}.npz',**c);row=dict(case=case,budget=budget,gpu=gs,cpu=cs)
        if gs['returncode'] or cs['returncode']:failures.append(row)
        else:
            law=np.prod(np.where(patterns==1,g['probabilities'],1-g['probabilities']),axis=1);reference=np.load(ROOT/f'results/iteration71_branch_pilot_v1/law_{case}_{budget}.npz')['law'];row.update(law_tv_to_cpu=float(abs(law-reference).sum()/2),law_tv_to_target=float(abs(law-data['target']).sum()/2),probability_max_error=float(abs(g['probabilities']-c['probabilities']).max()),wrong_bits=int(np.count_nonzero(g['output']!=patterns)),mass_error=float(abs(law.sum()-1)));np.savez_compressed(OUT/f'law_{label}.npz',law=law,target=data['target'],reference=reference)
            if row['wrong_bits'] or row['law_tv_to_cpu']>1e-8 or row['mass_error']>1e-8:failures.append(row)
        small.append(row);save(f'small_{label}.json',row);print(row,flush=True)
    cpu.tc70_destroy(h)
save('small_summary.json',dict(passed=not failures,rows=small,failures=failures))
assert not failures,'GPU small-law validation failed; see preserved records'
large=np.load(ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110/input.npz');v=large['v'];x=large['x0'];mean=large['mean'];variance=large['variance'];m,n=v.shape;h=cpu.tc70_create(m,n,ptr(v),ptr(x),error,len(error));assert h,error.value;rows=[]
for rep in range(3):
    means=np.zeros((512,2*n));draws=np.zeros((512,m));start=time.perf_counter();rc=cpu.tc72_draws(m,2*n,512,715901+rep,ptr(mean),ptr(variance),ptr(means),ptr(draws),error,len(error));draw_seconds=time.perf_counter()-start;assert rc==0,error.value;np.savez_compressed(OUT/f'inputs_large_{rep}.npz',means=means,draws=draws)
    for budget in ([1,4,8][rep:]+[1,4,8][:rep]):
        if rep%2:g,gs=run_gpu(v,x,means,draws,budget);c,cs=run_cpu(h,means,draws,budget)
        else:c,cs=run_cpu(h,means,draws,budget);g,gs=run_gpu(v,x,means,draws,budget)
        label=f'{rep}_{budget}';np.savez_compressed(OUT/f'large_gpu_{label}.npz',**g);np.savez_compressed(OUT/f'large_cpu_{label}.npz',**c);row=dict(rep=rep,budget=budget,gpu=gs,cpu=cs,draw_seconds=draw_seconds)
        if gs['returncode']==cs['returncode']==0:
            old=np.load(ROOT/f'results/iteration71_frugal_comparison/large_{rep}_branch_{budget}.npz')['clicks'];row.update(gpu_cpu_mismatched_bits=int(np.count_nonzero(g['output']!=c['output'])),cpu_archived_mismatched_bits=int(np.count_nonzero(c['output']!=old)),probability_max_error=float(abs(g['probabilities']-c['probabilities']).max()))
        rows.append(row);save(f'large_{label}.json',row);print(row,flush=True)
cpu.tc70_destroy(h);passed=all(r['gpu']['returncode']==r['cpu']['returncode']==0 and r['gpu_cpu_mismatched_bits']==r['cpu_archived_mismatched_bits']==0 for r in rows)
save('summary.json',dict(passed=passed,small=small,large=rows,binaries={p.name:H(p) for p in OUT.glob('*.dll')}));assert {p.relative_to(ROOT).as_posix():H(p) for p in sources}==json.loads((OUT/'plan.json').read_text())['source_hashes'];save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
