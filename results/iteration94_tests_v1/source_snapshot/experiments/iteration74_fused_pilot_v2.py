"""Build fused-overflow warp-reduction CUDA Philox; thermal and CPU trajectory replay."""
from pathlib import Path
import os,ctypes as C,json,hashlib,shutil,subprocess,time,uuid,argparse,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_philox_click import NativePhiloxClickSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def ptr(a):return a.ctypes.data_as(C.POINTER(C.c_double))
nvcc=Path(shutil.which('nvcc'));gcc=Path(shutil.which('g++'));cpu=ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll';sources=[Path(__file__),ROOT/'native/gaussian_click_philox74_v2.cu',ROOT/'native/gaussian_click_fused74_v2.cu',ROOT/'src/gbslab/native_philox_click.py',ROOT/'src/gbslab/native_cuda_click_branches.py']
for p in sources:shutil.copy2(p,OUT/p.name)
save('plan.json',dict(source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},cpu_library_sha256=H(cpu),scope=__doc__,rng='Philox4x32_10; curand_normal2_double and 1-curand_uniform2_double; subsequence is row_offset+row; same approximation, different seed mapping from iteration72.',thermal_samples=200000,thermal_variance=.4,thermal_tv_limit=.005,replay_rows=4096,replay_cases=[3,4,5],replay_budgets=[1,4,8],seeds=[733001,733003,733004,733005]))
vcvars=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat';build=Path(os.environ['LOCALAPPDATA'])/'Temp'/('deziro_philox74_'+uuid.uuid4().hex);build.mkdir(exist_ok=False)
for p in sources[1:3]:shutil.copy2(p,build/p.name)
(build/'environment.bat').write_text('@call "'+str(vcvars)+'" >nul\n@set\n',encoding='ascii');values=subprocess.check_output(['cmd','/d','/c','environment.bat'],cwd=build,text=True,errors='replace');env=os.environ.copy()
for line in values.splitlines():
    if '=' in line:key,value=line.split('=',1);env[key]=value
command=[str(nvcc),'-O3','-arch=sm_120','--fmad=false','-std=c++17','-shared','-Xcompiler','/MD,/utf-8',sources[1].name,'-o','philox.dll'];start=time.perf_counter();r=subprocess.run(command,cwd=build,env=env,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start,build_directory=str(build)));assert r.returncode==0,r.stderr;shutil.copy2(build/'philox.dll',OUT/'philox.dll');print('Philox CUDA built',flush=True)
runtime=[gcc.parent,nvcc.parent];rows=[]
with NativePhiloxClickSampler(np.eye(2,dtype=complex),np.zeros((4,4)),cpu,gpu_library=OUT/'philox.dll',noise_variance=np.full(4,.4),runtime_dirs=runtime) as model:
    result=model.sample(200000,733001,export_random=True);means=result['realized_means'];draws=result['uniform_draws'];codes=(result['clicks']@np.array([1,2])).astype(np.int64);p=.4/1.4;target=np.array([(1-p)**2,p*(1-p),p*(1-p),p*p]);hist=np.bincount(codes,minlength=4)/len(codes)
    check=dict(thermal_tv=float(abs(hist-target).sum()/2),normal_means=means.mean(axis=0).tolist(),normal_variances=means.var(axis=0).tolist(),uniform_means=draws.mean(axis=0).tolist(),uniform_variances=draws.var(axis=0).tolist(),min_uniform=float(draws.min()),max_uniform=float(draws.max()),non_32bit_grid=int(np.count_nonzero(np.modf(draws*2**32)[0])),seconds=result['wall_seconds'])
    np.savez_compressed(OUT/'thermal.npz',means=means,draws=draws,clicks=result['clicks'],target=target);save('thermal.json',check)
    assert check['thermal_tv']<.005 and np.max(abs(means.mean(axis=0)))<.006 and np.max(abs(means.var(axis=0)-.4))<.006 and np.max(abs(draws.mean(axis=0)-.5))<.004 and np.max(abs(draws.var(axis=0)-1/12))<.002 and draws.min()>=0 and draws.max()<1 and check['non_32bit_grid']>300000,check
    whole=model.sample(4096,733002,row_offset=17,export_random=True);a=model.sample(1999,733002,row_offset=17,export_random=True);b=model.sample(2097,733002,row_offset=2016,export_random=True)
    for key in ('clicks','realized_means','uniform_draws','conditional_click_probabilities'):assert np.array_equal(whole[key],np.concatenate([a[key],b[key]])),key
    save('batch_invariance.json',dict(passed=True,seed=733002,start=17,rows=4096,split=1999));np.savez_compressed(OUT/'batch_invariance.npz',**{k:whole[k] for k in ('clicks','realized_means','uniform_draws')})
for case in (3,4,5):
    data=np.load(ROOT/f'results/iteration71_branch_pilot_v1/input_{case}.npz')
    with NativePhiloxClickSampler(data['v'],data['x'],cpu,gpu_library=OUT/'philox.dll',mean=data['mean'],noise_variance=data['variance'],runtime_dirs=runtime) as model:
        model.lib.tc72_cpu.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_double),C.POINTER(C.c_uint8),C.POINTER(C.c_double),C.c_char_p,C.c_int]
        for budget in (1,4,8):
            result=model.sample(4096,733000+case,branch_budget=budget,export_random=True);means=result['realized_means'];draws=result['uniform_draws'];output=np.zeros_like(result['clicks']);prob=np.zeros_like(draws);error=C.create_string_buffer(4096);rc=model.lib.tc72_cpu(model.handle,4096,16,budget,ptr(means),ptr(draws),output.ctypes.data_as(C.POINTER(C.c_uint8)),ptr(prob),error,len(error));row=dict(case=case,budget=budget,returncode=rc,error=error.value.decode() if rc else '',mismatched_bits=int(np.count_nonzero(output!=result['clicks'])),maximum_conditional_error=float(abs(prob-result['conditional_click_probabilities']).max()),wall_seconds=result['wall_seconds']);save(f'replay_{case}_{budget}.json',row);np.savez_compressed(OUT/f'replay_{case}_{budget}.npz',means=means,draws=draws,cpu=output,gpu=result['clicks'],cpu_probabilities=prob,gpu_probabilities=result['conditional_click_probabilities']);rows.append(row);print(row,flush=True);assert rc==0 and row['mismatched_bits']==0 and row['maximum_conditional_error']<1e-8,row
save('summary.json',dict(passed=True,thermal=check,replays=rows,binary_sha256=H(OUT/'philox.dll')))
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
