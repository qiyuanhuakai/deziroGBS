"""Delayed GPU compression: independent full block-law and width-one replay.
All builds and checks are fresh; frozen prior laws are independent CPU oracles.
"""
from pathlib import Path
import os,ctypes as C,json,hashlib,shutil,subprocess,time,uuid,argparse,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_block_philox_click import NativeBlockPhiloxClickSampler
from gbslab.native_philox_click import NativePhiloxClickSampler
from gbslab.native_block_click import NativeBlockClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
nvcc=Path(shutil.which('nvcc'));gcc=Path(shutil.which('g++'));cpu=ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll';sources=[Path(__file__),ROOT/'native/gaussian_click_block_philox79_v1.cu',ROOT/'native/gaussian_click_block79_v1.cu',ROOT/'src/gbslab/native_block_philox_click.py',ROOT/'src/gbslab/native_cuda_click_branches.py']
for source in sources:shutil.copy2(source,OUT/source.name)
save('plan.json',dict(scope=__doc__,source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},cpu_sha256=H(cpu),seed_base=0x9a000000000000,full_law_absolute_tolerance=1e-9,width1_probability_tolerance=0,large_rows=32))
vcvars=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat';build=Path(os.environ['LOCALAPPDATA'])/'Temp'/('deziro_block79_'+uuid.uuid4().hex);build.mkdir(exist_ok=False)
for source in sources[1:3]:shutil.copy2(source,build/source.name)
(build/'environment.bat').write_text('@call "'+str(vcvars)+'" >nul\n@set\n',encoding='ascii');values=subprocess.check_output(['cmd','/d','/c','environment.bat'],cwd=build,text=True,errors='replace');env=os.environ.copy()
for line in values.splitlines():
    if '=' in line:key,value=line.split('=',1);env[key]=value
command=[str(nvcc),'-O3','-arch=sm_120','--fmad=false','-std=c++17','-shared','-Xcompiler','/MD,/utf-8',sources[1].name,'-o','block.dll'];start=time.perf_counter();r=subprocess.run(command,cwd=build,env=env,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start,build_directory=str(build)));assert r.returncode==0,r.stderr;shutil.copy2(build/'block.dll',OUT/'block.dll');print('GPU block built',flush=True)
runtime=[gcc.parent,nvcc.parent];rows=[]
for case in (0,1):
    data=np.load(ROOT/f'results/iteration79_pilot_v1/click_input_{case}.npz');v,x,target=data['v'],data['x'],data['target'];m,n=v.shape;patterns=np.array([[(mask>>i)&1 for i in range(m)] for mask in range(1<<m)],np.uint8)
    for width,budget in [(1,1),(1,4),(1,128),(2,1),(2,4),(2,64),(4,1),(4,4),(4,16)]:
        try:
            with NativeBlockPhiloxClickSampler(v,x,cpu,gpu_library=OUT/'block.dll',block_size=width,branch_budget=budget,runtime_dirs=runtime) as model:
                forced=np.where(patterns==1,0.,np.nextafter(1.,0.));result=model.replay(np.zeros((len(patterns),2*n)),forced);assert np.array_equal(result['clicks'],patterns)
                probabilities=result['conditional_click_probabilities'];law=np.prod(np.where(patterns==1,probabilities,1-probabilities),axis=1)
                oracle=np.load(ROOT/f'results/iteration79_pilot_v1/click_{case}_{width}_{budget}.npz')['law'];delta=float(abs(law-oracle).max());tv=float(abs(law-target).sum()/2);assert delta<1e-9 and abs(law.sum()-1)<1e-9
                sample=model.sample(4096,0x9a000000000000+(case*16+width)*2**40,export_random=True);replay=model.replay(sample['realized_means'],sample['uniform_draws']);assert np.array_equal(sample['clicks'],replay['clicks']) and np.array_equal(sample['conditional_click_probabilities'],replay['conditional_click_probabilities'])
                a=model.sample(1999,0x9a000000000000+(case*16+width)*2**40);b=model.sample(2097,0x9a000000000000+(case*16+width)*2**40,row_offset=1999);assert np.array_equal(sample['clicks'],np.concatenate([a['clicks'],b['clicks']]))
                identical=None
                if width==1:
                    with NativePhiloxClickSampler(v,x,cpu,gpu_library=ROOT/'results/iteration74_fused_pilot_v2/philox.dll',branch_budget=budget,runtime_dirs=runtime) as old:previous=old.replay(sample['realized_means'],sample['uniform_draws'])
                    identical=np.array_equal(previous['clicks'],sample['clicks']) and np.array_equal(previous['conditional_click_probabilities'],sample['conditional_click_probabilities']);assert identical
            row=dict(success=True,case=case,width=width,budget=budget,numerical_tv=tv,cpu_maximum_law_error=delta,law_sum=float(law.sum()),width1_bitwise_replay=identical,statistics=sample['statistics'],cuda_timings=sample['cuda_timings']);np.savez_compressed(OUT/f'law_{case}_{width}_{budget}.npz',patterns=patterns,probabilities=probabilities,law=law,cpu_law=oracle,target=target,clicks=sample['clicks'],draws=sample['uniform_draws'],means=sample['realized_means'])
        except Exception as e:row=dict(success=False,case=case,width=width,budget=budget,error=str(e))
        save(f'law_{case}_{width}_{budget}.json',row);rows.append(row);print(row,flush=True)
# Displaced/noisy per-trajectory CPU prefix oracle; noise is held fixed, not marginalized.
displaced=[]
data=np.load(ROOT/'results/iteration71_branch_pilot_v1/input_5.npz');v,x,mean,variance=(data[k] for k in ('v','x','mean','variance'))
for width,budget in [(2,1),(2,4),(4,1),(4,4)]:
    try:
        with NativeBlockPhiloxClickSampler(v,x,cpu,gpu_library=OUT/'block.dll',block_size=width,branch_budget=budget,mean=mean,noise_variance=variance,runtime_dirs=runtime) as model:r=model.sample(64,0x9b000000000000+width*2**40,export_random=True)
        with NativeBlockClickSampler(v,x,ROOT/'results/iteration79_pilot_v1/block.dll',block_size=width,branch_budget=budget,mean=mean,noise_variance=variance,runtime_dirs=runtime) as model:oracle=np.array([model.prefix_probability(bits,mean=w)['probability'] for bits,w in zip(r['clicks'],r['realized_means'])])
        law=np.prod(np.where(r['clicks'],r['conditional_click_probabilities'],1-r['conditional_click_probabilities']),axis=1);error=float(abs(law-oracle).max());assert error<1e-9;row=dict(success=True,width=width,budget=budget,maximum_trajectory_mass_error=error);np.savez_compressed(OUT/f'displaced_{width}_{budget}.npz',law=law,oracle=oracle,**{k:r[k] for k in ('clicks','realized_means','uniform_draws','conditional_click_probabilities')})
    except Exception as e:row=dict(success=False,width=width,budget=budget,error=str(e))
    save(f'displaced_{width}_{budget}.json',row);displaced.append(row);print(row,flush=True)
large=[]
for case in (3,4):
    data=np.load(ROOT/f'results/iteration79_pilot_v1/large_input_{case}.npz')
    for width,budget in [(1,1),(1,4),(1,8),(2,1),(2,4),(4,1),(4,4)]:
        try:
            with NativeBlockPhiloxClickSampler(data['v'],data['x'],cpu,gpu_library=OUT/'block.dll',block_size=width,branch_budget=budget,runtime_dirs=runtime) as model:r=model.sample(32,0x9c000000000000+case*2**40)
            row=dict(success=True,case=case,width=width,budget=budget,seconds=r['wall_seconds'],statistics=r['statistics'],cuda_timings=r['cuda_timings']);np.savez_compressed(OUT/f'large_{case}_{width}_{budget}.npz',clicks=r['clicks'],probabilities=r['conditional_click_probabilities'],diagnostics=r['trajectory_diagnostics'])
        except Exception as e:row=dict(success=False,case=case,width=width,budget=budget,error=str(e))
        save(f'large_{case}_{width}_{budget}.json',row);large.append(row);print(row,flush=True)
save('summary.json',dict(completed=True,laws=rows,displaced=displaced,large=large));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
