"""CPU guarded rolling compression: complete small laws and large failures.
Fixed branch capacities are compared with the old all-to-one rule.
"""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,subprocess,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_guarded_rolling_click import NativeGuardedRollingClickSampler
from gbslab.native_click_branches import NativeBranchClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));runtime=[gcc.parent];eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';sources=[Path(__file__),ROOT/'src/gbslab/native_guarded_rolling_click.py',*sorted((ROOT/'native').glob('*.cpp')),*sorted((ROOT/'native').glob('*.hpp'))];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},workers=16,seed_base=0xa0000000000000));shutil.copy2(__file__,OUT/Path(__file__).name)
command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(ROOT/'native/gaussian_click_guarded80_v1.cpp'),'-o',str(OUT/'rolling.dll')];start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,seconds=time.perf_counter()-start,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr;print('rolling built',flush=True)
rows=[]
for case in (0,1):
    data=np.load(ROOT/f'results/iteration79_pilot_v1/click_input_{case}.npz');v,x,target=data['v'],data['x'],data['target'];m,n=v.shape;patterns=np.array([[(s>>j)&1 for j in range(m)] for s in range(1<<m)],np.uint8)
    for B in (1,2,4,8,16,32,128):
        for kind,cls,binary in [('rolling',NativeGuardedRollingClickSampler,OUT/'rolling.dll'),('old',NativeBranchClickSampler,ROOT/'results/iteration71_branch_pilot_v1/branches.dll')]:
            try:
                with cls(v,x,binary,branch_budget=B,workers=16,runtime_dirs=runtime) as model:
                    law=np.array([model.prefix_probability(bits)['probability'] for bits in patterns]);r=model.sample(8192,0xa0000000000000+case*2**40)
                assert abs(law.sum()-1)<1e-9 and np.all(law>=0);tv=float(abs(law-target).sum()/2);row=dict(success=True,case=case,kind=kind,budget=B,numerical_full_tv=tv,seconds=r['wall_seconds'],statistics=r['statistics']);np.savez_compressed(OUT/f'law_{case}_{kind}_{B}.npz',law=law,target=target,clicks=r['clicks'])
            except Exception as e:row=dict(success=False,case=case,kind=kind,budget=B,error=str(e))
            rows.append(row);save(f'law_{case}_{kind}_{B}.json',row);print(row,flush=True)
large=[]
for case in (3,4):
    data=np.load(ROOT/f'results/iteration79_pilot_v1/large_input_{case}.npz')
    for B in (2,4,8):
        try:
            with NativeGuardedRollingClickSampler(data['v'],data['x'],OUT/'rolling.dll',branch_budget=B,workers=16,runtime_dirs=runtime) as model:r=model.sample(16,0xa1000000000000+case*2**40)
            row=dict(success=True,case=case,budget=B,seconds=r['wall_seconds'],statistics=r['statistics']);np.savez_compressed(OUT/f'large_{case}_{B}.npz',clicks=r['clicks'])
        except Exception as e:row=dict(success=False,case=case,budget=B,error=str(e))
        large.append(row);save(f'large_{case}_{B}.json',row);print(row,flush=True)
save('summary.json',dict(completed=True,laws=rows,large=large));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
