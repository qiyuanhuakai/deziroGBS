"""Build joint photon blocks and local Gaussian click blocks; full-law checks.
Independent photon-block convolution is an approximation to coherent pure GBS.
"""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,subprocess,time,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_joint_number_pnr import NativeJointNumberPNRSampler
from gbslab.native_block_click import NativeBlockClickSampler
from gbslab.native_click_branches import NativeBranchClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));runtime=[gcc.parent];eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';paths=[Path(__file__),ROOT/'src/gbslab/native_joint_number_pnr.py',ROOT/'src/gbslab/native_block_click.py',*sorted((ROOT/'native').glob('*.cpp')),*sorted((ROOT/'native').glob('*.hpp'))];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in paths},seed_base=0x96000000000000,workers=16));shutil.copy2(__file__,OUT/Path(__file__).name)
for kind,source in [('joint','gaussian_pnr_joint79_v1.cpp'),('block','gaussian_block_click79_v1.cpp')]:
    command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(ROOT/'native'/source),'-o',str(OUT/f'{kind}.dll')];start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save(f'build_{kind}.json',dict(command=command,returncode=r.returncode,seconds=time.perf_counter()-start,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr;print('built',kind,flush=True)
d=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');v,t=d['v'],d['t'];base=int(d['cutoff'])+1;patterns=np.array([(d['codes']//base**j)%base for j in range(4)]).T.astype(np.int32);totals=patterns.sum(axis=1)
def pure(N):
    mask=totals==N;q=d['target'][:-1][mask];q=q/q.sum();return {tuple(x):float(p) for x,p in zip(patterns[mask],q)}
def blocklaw(N,b):
    law={(0,0,0,0):1.};left=N
    while left:
        k=min(left,b);factor=pure(k);out={}
        for x,p in law.items():
            for y,q in factor.items():key=tuple(a+b for a,b in zip(x,y));out[key]=out.get(key,0.)+p*q
        law=out;left-=k
    return law
joint_rows=[]
for case,(N,b) in enumerate([(2,2),(4,4),(6,6),(8,2),(8,4),(14,2),(14,4)]):
    law=blocklaw(N,b);states=np.array(sorted(law),np.int32);target=np.array([law[tuple(s)] for s in states]);pure_target=pure(N);ground=np.array([pure_target[tuple(s)] for s in states]);lookup={tuple(s):i for i,s in enumerate(states)}
    with NativeJointNumberPNRSampler(v,t,OUT/'joint.dll',exact_total_budget=0,joint_block_size=b,workers=16,runtime_dirs=runtime) as model:r=model.sample(100000,0x96000000000000+case*2**40,fixed_total=N)
    hist=np.bincount([lookup[tuple(x)] for x in r['counts']],minlength=len(states));empirical=float(abs(hist/len(r['counts'])-target).sum()/2);assert empirical<.04 and abs(target.sum()-1)<1e-11;row=dict(case=case,total=N,block=b,records=100000,empirical_tv_to_block_law=empirical,numerical_block_law_tv_to_pure=float(abs(target-ground).sum()/2),seconds=r['wall_seconds'],statistics=r['statistics']);joint_rows.append(row);save(f'joint_{case}.json',row);np.savez_compressed(OUT/f'joint_{case}.npz',states=states,target=target,pure_target=ground,counts=r['counts'],histogram=hist);print('joint law',N,b,row['numerical_block_law_tv_to_pure'],flush=True)
with NativeJointNumberPNRSampler(v,t,OUT/'joint.dll',exact_total_budget=64,joint_block_size=2,workers=16,runtime_dirs=runtime) as model:r=model.sample(10000,0x88000000000000+2**40)
old=np.load(ROOT/'results/iteration77_number_pilot_v2/counts_1.npz');assert np.array_equal(r['counts'],old['counts']);np.savez_compressed(OUT/'strict_replay.npz',counts=r['counts'],totals=r['total_photons'])
click_rows=[]
for case in (0,1):
    d=np.load(ROOT/f'results/iteration78_pilot_v2/table_{case+2}.npz');v,x,target=d['v'],d['x'],d['target'];m,n=v.shape;patterns=np.array([[(mask>>i)&1 for i in range(m)] for mask in range(1<<m)],np.uint8);np.savez_compressed(OUT/f'click_input_{case}.npz',v=v,x=x,target=target)
    for width,budget in [(1,1),(1,4),(1,128),(2,1),(2,4),(2,64),(4,1),(4,4),(4,16)]:
        start=time.perf_counter()
        try:
            with NativeBlockClickSampler(v,x,OUT/'block.dll',block_size=width,branch_budget=budget,workers=16,runtime_dirs=runtime) as model:
                law=np.array([model.prefix_probability(bits)['probability'] for bits in patterns]);r=model.sample(4096,0x97000000000000+(case*16+width)*2**40)
            tv=float(abs(law-target).sum()/2);assert abs(law.sum()-1)<1e-9;assert tv<1e-9 if budget>=2**(m-width) else True;row=dict(success=True,case=case,outputs=m,width=width,budget=budget,numerical_full_tv=tv,law_sum=float(law.sum()),statistics=r['statistics'],seconds=time.perf_counter()-start);np.savez_compressed(OUT/f'click_{case}_{width}_{budget}.npz',law=law,target=target,clicks=r['clicks'])
            if width==1:
                with NativeBranchClickSampler(v,x,ROOT/'results/iteration71_branch_pilot_v1/branches.dll',branch_budget=budget,workers=16,runtime_dirs=runtime) as previous:old=previous.sample(4096,0x97000000000000+(case*16+width)*2**40)
                assert np.array_equal(r['clicks'],old['clicks']);row['width1_old_records_identical']=True
        except Exception as e:row=dict(success=False,case=case,outputs=m,width=width,budget=budget,error=str(e),seconds=time.perf_counter()-start)
        click_rows.append(row);save(f'click_{case}_{width}_{budget}.json',row);print(row,flush=True)
large=[]
for case in (3,4):
    d=np.load(ROOT/f'results/iteration77_budget_pilot_v1/input_{case}.npz');v,t=d['v'],float(d['t']);m,n=v.shape;x=np.diag(np.r_[np.full(n,-t/(1+t)),np.full(n,t/(1-t))]);np.savez_compressed(OUT/f'large_input_{case}.npz',v=v,t=t,x=x)
    for b in (2,4,8):
        start=time.perf_counter()
        try:
            with NativeJointNumberPNRSampler(v,t,OUT/'joint.dll',exact_total_budget=12,joint_block_size=b,workers=16,runtime_dirs=runtime) as model:r=model.sample(256,0x98000000000000+case*2**40)
            row=dict(success=True,kind='joint',case=case,block=b,records=256,seconds=time.perf_counter()-start,statistics=r['statistics']);np.savez_compressed(OUT/f'large_joint_{case}_{b}.npz',counts=r['counts'],totals=r['total_photons'])
        except Exception as e:row=dict(success=False,kind='joint',case=case,block=b,seconds=time.perf_counter()-start,error=str(e))
        large.append(row);save(f'large_joint_{case}_{b}.json',row);print(row,flush=True)
    for width,budget in ((2,1),(4,1),(2,4),(4,4)):
        start=time.perf_counter()
        try:
            with NativeBlockClickSampler(v,x,OUT/'block.dll',block_size=width,branch_budget=budget,workers=16,runtime_dirs=runtime) as model:r=model.sample(16,0x99000000000000+case*2**40)
            row=dict(success=True,kind='block',case=case,width=width,budget=budget,records=16,seconds=time.perf_counter()-start,statistics=r['statistics']);np.savez_compressed(OUT/f'large_click_{case}_{width}_{budget}.npz',clicks=r['clicks'])
        except Exception as e:row=dict(success=False,kind='block',case=case,width=width,budget=budget,seconds=time.perf_counter()-start,error=str(e))
        large.append(row);save(f'large_click_{case}_{width}_{budget}.json',row);print(row,flush=True)
save('summary.json',dict(completed=True,joint=joint_rows,click=click_rows,large=large));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
