"""Matched generic CPU comparisons; exact small laws and retained large records."""
from pathlib import Path
import os,hashlib,json,shutil,subprocess,time,argparse,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_click_branches import NativeBranchClickSampler
from gbslab.frugal import frugal_sample
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
small_exe=ROOT/'results/iteration70_archive/iteration70_frugal_speed_v2_20261110/frugal.exe';large=ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110';dll=ROOT/'results/iteration71_branch_pilot_v1/branches.dll';runtime=[Path(shutil.which('g++')).parent]
save('plan.json',dict(source_sha256=H(Path(__file__)),wrapper_sha256=H(ROOT/'src/gbslab/native_click_branches.py'),dll_sha256=H(dll),small_frugal_sha256=H(small_exe),large_frugal_sha256=H(large/'frugal.exe'),small_cases=[3,4],small_samples={3:200000,4:100000},large_samples=512,repetitions=3,large_workers=16,small_workers=1,scope='Generic x86-64 CPU builds, identical physical inputs per panel, preparation and sampling both included. Four-mode B8 and K4 both numerically reproduce exact target; eight-mode Frugal truncations and branch budgets differ in error. Large full TV is not established. Native instruction-set Frugal failure in iteration70 means these are not strongest CPU/GPU baselines.'))
shutil.copy2(__file__,OUT/Path(__file__).name);shutil.copy2(ROOT/'src/gbslab/native_click_branches.py',OUT/'native_click_branches.py');rows=[]
for case in (3,4):
    data=np.load(ROOT/f'results/iteration71_branch_pilot_v1/input_{case}.npz');v=data['v'];x=data['x'];mean=data['mean'];target=data['target'];m,n=v.shape;assert not mean.any();o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.eye(2*m)+2*o@x@o.T;count=200000 if case==3 else 100000
    methods=[('branch',b) for b in ([1,4,8] if case==3 else [8,32,128])]+[('frugal',k) for k in ([2,3,4] if case==3 else [3,4,5])]
    for rep in range(3):
        for kind,budget in methods[rep:]+methods[:rep]:
            seed=715001+100*case+rep;label=f'{kind}_{budget}';row=dict(panel=f'small_{case}',case=case,rep=rep,method=label,samples=count,seed=seed,workers=1)
            if kind=='branch':
                start=time.perf_counter();model=NativeBranchClickSampler(v,x,dll,branch_budget=budget,runtime_dirs=runtime);setup=time.perf_counter()-start;result=model.sample(count,seed);model.close();clicks=result['clicks'];law=np.load(ROOT/f'results/iteration71_branch_pilot_v1/law_{case}_{budget}.npz')['law'];np.savez_compressed(OUT/f'clicks_{case}_{rep}_{label}.npz',clicks=clicks);row.update(setup_seconds=setup,sampling_seconds=result['wall_seconds'],compute_seconds=setup+result['wall_seconds'],induced_tv=float(abs(law-target).sum()/2))
            else:
                result=frugal_sample(cov,budget,count,seed,executable=small_exe,threads=1);save(f'frugal_{case}_{rep}_{budget}.json',result);t=result['timings_seconds'];setup=t['layout']+t['target'];row.update(setup_seconds=setup,sampling_seconds=t['sampling'],compute_seconds=setup+t['sampling'],induced_tv=float(abs(np.asarray(result['distribution'])-target).sum()/2),wrapper_wall_seconds=result['wrapper_wall_seconds'])
            rows.append(row);save(f'row_{case}_{rep}_{label}.json',row);print(row,flush=True)
data=np.load(large/'input.npz');v=data['v'];x=data['x0'];mean=data['mean'];variance=data['variance'];methods=[('branch',b) for b in [1,4,8]]+[('frugal',2),('frugal',3)]
for rep in range(3):
    for kind,budget in methods[rep:]+methods[:rep]:
        seed=715901+rep;label=f'{kind}_{budget}';row=dict(panel='large_144',rep=rep,method=label,samples=512,seed=seed,workers=16)
        if kind=='branch':
            start=time.perf_counter();model=NativeBranchClickSampler(v,x,dll,mean=mean,noise_variance=variance,workers=16,branch_budget=budget,runtime_dirs=runtime);setup=time.perf_counter()-start;result=model.sample(512,seed);model.close();np.savez_compressed(OUT/f'large_{rep}_{label}.npz',clicks=result['clicks']);row.update(setup_seconds=setup,sampling_seconds=result['wall_seconds'],compute_seconds=setup+result['wall_seconds'],statistics=result['statistics'])
        else:
            command=[str(large/'frugal.exe'),'--input',str(large/'covariance.csv'),'--modes','144','--cutoff',str(budget),'--samples','512','--seed',str(seed),'--threads','16'];start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');wall=time.perf_counter()-start;save(f'process_{rep}_{budget}.json',dict(command=command,returncode=r.returncode,stderr=r.stderr,wall_seconds=wall));assert r.returncode==0,r.stderr;result=json.loads(r.stdout);save(f'large_{rep}_{label}.json',result);t=result['timings_seconds'];setup=t['layout']+t['cumulants']+t['target'];row.update(setup_seconds=setup,sampling_seconds=t['official_sampling'],compute_seconds=setup+t['official_sampling'],wrapper_wall_seconds=wall)
        rows.append(row);save(f'row_large_{rep}_{label}.json',row);print(row,flush=True)
save('summary.json',dict(passed=True,rows=rows))
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
