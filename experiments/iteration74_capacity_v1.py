"""Verify doubled large-batch capacity with unchanged records or explicit rejection."""
from pathlib import Path
import os,sys,json,shutil,hashlib,time,argparse
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_philox_click import NativePhiloxClickSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
runtime=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent];cpu=ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll';libs={name:ROOT/f'results/{folder}/philox.dll' for name,folder in [('old','iteration73_philox_pilot_v1'),('fused','iteration74_fused_pilot_v1'),('reduced','iteration74_fused_pilot_v2')]};data=np.load(ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110/input.npz');seed=744501
save('plan.json',dict(driver_sha256=H(Path(__file__)),binary_hashes={k:H(p) for k,p in libs.items()},seed=seed,count=16384,budget=8,repetitions=3,scope=__doc__))
rows=[];reference=[]
for rep in range(3):
    for name in (('old','fused','reduced') if rep%2==0 else ('reduced','fused','old')):
        row=dict(rep=rep,kind=name);save(f'started_{rep}_{name}.json',row);start=time.perf_counter()
        try:
            with NativePhiloxClickSampler(data['v'],data['x0'],cpu,gpu_library=libs[name],mean=data['mean'],noise_variance=data['variance'],branch_budget=8,runtime_dirs=runtime) as model:r=model.sample(16384,seed+rep)
            row.update(success=True,wall_seconds=time.perf_counter()-start,cuda_timings=r['cuda_timings']);np.savez_compressed(OUT/f'records_{rep}_{name}.npz',clicks=r['clicks'],probabilities=r['conditional_click_probabilities'])
        except Exception as e:row.update(success=False,wall_seconds=time.perf_counter()-start,error=str(e))
        rows.append(row);save(f'row_{rep}_{name}.json',row);print(row,flush=True)
    with NativePhiloxClickSampler(data['v'],data['x0'],cpu,gpu_library=libs['old'],mean=data['mean'],noise_variance=data['variance'],branch_budget=8,runtime_dirs=runtime) as model:
        old=[model.sample(4096,seed+rep,row_offset=4096*j) for j in range(4)]
    bits=np.concatenate([r['clicks'] for r in old]);probs=np.concatenate([r['conditional_click_probabilities'] for r in old]);np.savez_compressed(OUT/f'old_split_{rep}.npz',clicks=bits,probabilities=probs)
    for name in ('fused','reduced'):
        saved=np.load(OUT/f'records_{rep}_{name}.npz');row=dict(rep=rep,kind=name,mismatched_bits=int(np.count_nonzero(saved['clicks']!=bits)),maximum_probability_difference=float(abs(saved['probabilities']-probs).max()));reference.append(row);save(f'replay_{rep}_{name}.json',row);assert row['mismatched_bits']==0 and row['maximum_probability_difference']<1e-8,row
save('summary.json',dict(passed=True,rows=rows,replays=reference));shutil.copy2(__file__,OUT/Path(__file__).name);save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
