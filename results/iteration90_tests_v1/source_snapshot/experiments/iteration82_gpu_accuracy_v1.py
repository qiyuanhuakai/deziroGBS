"""Actual GPU forced-prefix law on the identical timed synthetic inputs.
Never assign CPU quality to a GPU configuration without replay verification.
"""
from pathlib import Path
import os,sys,json,hashlib,shutil,argparse
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_philox_click import NativePhiloxClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
 with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
dirs=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent];cpu=ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll';gpu=ROOT/'results/iteration74_fused_pilot_v2/philox.dll';save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),cpu,gpu]},labels=['wide2','large128'],budgets=[4,8]));rows=[]
for label in ('wide2','large128'):
 d=np.load(ROOT/f'results/iteration82_factor_accuracy_v1/input_{label}.npz');v,x,patterns,target=d['v'],d['x'],d['patterns'],d['target'];depth=patterns.shape[1];draws=np.full((len(patterns),len(v)),.5);draws[:,:depth]=np.where(patterns,0.,np.nextafter(1.,0.))
 for budget in (4,8):
  try:
   with NativePhiloxClickSampler(v,x,cpu,gpu_library=gpu,branch_budget=budget,runtime_dirs=dirs) as model:r=model.replay(np.zeros((len(patterns),2*v.shape[1])),draws)
   assert np.array_equal(r['clicks'][:,:depth],patterns);q=r['conditional_click_probabilities'][:,:depth];law=np.prod(np.where(patterns,q,1-q),axis=1);claw=np.load(ROOT/f'results/iteration82_factor_accuracy_v1/direct_{label}_{budget}.npz')['law'];difference=float(abs(law-claw).max());assert difference<1e-9;row=dict(label=label,budget=budget,passed=True,numerical_tv=float(abs(law-target).sum()/2),maximum_cpu_difference=difference,normalization_error=float(abs(law.sum()-1)));np.savez_compressed(OUT/f'law_{label}_{budget}.npz',law=law,conditional=q,clicks=r['clicks'])
  except (ArithmeticError,AssertionError) as e:row=dict(label=label,budget=budget,passed=False,error=str(e))
  rows.append(row);save(f'row_{label}_{budget}.json',row);print(row,flush=True)
save('summary.json',dict(completed=True,rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
