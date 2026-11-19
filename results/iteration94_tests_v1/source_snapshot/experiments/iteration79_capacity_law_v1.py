"""Complete small click laws for equal allocated capacity timing controls.
This supplements the frozen pilot without changing or selecting timing panels.
"""
from pathlib import Path
import os,sys,shutil,json,hashlib,argparse
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_block_click import NativeBlockClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
binary=ROOT/'results/iteration79_pilot_v1/block.dll';save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),binary,ROOT/'src/gbslab/native_block_click.py']}));d=np.load(ROOT/'results/iteration79_pilot_v1/click_input_1.npz');rows=[]
for B in (4,8,32,128):
    with NativeBlockClickSampler(d['v'],d['x'],binary,block_size=1,branch_budget=B,runtime_dirs=[Path(shutil.which('g++')).parent]) as model:law=np.array([model.prefix_probability([(mask>>i)&1 for i in range(8)])['probability'] for mask in range(256)])
    assert abs(law.sum()-1)<1e-12 and np.all(law>=0);tv=float(abs(law-d['target']).sum()/2);row=dict(budget=B,workspace_capacity=B,numerical_full_tv=tv);rows.append(row);np.savez_compressed(OUT/f'law_{B}.npz',law=law,target=d['target']);save(f'row_{B}.json',row)
save('summary.json',dict(passed=True,rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print(json.dumps(rows),flush=True)
