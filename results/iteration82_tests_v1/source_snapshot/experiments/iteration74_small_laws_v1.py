"""Enumerate complete induced click laws of both fused GPU variants."""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_philox_click import NativePhiloxClickSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
runtime=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent];cpu=ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll';libs={version:ROOT/f'results/iteration74_fused_pilot_v{version}/philox.dll' for version in (1,2)};save('plan.json',dict(driver_sha256=H(Path(__file__)),binary_hashes={str(k):H(p) for k,p in libs.items()},scope=__doc__));rows=[]
for case in range(6):
    data=np.load(ROOT/f'results/iteration71_branch_pilot_v1/input_{case}.npz');m,n=data['v'].shape;assert not data['variance'].any();patterns=np.array([[(mask>>i)&1 for i in range(m)] for mask in range(2**m)],np.uint8);draws=np.where(patterns==1,0.,np.nextafter(1.,0.));means=np.tile(data['mean'],(2**m,1))
    for version in (1,2):
        with NativePhiloxClickSampler(data['v'],data['x'],cpu,gpu_library=libs[version],mean=data['mean'],runtime_dirs=runtime) as model:
            for budget in ([1,2,4,8] if m==4 else [1,2,4,8,16,32,64,128]):
                result=model.replay(means,draws,branch_budget=budget);prob=result['conditional_click_probabilities'];law=np.prod(np.where(patterns,prob,1-prob),axis=1);reference=np.load(ROOT/f'results/iteration71_branch_pilot_v1/law_{case}_{budget}.npz')['law'];row=dict(case=case,version=version,budget=budget,wrong_bits=int(np.count_nonzero(patterns!=result['clicks'])),mass_error=float(abs(law.sum()-1)),tv_to_cpu=float(abs(law-reference).sum()/2),tv_to_target=float(abs(law-data['target']).sum()/2));rows.append(row);label=f'{case}_{version}_{budget}';np.savez_compressed(OUT/f'law_{label}.npz',patterns=patterns,clicks=result['clicks'],probabilities=prob,law=law,target=data['target'],reference=reference);save(f'row_{label}.json',row);assert row['wrong_bits']==0 and row['mass_error']<1e-8 and row['tv_to_cpu']<1e-8,row
    print(dict(case=case,passed=True),flush=True)
save('summary.json',dict(passed=True,rows=rows));shutil.copy2(__file__,OUT/Path(__file__).name);save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
