"""Replay the two previous noisy K3 holdouts; these are not new cohorts."""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_structured_pnr import NativeStructuredPNRSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
library=ROOT/'results/iteration74_structured_pilot_v1/structured.dll';runtime=[Path(shutil.which('g++')).parent];save('plan.json',dict(driver_sha256=H(Path(__file__)),binary_sha256=H(library),cases=[3,4],retained=3,batches=20,records_per_batch=100000,scope=__doc__));rows=[]
for case in (3,4):
    data=np.load(ROOT/f'results/iteration73_polynomial_pilot_v2/input_{case}.npz');n=data['v'].shape[1];t=-np.diag(data['x'])[:n]/(1+np.diag(data['x'])[:n])
    with NativeStructuredPNRSampler(data['v'],t,library,noise_variance=data['variance'],retained=3,workers=16,runtime_dirs=runtime) as model:
        for batch in range(20):
            prior=json.loads((ROOT/f'results/iteration73_pnr_holdout_v1/status_{case}_poly3_{batch}.json').read_text());r=model.sample(100000,prior['seed']);old=np.load(ROOT/f'results/iteration73_pnr_holdout_v1/counts_{case}_poly3_{batch}.npz')['counts'];row=dict(case=case,batch=batch,seed=prior['seed'],mismatched_entries=int(np.count_nonzero(old!=r['counts'])),seconds=r['wall_seconds'],statistics=r['statistics']);rows.append(row);np.savez_compressed(OUT/f'counts_{case}_{batch}.npz',counts=r['counts']);save(f'row_{case}_{batch}.json',row)
    print(dict(case=case,mismatches=sum(r['mismatched_entries'] for r in rows if r['case']==case)),flush=True)
save('summary.json',dict(completed=True,passed=all(r['mismatched_entries']==0 for r in rows),rows=rows,records=4000000));shutil.copy2(__file__,OUT/Path(__file__).name);save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
