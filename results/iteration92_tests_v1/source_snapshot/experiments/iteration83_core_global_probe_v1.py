"""Bounded lightweight reference: 100 iterations of archived all-basis dual search.
This is an existing Oh/Kalra-Kocharovsky SDP method, not a new algorithm. The small
100-dimensional reference supplies dual lower estimates; candidate primal states
are not accepted merely because an optimizer terminates. No covariance clipping.
"""
from pathlib import Path
import os,sys,argparse,json,hashlib,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian_resource import search_resource
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),inp,ROOT/'src/gbslab/gaussian_resource.py',ROOT/'src/gbslab/exact.py'];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},maximum_iterations=100,source='https://arxiv.org/html/2606.29739v1'));xs=np.load(inp)['x'];rows=[]
for sector,x in enumerate(xs):
 V=np.eye(len(x))+2*x;start=time.perf_counter();result=search_resource(V,max_iterations=100,gradient_tolerance=1e-9);elapsed=time.perf_counter()-start;d=result['diagnostics'];row=dict(sector=sector,seconds=elapsed,iterations=result['iterations'],evaluations=result['evaluations'],optimizer_success=result['optimizer_success'],optimizer_message=result['optimizer_message'],diagnostics=d,candidate_numerically_feasible=bool(d['minimum_remainder_eigenvalue']>=0 and d['purity_residual']<1e-9));rows.append(row);save(f'row_{sector}.json',row);save(f'history_{sector}.json',result['history']);np.savez_compressed(OUT/f'candidate_{sector}.npz',pure=result['pure_covariance'],S=result['dual_multiplier'],L=result['dual_factor'],remainder=result['remainder']);print(row,flush=True)
save('summary.json',dict(rows=rows,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
