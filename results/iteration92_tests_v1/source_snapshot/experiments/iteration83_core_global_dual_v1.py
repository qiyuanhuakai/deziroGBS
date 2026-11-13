"""Lightweight all-basis SDP dual check for the fixed-basis pure-core candidate.
Reuse the archived weighted-pure oracle (Oh et al.; Kalra-Kocharovsky 2606.29739).
Scale the fixed-basis PSD multiplier, evaluate a valid full physical-cone dual.
This is a numerical optimality bracket, not a sampling-cost or TV certificate.
"""
from pathlib import Path
import os,sys,argparse,json,hashlib,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian_resource import weighted_pure
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';base=ROOT/'results/iteration83_core_energy_probe_v2';files=[Path(__file__),inp,ROOT/'src/gbslab/gaussian_resource.py',base/'split_0.npz',base/'split_1.npz'];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},source='https://arxiv.org/html/2606.29739v1',scale_grid=[0,.01,.03,.1,.3,1,3,10,100]));xs=np.load(inp)['x'];rows=[]
for s,x in enumerate(xs):
 start=time.perf_counter();d=np.load(base/f'split_{s}.npz');V=np.eye(len(x))+2*x;R,Y=d['rotation'],d['dual'];S=R@Y@R.T;S=(S+S.T)/2;assert np.linalg.eigvalsh(S).min()>0;trace=[]
 def objective(scale):
  P,phi=weighted_pure(np.eye(len(x))+scale*S);dual=phi-scale*np.sum(S*V.T);trace.append(dict(scale=float(scale),dual_trace=float(dual)));return -dual
 grid=[0,.01,.03,.1,.3,1,3,10,100];values=[objective(a) for a in grid];i=int(np.argmin(values));lo,hi=grid[max(0,i-1)],grid[min(len(grid)-1,i+1)];opt=minimize_scalar(objective,bounds=(lo,hi),method='bounded',options=dict(xatol=1e-10));best=max(trace,key=lambda a:a['dual_trace']);scale=best['scale'];P,phi=weighted_pure(np.eye(len(x))+scale*S);upper=(np.trace(d['pure'])-len(x))/4;lower=(best['dual_trace']-len(x))/4;gap=upper-lower;assert gap>=-1e-9;row=dict(sector=s,fixed_basis_feasible_core_mean=float(upper),all_basis_dual_core_lower=float(lower),all_basis_core_gap=float(gap),scale=scale,scale_search_success=bool(opt.success),minimum_scaled_dual_eigenvalue=float(np.linalg.eigvalsh(scale*S).min()),oracle_minimum_noise_eigenvalue=float(np.linalg.eigvalsh(V-P).min()),seconds=time.perf_counter()-start,evaluations=len(trace));rows.append(row);save(f'row_{s}.json',row);save(f'trace_{s}.json',trace);np.savez_compressed(OUT/f'dual_{s}.npz',S=scale*S,oracle=P,feasible=d['pure']);print(row,flush=True)
save('summary.json',dict(rows=rows,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
