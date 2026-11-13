"""Photon-first route on the same complete lossy 1152-output Jiuzhang model.
Use the frozen analytic polynomial photon-subtraction core on the new active
mixed covariance. Two sectors produce integer counts independently; add counts
then threshold. This is approximate after Gaussian compression, not factor82's
equal-squeezing pure-state algorithm. Keep all explicit failures.
"""
from pathlib import Path
import os,sys,json,hashlib,shutil,time,argparse
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_polynomial_pnr import NativePolynomialPNRSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';dll=ROOT/'results/iteration73_polynomial_pilot_v2/polynomial.dll';d=np.load(inp);v,x=d['v'],d['x'];dirs=[Path(shutil.which('g++')).parent];count=32;rows=[];save('plan.json',dict(scope=__doc__,count=count,retained=[0,1,3],workers=16,seed=831300,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),inp,dll,ROOT/'src/gbslab/native_polynomial_pnr.py',ROOT/'native/gaussian_pnr_polynomial73_v1.cpp']},reference_mean_total_photons=float(sum(np.trace(xs) for xs in x)/2)))
for retained in (0,1,3):
 row=dict(retained=retained,count=count,success=False);outputs=[];start=time.perf_counter()
 try:
  for s in range(2):
   with NativePolynomialPNRSampler(v,x[s],dll,retained=retained,workers=16,runtime_dirs=dirs) as model:result=model.sample(count,831300+s)
   outputs.append(result['counts']);np.savez_compressed(OUT/f'sector_{retained}_{s}.npz',counts=result['counts']);save(f'sector_{retained}_{s}.json',dict(statistics=result['statistics'],wall_seconds=result['wall_seconds']))
  counts=outputs[0].astype(np.uint64)+outputs[1];bits=counts>0;row.update(success=True,mean_total_photons=float(counts.sum(1).mean()),mean_clicks=float(bits.sum(1).mean()),maximum_total_photons=int(counts.sum(1).max()));np.savez_compressed(OUT/f'combined_{retained}.npz',counts=counts,clicks=bits)
 except Exception as e:row['error']=str(e)
 row['seconds']=time.perf_counter()-start;rows.append(row);save(f'row_{retained}.json',row);print(row,flush=True)
save('summary.json',dict(rows=rows,scope='exploratory feasibility, no quality certificate from 32 records'));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
