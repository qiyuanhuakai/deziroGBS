"""144-output 50-input PNR cost, retaining uncut counts and failures.

Small fixed batches quantify the cost of retained analytic photon factors;
this is not a large-distribution quality certificate or a click-only comparison.
"""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_polynomial_pnr import NativePolynomialPNRSampler
from gbslab.native_pnr_newton import NativeNewtonPNRSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
runtime=[Path(shutil.which('g++')).parent];poly=ROOT/'results/iteration73_polynomial_pilot_v2/polynomial.dll';finite=ROOT/'results/iteration72_newton_pilot_v1/newton.dll';source=ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110/input.npz';data=np.load(source)
save('plan.json',dict(driver_sha256=H(Path(__file__)),binary_hashes={str(p):H(p) for p in (poly,finite)},input_sha256=H(source),methods=['poly0','poly1','poly3','finite'],samples=32,repetitions=3,workers=16,seed_base=738001,scope=__doc__))
rows=[];methods=['poly0','poly1','poly3','finite']
for rep in range(3):
    for method in methods[rep:]+methods[:rep]:
        label=f'{rep}_{method}';save(f'started_{label}.json',dict(seed=738001+rep,samples=32));start=time.perf_counter();row=dict(rep=rep,method=method,samples=32,seed=738001+rep)
        try:
            model=NativeNewtonPNRSampler(data['v'],data['x0'],finite,mean=data['mean'],noise_variance=data['variance'],workers=16,runtime_dirs=runtime) if method=='finite' else NativePolynomialPNRSampler(data['v'],data['x0'],poly,mean=data['mean'],noise_variance=data['variance'],retained=int(method[-1]),workers=16,runtime_dirs=runtime)
            with model:r=model.sample(32,738001+rep)
            row.update(success=True,wall_seconds=time.perf_counter()-start,sampling_seconds=r['wall_seconds'],statistics=r['statistics']);np.savez_compressed(OUT/f'counts_{label}.npz',counts=r['counts'])
        except Exception as e:row.update(success=False,wall_seconds=time.perf_counter()-start,error=str(e))
        rows.append(row);save(f'row_{label}.json',row);print(row,flush=True)
save('summary.json',dict(completed=True,all_succeeded=all(r['success'] for r in rows),rows=rows));shutil.copy2(__file__,OUT/Path(__file__).name);save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
