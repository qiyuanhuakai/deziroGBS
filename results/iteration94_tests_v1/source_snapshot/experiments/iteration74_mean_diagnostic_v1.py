"""Investigate pure-state mean deficit; retain paired old/new count records."""
from pathlib import Path
import os,sys,json,shutil,hashlib
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_structured_pnr import NativeStructuredPNRSampler
from gbslab.native_polynomial_pnr import NativePolynomialPNRSampler
OUT=ROOT/'results/iteration74_mean_diagnostic_v1';OUT.mkdir(exist_ok=False)
data=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');v=data['v'];t=data['t'];x=np.diag(np.r_[-t/(1+t),t/(1-t)]);runtime=[Path(shutil.which('g++')).parent];rows=[];records={}
for name,cls,lib in [('old',NativePolynomialPNRSampler,'iteration73_polynomial_pilot_v2/polynomial.dll'),('new',NativeStructuredPNRSampler,'iteration74_structured_pilot_v1/structured.dll')]:
    with cls(v,x if name=='old' else t,ROOT/'results'/lib,retained=0,workers=16,runtime_dirs=runtime) as model:r=model.sample(200000,744001)
    records[name]=r['counts'];np.savez_compressed(OUT/f'{name}.npz',counts=r['counts']);rows.append(dict(name=name,mean=float(r['counts'].sum()/200000),statistics=r['statistics']))
values=np.diag(x)
def integrand(s):
    xt=values/(1+s*values);survival=np.prod(1+s*values)**(-.5);density=.5*xt.sum()*survival;remaining=(1-s)*xt;after=.5*remaining.sum()+(remaining@remaining)/remaining.sum()
    return density*(1+after)
integral,error=quad(integrand,0,1,epsabs=1e-12)
result=dict(rows=rows,mismatches=int(np.count_nonzero(records['old']!=records['new'])),target_mean=float(np.trace(x)/2),first_jump_plus_conditional_remaining=integral,quadrature_error=error,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(OUT/'summary.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());shutil.copy2(__file__,OUT/Path(__file__).name);print(json.dumps(result),flush=True)
