"""Exact Rayleigh witnesses distinguish a physicality violation from eigensolver noise."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,shutil
from fractions import Fraction as F
from pathlib import Path
from time import perf_counter
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
ROOT=Path(__file__).resolve().parents[1]

def integers(a):
    values=[F(float(x)) for x in np.asarray(a).flat];den=max(v.denominator for v in values);return np.array([v.numerator*(den//v.denominator) for v in values],dtype=object).reshape(a.shape),den

def main():
    out=ROOT/'results/iteration34_physicality';out.mkdir(exist_ok=False);names=['experiments/iteration34_physicality.py','experiments/iteration22_external.py','experiments/iteration33_run.py'];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    name='results/iteration31_panel/large_input.npz';value=digest(ROOT/name);assert value==json.loads((ROOT/'results/iteration31_manifest.json').read_text())['artifacts'][name];dependencies={name:value};m=144;n=2*m;O=np.block([[np.zeros((m,m)),np.eye(m)],[-np.eye(m),np.zeros((m,m))]]);jobs=['stored_mixed','thermal_8','thermal_40','rank_one_5','rank_one_4'];save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=dependencies,method='Lowest float eigenvector of V+iOmega, then exact dyadic quadratic form. A negative exact quotient proves violation; nonnegative quotient alone is not a PSD certificate. Structured controls have independent physical constructions.'))
    rows=[]
    for i,job in enumerate(jobs):
        start=perf_counter()
        if i==0:V=np.load(ROOT/name)['covariance']
        elif job.startswith('thermal'):V=(1+2.**(-int(job.split('_')[-1])))*np.eye(n)
        else:
            a=np.array([(-1 if j%3==1 else 1)*2.**(-int(job.split('_')[-1])) for j in range(n)]);V=np.eye(n)+np.outer(a,a)
        vals,vecs=np.linalg.eigh(V+1j*O);w=vecs[:,0];Z,dv=integers(V);U,du=integers(np.stack([w.real,w.imag]));a,b=U;omega_b=np.r_[b[m:],-b[:m]];numerator=int(a@(Z@a)+b@(Z@b)-2*dv*(a@omega_b));form=F(numerator,dv*du*du);norm=F(int(a@a+b@b),du*du);quotient=form/norm
        # Independent direct Fraction summation, not the integer matrix product.
        ar=[F(float(x)) for x in w.real];br=[F(float(x)) for x in w.imag];diagonal=sum((F(float(V[j,k]))*(ar[j]*ar[k]+br[j]*br[k]) for j in range(n) for k in range(n)),F(0));symplectic=2*sum((ar[j]*br[j+m]-ar[j+m]*br[j] for j in range(m)),F(0));assert diagonal-symplectic==form
        np.savez_compressed(out/f'case_{i:02d}.npz',covariance=V,witness=w);row=dict(sequence=i,job=job,floating_minimum_eigenvalue=float(vals[0]),exact_quadratic_form=form,exact_norm_squared=norm,exact_rayleigh=quotient,exact_rayleigh_float=float(quotient),strict_violation_proved=quotient<0,independent_fraction_check=True,interpretation='Strict violation of V+iOmega PSD for these exact binary64 entries' if quotient<0 else 'This witness is nonnegative; full PSD is only independently known for the structured controls',seconds=perf_counter()-start);rows.append(row);save(out/f'case_{i:02d}.json',encode(row));print(job,float(vals[0]),float(quotient),quotient<0,flush=True)
    unchanged=hashes=={n:digest(ROOT/n) for n in names};assert unchanged;save(out/'rows.json',encode(rows));save(out/'summary.json',dict(planned=5,completed=5,unchanged_sources=unchanged,source_hashes=hashes,dependencies=dependencies,strict_violations=[r['job'] for r in rows if r['strict_violation_proved']]))
if __name__=='__main__':main()
