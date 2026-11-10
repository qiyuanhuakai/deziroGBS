"""Exactly shared pure Gaussian targets, with explicit MPS input rounding."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from iteration44_physical import exact_target
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def physical(n,strength,order):
    phases=[1,-1,1j,1] if n==4 else [1,-1,1j,1,-1j,1,1j,-1];H=np.array([[(-1)**((i&j).bit_count()) for j in range(n)] for i in range(n)],dtype=object);a=H@np.diag([F(int(complex(z).real)) for z in phases])@H/n;b=H@np.diag([F(int(complex(z).imag)) for z in phases])@H/n
    q=([F(1,2),F(1,4),F(1,2),F(1,8)] if strength=='moderate' else [F(1,4),F(1,8),F(1,16),F(1,4)])*(n//4);p=[1/x for x in q];S=np.block([[a,-b],[b,a]]);V=S@np.diag(p+q)@S.T;parameters=[(1-x)/(1+x) for x in q];D=np.diag(parameters);br=a@D@a.T-b@D@b.T;bi=a@D@b.T+b@D@a.T;ix=order+[j+n for j in order];V=V[np.ix_(ix,ix)];br=br[np.ix_(order,order)];bi=bi[np.ix_(order,order)];rounded=np.asarray(br,dtype=float)+1j*np.asarray(bi,dtype=float);error=max(abs(F(float(rounded[i,j].real))-br[i,j])+abs(F(float(rounded[i,j].imag))-bi[i,j]) for i in range(n) for j in range(n));assert all(F(float(v))==v for v in V.flat)
    return dict(covariance=V.tolist(),unitary_real=a.tolist(),unitary_imag=b.tolist(),small_variances=q,large_variances=p,pairing_parameters=parameters,pairing_real=br.tolist(),pairing_imag=bi.tolist(),entrywise_pairing_l1_rounding_max=error,order=order),rounded

def main():
    out=ROOT/'results/iteration47_inputs';out.mkdir(exist_ok=False);names=['research/iteration47_mixture_plan.md',*[f'experiments/{n}.py' for n in ('iteration47_inputs','iteration44_physical','iteration22_external','iteration33_run')],'src/gbslab/rational_gaussian.py'];hashes={n:digest(ROOT/n) for n in names}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    jobs=[dict(modes=n,strength=s,order=list(range(n)) if o=='forward' else list(reversed(range(n))),order_label=o) for n in (4,8) for s in ('moderate','strong') for o in ('forward','reverse')];save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies={}));rows=[]
    for i,job in enumerate(jobs):
        p,B=physical(job['modes'],job['strength'],job['order']);t=exact_target(p['covariance']);save(out/f'case_{i:02d}.json',encode(dict(job=job,physical=p,target=t)));np.savez_compressed(out/f'case_{i:02d}.npz',B=B,V=np.array(p['covariance'],float));rows.append(dict(case=i,**job,min_target_lower=float(min(a for a,b in t['target_intervals'])),pairing_rounding=float(p['entrywise_pairing_l1_rounding_max'])));print('input',i,flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items());save(out/'rows.json',rows);save(out/'summary.json',dict(planned=8,completed=8,unchanged_sources=True))
if __name__=='__main__':main()
