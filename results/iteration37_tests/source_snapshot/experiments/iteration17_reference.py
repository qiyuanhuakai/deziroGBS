"""Independent high-precision binomial expansion of physical (unshifted) Wick moments."""
import argparse
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import sys
import mpmath as mp
import numpy as np
from iteration14_run import save

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.meixner_probe import uniform_parameters


def reference(u,r,eta,indices,V,digits):
    m=len(u); parameters=uniform_parameters(m,r,eta)
    M=-eta*math.sinh(r)*math.cosh(r)*(u@u.T)
    with mp.workdps(digits):
        cast=lambda z:mp.mpc(float(np.real(z)),float(np.imag(z)))
        n=mp.mpf(parameters['nout']); b=mp.mpf(parameters['b']); v=b*(1+b)
        matrix=[[mp.mpc(0) for _ in range(2*m)] for _ in range(2*m)]
        for i in range(m):
            for j in range(m): matrix[i][j]=cast(M[i,j]).conjugate(); matrix[i+m][j+m]=cast(M[i,j])
            matrix[i][i+m]=n; matrix[i+m][i]=n
        @lru_cache(None)
        def wick(counts):
            if not any(counts): return mp.mpc(1)
            i=max(k for k,x in enumerate(counts) if x); remaining=list(counts); remaining[i]-=1; terms=[]
            for j,multiplicity in enumerate(remaining):
                if multiplicity and matrix[i][j]:
                    remaining[j]-=1; terms.append(multiplicity*matrix[i][j]*wick(tuple(remaining))); remaining[j]+=1
            return mp.fsum(terms)
        @lru_cache(None)
        def factorial(index): return wick(index+index)/math.prod(math.factorial(k) for k in index)
        cumulative=[mp.mpf(0)]*(1<<m); coefficients=[]; laws=[]; max_degree=int(np.max(np.sum(indices,axis=1)))
        for degree in range(max_degree+1):
            for array in indices[np.sum(indices,axis=1)==degree]:
                index=tuple(map(int,array)); terms=[]
                for sub in itertools.product(*(range(k+1) for k in index)):
                    terms.append(math.prod(math.comb(k,j) for k,j in zip(index,sub))*(-b)**(degree-sum(sub))*factorial(sub))
                coefficient=mp.fsum(terms)
                if abs(coefficient.imag)>mp.mpf(10)**(-digits//2): raise ArithmeticError('Complex reference coefficient')
                c=coefficient.real; coefficients.append(c)
                mask=sum(1<<i for i,k in enumerate(index) if k)
                cumulative[mask]+=c*(-b)**(degree-mask.bit_count())/v**degree
            # Direct mask-by-subset evaluation, independent of the fast butterfly.
            raw=[]
            for mask in range(1<<m):
                density=mp.fsum(A*mp.fprod(1 if (mask>>i)&1 else -b for i in range(m) if (support>>i)&1) for support,A in enumerate(cumulative))
                raw.append((b/(1+b))**mask.bit_count()*(1/(1+b))**(m-mask.bit_count())*density)
            laws.append([mp.nstr(x,digits) for x in raw])
        # Independent high-precision target from vacuum determinants + inclusion/exclusion.
        no_click=[]
        for mask in range(1<<m):
            active=[i for i in range(m) if (mask>>i)&1]; ids=active+[i+m for i in active]
            if not ids: no_click.append(mp.mpf(1)); continue
            sub=mp.matrix([[(mp.mpf(float(V[i,j]))+int(i==j))/2 for j in ids] for i in ids])
            no_click.append(1/mp.sqrt(mp.det(sub)))
        target=[]
        for mask in range(1<<m):
            zeros=((1<<m)-1)^mask; subset=mask; terms=[]
            while True:
                terms.append((-1)**subset.bit_count()*no_click[zeros|subset])
                if subset==0: break
                subset=(subset-1)&mask
            target.append(mp.fsum(terms))
        return {'digits':digits,'coefficients_decimal':[mp.nstr(x,digits) for x in coefficients],
            'raw_laws_decimal':laws,'target_decimal':[mp.nstr(x,digits) for x in target],
            'physical_wick_cache_states':wick.cache_info().currsize}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); panel=ROOT/'results/iteration17_panel'
    inputs=[r for r in json.loads((panel/'inputs.json').read_text()) if r['modes']==4]
    save(out/'plan.json',{'input_indices':[r['index'] for r in inputs],'digits':[80,120],
        'scope':'All predeclared m4 inputs. Independent unshifted moment/binomial path and target determinants. High precision convergence, not outward-rounded certification.'})
    rows=[]
    for job in inputs:
        i=job['index']
        with np.load(panel/f'input_{i:02d}.npz') as data: u=data['U']; V=data['covariance']; target=data['target']
        with np.load(panel/f'input_{i:02d}_coefficients.npz') as data: indices=data['indices']; coefficients=data['coefficients']
        low=reference(u,job['r'],job['eta'],indices,V,80); high=reference(u,job['r'],job['eta'],indices,V,120)
        save(out/f'input_{i:02d}.json',{'reference80':low,'reference120':high})
        with mp.workdps(140):
            c=list(map(mp.mpf,high['coefficients_decimal'])); hc=max(abs(mp.mpf(a)-b) for a,b in zip(low['coefficients_decimal'],c))
            ce=max(abs(mp.mpf(float(a))-b) for a,b in zip(coefficients,c))
            relative=max(abs(mp.mpf(float(a))-b)/max(1,abs(b)) for a,b in zip(coefficients,c))
            target_error=max(abs(mp.mpf(float(a))-mp.mpf(b)) for a,b in zip(target,high['target_decimal']))
            raw_error=mp.mpf(0); raw_convergence=mp.mpf(0)
            for d,law in enumerate(high['raw_laws_decimal']):
                with np.load(panel/f'input_{i:02d}_D{d}.npz') as data: raw=data['raw']
                raw_error=max(raw_error,max(abs(mp.mpf(float(a))-mp.mpf(b)) for a,b in zip(raw,law)))
                raw_convergence=max(raw_convergence,max(abs(mp.mpf(a)-mp.mpf(b)) for a,b in zip(low['raw_laws_decimal'][d],law)))
            row=dict(index=i,**{k:mp.nstr(v,40) for k,v in {'coefficient_abs_error':ce,'coefficient_scaled_error':relative,
                'coefficient_precision_convergence':hc,'raw_law_abs_error':raw_error,'raw_law_precision_convergence':raw_convergence,
                'target_abs_error':target_error}.items()})
        rows.append(row); save(out/'rows.json',rows); print(row,flush=True)
    files=['experiments/iteration17_reference.py','src/gbslab/meixner_probe.py']
    for f in files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'source_hashes.json',{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})


if __name__=='__main__': main()
