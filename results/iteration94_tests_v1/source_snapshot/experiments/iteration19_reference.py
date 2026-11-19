"""Independent high-precision one-mode tails from two binomial series factors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import mpmath as mp
import numpy as np
from iteration14_run import save
ROOT=Path(__file__).resolve().parents[1]


def tails(n,s,dimensions,digits):
    with mp.workdps(digits):
        n=mp.mpf(n); s=mp.mpf(s); q=[(n-s)/(1+n-s),(n+s)/(1+n+s)]
        factors=[]
        for z in q:
            a=[mp.mpf(1)]
            for k in range(1,max(dimensions)): a.append(a[-1]*(mp.mpf(k)-mp.mpf('.5'))/k*z)
            factors.append(a)
        scale=((1+n-s)*(1+n+s))**(-mp.mpf('.5'))
        probabilities=[scale*mp.fsum(factors[0][j]*factors[1][k-j] for j in range(k+1)) for k in range(max(dimensions))]
        return [mp.nstr(1-mp.fsum(probabilities[:d]),digits) for d in dimensions]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); panel=ROOT/'results/iteration19_panel'
    plan=json.loads((panel/'plan.json').read_text()); dims=plan['dimensions']; n=math.sinh(plan['physical_parameters']['core_squeezing'])**2
    save(out/'plan.json',dict(input_indices=list(range(20)),dimensions=dims,digits=[80,120],scope='All 1000 one-mode marginals, factorized binomial generating series. Binary64 n and |Mii| inputs; high precision convergence, not interval certification.'))
    rows=[]
    for i in range(20):
        with np.load(panel/f'input_{i:02d}.npz') as data: M=data['anomalous']; production=data['marginal_tail_upper_estimates']; remainders=data['remainder_upper_estimates']
        low=[tails(n,float(abs(M[j,j])),dims,80) for j in range(len(M))]
        high=[tails(n,float(abs(M[j,j])),dims,120) for j in range(len(M))]
        save(out/f'input_{i:02d}.json',dict(reference80=low,reference120=high))
        with mp.workdps(140):
            convergence=max(abs(mp.mpf(a)-mp.mpf(b)) for la,lb in zip(low,high) for a,b in zip(la,lb))
            absolute=max(abs(mp.mpf(float(a))-mp.mpf(b)) for la,lb in zip(production,high) for a,b in zip(la,lb))
            relative=max(abs(mp.mpf(float(a))-mp.mpf(b))/mp.mpf(b) for la,lb in zip(production,high) for a,b in zip(la,lb))
        row=dict(index=i,marginals=len(M),max_abs_error=str(absolute),max_relative_error=str(relative),precision_convergence=str(convergence),max_chernoff_remainder=float(max(remainders)))
        rows.append(row); save(out/'rows.json',rows); print(row,flush=True)
    shutil.copy2(__file__,out/'iteration19_reference.py'); save(out/'source_hashes.json',{'experiments/iteration19_reference.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

if __name__=='__main__': main()
