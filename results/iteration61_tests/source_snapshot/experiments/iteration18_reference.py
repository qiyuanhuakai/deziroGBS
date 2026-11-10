"""High-precision matrix-trace/characteristic-polynomial reference, without SVD."""
import argparse
import hashlib
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


def reference(u,r,eta,b,degree,digits):
    m=len(u); n=eta*math.sinh(r)**2; M=-eta*math.sinh(r)*math.cosh(r)*(u@u.T)
    with mp.workdps(digits):
        n=mp.mpf(n); b=mp.mpf(b)
        real=[[mp.mpf(float(z.real)) for z in row] for row in M]; imag=[[mp.mpf(float(z.imag)) for z in row] for row in M]
        C=mp.matrix(2*m)
        for i in range(m):
            for j in range(m):
                C[i,j]=real[i][j]+(n if i==j else 0); C[i+m,j+m]=-real[i][j]+(n if i==j else 0)
                C[i,j+m]=C[i+m,j]=imag[i][j]
        vacuum=[[mp.mpf(0)]*(1<<m) for _ in range(degree+1)]
        for d in range(degree+1): vacuum[d][0]=mp.mpf(1)
        exact=[mp.mpf(1)]*(1<<m)
        for mask in range(1,1<<m):
            modes=[i for i in range(m) if (mask>>i)&1]; ids=modes+[i+m for i in modes]; size=len(ids)
            sub=mp.matrix([[C[i,j] for j in ids] for i in ids]); B=(sub-b*mp.eye(size))/(1+b)
            power=mp.eye(size); traces=[]; polynomial=[mp.mpf(1)]
            for k in range(1,size+1):
                power=power*B; traces.append(mp.fsum(power[j,j] for j in range(size)))
                polynomial.append(mp.fsum((-1)**(j-1)*polynomial[k-j]*traces[j-1] for j in range(1,k+1))/k)
            g=[mp.mpf(1)]; cumulative=mp.mpf(1); vacuum[0][mask]=cumulative/(1+b)**len(modes)
            for d in range(1,degree+1):
                g.append(-mp.fsum((d-mp.mpf(j)/2)*polynomial[j]*g[d-j] for j in range(1,min(d,size)+1))/d)
                cumulative+=g[-1]; vacuum[d][mask]=cumulative/(1+b)**len(modes)
            exact[mask]=mp.det(mp.eye(size)+sub)**(-mp.mpf('.5'))
        def invert(values):
            output=[]
            for bit in range(1<<m):
                zero=((1<<m)-1)^bit; subset=bit; terms=[]
                while True:
                    terms.append((-1)**subset.bit_count()*values[zero|subset])
                    if subset==0: break
                    subset=(subset-1)&bit
                output.append(mp.fsum(terms))
            return [mp.nstr(x,digits) for x in output]
        return dict(digits=digits,raw_laws_decimal=[invert(v) for v in vacuum],target_decimal=invert(exact))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); panel=ROOT/'results/iteration18_panel'
    jobs=json.loads((panel/'plan.json').read_text())['jobs'][:12]
    save(out/'plan.json',dict(input_indices=list(range(12)),policies=['paper','physical_mean'],degree=64,digits=[80,120],scope='All m4 inputs, characteristic polynomial from matrix traces, explicit inclusion-exclusion. Binary64 physical moments as inputs; convergence is not interval certification.'))
    rows=[]
    for i,job in enumerate(jobs):
        with np.load(panel/f'input_{i:02d}.npz') as data: u=data['U']; target=data['target']
        pars=uniform_parameters(4,job['r'],job['eta'])
        for policy in ('paper','physical_mean'):
            b=pars['b'] if policy=='paper' else pars['nout']
            low=reference(u,job['r'],job['eta'],b,64,80); high=reference(u,job['r'],job['eta'],b,64,120)
            save(out/f'input_{i:02d}_{policy}.json',dict(reference80=low,reference120=high))
            with np.load(panel/f'input_{i:02d}_{policy}.npz') as data: degrees=data['degrees']; raw=data['raw']
            with mp.workdps(140):
                convergence=max(abs(mp.mpf(a)-mp.mpf(b)) for al,bl in zip(low['raw_laws_decimal'],high['raw_laws_decimal']) for a,b in zip(al,bl))
                error=max(abs(mp.mpf(float(a))-mp.mpf(b)) for d,law in zip(degrees,raw) for a,b in zip(law,high['raw_laws_decimal'][int(d)]))
                target_error=max(abs(mp.mpf(float(a))-mp.mpf(b)) for a,b in zip(target,high['target_decimal']))
                row=dict(index=i,policy=policy,precision_convergence=mp.nstr(convergence,40),raw_law_abs_error=mp.nstr(error,40),target_abs_error=mp.nstr(target_error,40))
            rows.append(row); save(out/'rows.json',rows); print(row,flush=True)
    files=['experiments/iteration18_reference.py','src/gbslab/meixner_probe.py']
    for f in files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'source_hashes.json',{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})

if __name__=='__main__': main()
