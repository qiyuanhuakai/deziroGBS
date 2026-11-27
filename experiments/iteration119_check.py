"""Independent normalization, Fock-amplitude and high-order contraction checks."""
import json
import math
import numpy as np
from threadpoolctl import threadpool_limits
from iteration118_auxiliary import FAC
from iteration119_benchmark import OUT,stable
from iteration119_records import brute,proposal
from iteration119_matching import amplitudes


def completeness(alpha,d):
    # z=beta-sqrt(t)*alpha. Under the complex unit Gaussian z, the remaining
    # integrand is a polynomial of degree at most 2d in its real coordinates.
    x,w=np.polynomial.hermite.hermgauss(d+2);z=(x[:,None]+1j*x).ravel()
    W=(w[:,None]*w).ravel()/np.pi
    tt,ww=np.polynomial.legendre.leggauss(32)
    fac=np.sqrt(FAC[:d+1]);orders=np.arange(d+1)
    v=np.exp(-abs(alpha)**2/2)*(-alpha.conjugate())**orders/fac
    matrix=np.outer(v.conj(),v)
    for t,weight in zip((1+tt)/2,ww/2):
        ell=-(1-t)*alpha.conjugate()+np.sqrt(t)*z.conj()
        powers=ell[:,None]**orders/fac
        f=alpha*powers;f[:,1:]+=np.sqrt(orders[1:])*powers[:,:-1]
        matrix+=weight*np.exp(-(1-t)*abs(alpha)**2)*(f.conj().T@(W[:,None]*f))
    return float(abs(matrix-np.eye(d+1)).max())


def run():
    errors=[completeness(a,d) for a in (.05+.11j,.71-.18j,1.19+.68j) for d in (4,10,14)]
    rng=np.random.default_rng(11986);r=4;m=12
    q=np.linalg.qr(rng.normal(size=(m,r))+1j*rng.normal(size=(m,r)))[0]
    u=np.linalg.qr(rng.normal(size=(r,r))+1j*rng.normal(size=(r,r)))[0]
    S=(u*np.array([.31,.23,.19,.16]))@u.T
    v=rng.normal(size=r)+1j*rng.normal(size=r)
    rows=[]
    for k in (5,7,9,10):
        for scale in (1.,.19):
            J=q[:11].copy();J[1]/=scale;J[5]/=scale
            actual,bad=stable(S,v,J,k,.8)
            ref=brute(S,v,J,k)
            rows.append(dict(pairs=k,monomers=len(J),deferred=bad,max_abs=float(abs(ref-actual).max())))
    assert max(errors)<2e-12 and max(x["max_abs"] for x in rows)<1e-9,(errors,rows)
    info=dict(continuous_click_completeness=max(errors),quadrature_cases=len(errors),exact_subset_checks=rows)
    (OUT/"checks.json").write_text(json.dumps(info,indent=2)+"\n",encoding="utf8")
    print(info,flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):run()
