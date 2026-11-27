"""Check the bright number instrument against matrix exponentials and Charlier CDFs."""
import argparse
import ctypes as ct
import json
from math import factorial,comb,sqrt
from pathlib import Path
import tempfile
import numpy as np
from scipy.linalg import expm
from scipy.special import pdtr
from threadpoolctl import threadpool_limits

from iteration122_loss import build,OUT,json_write


def cdf(gram,alpha,k):
    u=abs(alpha)**2
    k=np.array(k,dtype=float)
    degree=len(gram)-1
    phase=np.exp(-1j*np.angle(alpha)*np.arange(degree+1))
    coef=np.zeros(2*degree+1)
    for i,j in np.ndindex(degree+1,degree+1):
        weight=(phase[i].conjugate()*gram[i,j]*phase[j]).real
        # G(t)G(s)=exp(ts) G(t+s+ts/sqrt(u)).
        for p in range(min(i,j)+1):
            for a in range(min(i,j)-p+1):
                x,y=i-p-a,j-p-a
                l=x+y+a
                term=sqrt(factorial(i)*factorial(j)*factorial(l))/(factorial(x)*factorial(y)*factorial(a)*factorial(p))
                coef[l]+=weight*term/u**(a/2)
    assert abs(coef[0]-1)<1e-12
    poly=[np.ones_like(k),(k-u)/sqrt(u)]
    for i in range(1,2*degree-1):
        poly.append(((k-u-i)/sqrt(u)*poly[-1]-sqrt(i)*poly[-2])/sqrt(i+1))
    # Centre-stable Poisson mass; use the large-n Stirling expansion only at n>=32.
    from scipy.special import gammaln,xlogy
    kplus=np.maximum(1,k)
    x=(k-u)/u
    near=abs(x)<.01
    d=np.empty_like(k)
    d[~near]=xlogy(k[~near],k[~near]/u)-(k[~near]-u)
    d[near]=u*sum((-1)**j*x[near]**j/(j*(j-1)) for j in range(2,13))
    logp=-d-.5*np.log(2*np.pi*kplus)-(1/12-1/(360*kplus**2)+1/(1260*kplus**4))/kplus
    small=k<32
    logp[small]=-u+k[small]*np.log(u)-gammaln(k[small]+1)
    return coef[0]*pdtr(k,u)-sqrt(u)*np.exp(logp)*sum(coef[j]*poly[j-1]/sqrt(j) for j in range(1,len(coef)))


def test(lib):
    rng=np.random.default_rng([122,610])
    rand=rng.standard_normal((5,3))+1j*rng.standard_normal((5,3))
    mixed=rand@rand.conj().T;mixed/=mixed.trace()
    fock=np.diag([0.,0.,0.,0.,1.]).astype(complex)
    cases=[(mixed,7.),(fock,17.),(mixed,35.),(fock,2e5),(mixed,2e5),(mixed,1e10),(fock,1e10)]
    results=[];n=131072
    allowance=sqrt(np.log(2*len(cases)/.01)/(2*n))
    for i,(rho,u) in enumerate(cases):
        alpha=np.array([np.sqrt(u)*np.exp(.71j)])
        records=np.zeros(n,np.uint64);trials=np.zeros(1,np.uint64)
        rc=lib.local122(n,rho.ctypes.data,alpha.ctypes.data,1220610+i,records.ctypes.data,trials.ctypes.data)
        assert rc==0
        # Include the jump's two sides. Truncate no histogram tails.
        keys,counts=np.unique(records,return_counts=True)
        before=keys.astype(float)-1
        low=np.zeros(len(keys));positive=before>=0
        low[positive]=cdf(rho,alpha[0],before[positive])
        ref=cdf(rho,alpha[0],keys)
        cum=counts.cumsum()/n
        ks=max(abs(cum-ref).max(),abs(cum-counts/n-low).max())
        assert ks<=allowance,(u,ks,allowance)
        row=dict(intensity=u,input='Fock 4' if rho is fock else 'mixed coherent superpositions',
                 rows=n,ks=float(ks),simultaneous_dkw99=allowance,outer_proposals=int(trials[0]))
        if u<=35:
            cutoff=160
            ad=np.diag(np.sqrt(np.arange(1,cutoff)),-1)
            displacement=expm(alpha[0]*ad-alpha[0].conjugate()*ad.T)[:,:5]
            prob=np.einsum('ij,jk,ik->i',displacement.conj(),rho,displacement).real
            err=float(abs(np.cumsum(prob)-cdf(rho,alpha[0],np.arange(cutoff))).max())
            assert err<1e-11,(u,err)
            row['independent_fock_cdf_max_error']=err
        results.append(row)
    return results


if __name__=='__main__':
    with threadpool_limits(1),tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        lib=build(tmp)
        try:
            rows=test(lib)
            json_write(OUT/'bright_instrument.json',rows)
            print([(r['intensity'],r['ks']) for r in rows],flush=True)
        finally:
            import os
            if os.name=='nt':
                import _ctypes
                _ctypes.FreeLibrary(lib._handle)
