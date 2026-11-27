"""Coherent proposals and rank-one full-record weights for two detector readouts."""
import time
import numpy as np
from iteration119_matching import amplitudes


def proposal(q,S,alpha,direct,cap,rng):
    # The no-click bra is the displaced vacuum. For a click resolve
    # a^dagger int_0^1 t^N dt a by a heterodyne measurement after t^(N/2)a.
    intensity=abs(alpha)**2
    if direct:
        clicks=rng.random(len(q)) < -np.expm1(-intensity)
        sites=np.flatnonzero(clicks)
        u=rng.random(len(sites))
        eta=1+np.log1p(-u*(-np.expm1(-intensity[sites])))/intensity[sites]
        beta=np.sqrt(eta)*alpha[sites]+(rng.standard_normal(len(sites))+1j*rng.standard_normal(len(sites)))/np.sqrt(2)
        field=-alpha.conj()@q+(np.sqrt(eta)*beta.conj())@q[sites]
        row=clicks.astype(np.uint16)
    else:
        row=rng.poisson(intensity).astype(np.uint16)
        sites=np.repeat(np.arange(len(q)),row)
        field=-alpha.conj()@q
    J=q[sites]/alpha[sites,None]
    return row,field,J


def brute(S,v,J,cap):
    from functools import lru_cache
    n=len(J);cov=J@S@J.T;mean=J@S@v
    @lru_cache(None)
    def match(mask):
        if not mask:return np.r_[1.,np.zeros(cap)].astype(complex)
        first=mask&-mask;i=first.bit_length()-1;left=mask^first
        a=match(left).copy();a[1:]+=mean[i]*a[:-1].copy()
        rest=left
        while rest:
            bit=rest&-rest;j=bit.bit_length()-1
            a[1:]+=cov[i,j]*match(left^bit)[:-1];rest^=bit
        return a
    from scipy.special import factorial
    base=(v@S@v/2)**np.arange(cap+1)/factorial(np.arange(cap+1))
    return np.cumsum(np.convolve(base,match((1<<n)-1))[:cap+1])


def toy():
    from iteration119_matching import graphs
    rng=np.random.default_rng(11904)
    S=np.diag([.23,.31,.19]);q=np.linalg.qr(rng.normal(size=(12,3))+1j*rng.normal(size=(12,3)))[0]
    alpha=rng.normal(size=12)*.4+1j*.45;rows=[]
    for direct in (False,True):
        _,v,J=proposal(q,S,alpha,direct,7,rng)
        for cap in (3,5,7):
            start=time.perf_counter();actual=amplitudes(S,v,J,cap);ref=brute(S,v,J,cap)
            err=float(abs(actual-ref).max());assert err<1e-10,(err,cap)
            rows.append([direct,cap,len(J),len(graphs(cap)),err,time.perf_counter()-start])
    return rows


if __name__=="__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):print(toy(),flush=True)
