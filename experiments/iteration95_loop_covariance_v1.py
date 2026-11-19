"""Invert discrete Gaussian count covariances using the Hermite expansion."""
import numpy as np
from scipy.special import ndtri

def thresholds(law,finite_support):
    missing=np.maximum(0,1-law.sum(1))
    if finite_support:missing[:]=0.
    tails=np.column_stack([law[:,1:],missing])[:,::-1].cumsum(1)[:,::-1]
    return np.ascontiguousarray(-ndtri(tails.clip(0,1)))

def calibrate(cuts,target,order=96):
    valid=np.isfinite(cuts)
    x=np.where(valid,cuts,0.)
    phi=np.exp(-.5*x*x)/np.sqrt(2*np.pi)*valid
    # h_n = probabilists' Hermite polynomial divided by sqrt(n!).
    prev=np.zeros_like(x);current=np.ones_like(x)
    coeff=[]
    for n in range(1,order+1):
        coeff.append(np.sum(phi*current,axis=1)/np.sqrt(n))
        nxt=(x*current-np.sqrt(n-1)*prev)/np.sqrt(n)
        prev,current=current,nxt
    coeff=np.array(coeff)
    i,j=np.triu_indices(len(target),1)
    products=coeff[:,i]*coeff[:,j]
    def evaluate(rho):
        ans=np.zeros(len(i))
        for term in products[::-1]:ans=(ans+term)*rho
        return ans
    lo=np.full(len(i),-.85);hi=-lo
    goal=target[i,j]
    assert np.min(goal-evaluate(lo))>0 and np.min(evaluate(hi)-goal)>0
    for _ in range(48):
        mid=(lo+hi)/2
        smaller=evaluate(mid)<goal
        lo=np.where(smaller,mid,lo);hi=np.where(smaller,hi,mid)
    rho=(lo+hi)/2
    corr=np.eye(len(target));corr[i,j]=corr[j,i]=rho
    bound=np.max(abs(rho)**(order+1)*np.sqrt(target.diagonal()[i]*target.diagonal()[j]))
    return corr,dict(max_series_residual=float(max(abs(evaluate(rho)-goal))),hermite_tail_bound=float(bound),
                     min_rho=float(rho.min()),max_rho=float(rho.max()))
