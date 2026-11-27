"""Optimize positive antipodal coherent quadrature using the exact Gaussian Hilbert norm."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from scipy.spatial.distance import cdist


def positive(s,count,seed,steps=1200,initial=None):
    half=count//2
    dim=len(s)
    inv=1-np.tanh(s)
    pref=1/np.sqrt(np.cosh(s)).prod()
    scale=np.sqrt(np.tanh(s)/inv)
    seq=qmc.Sobol(dim,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(half))))
    start=norm.ppf(np.clip(seq[:half],1e-12,1-1e-12))*scale
    mass=np.exp(s.sum()/2)/(half+1)
    if initial is None:
        weights=np.full(half+1,mass)
    else:
        old,oldw=initial
        n=min(len(old),half)
        start[:n]=old[:n]
        weights=np.full(half+1,mass*.2)
        weights[:n+1]=oldw[:n+1]
    def fg(parameters):
        pts=np.vstack((np.zeros(dim),parameters[:half*dim].reshape(half,dim)))
        c=parameters[half*dim:]
        # Antipodal coherent states use the even part of the Gaussian kernel.
        plus=np.exp(-.5*cdist(pts,pts,'sqeuclidean'))
        minus=np.exp(-.5*cdist(pts,-pts,'sqeuclidean'))
        gram=(plus+minus)/2
        g=pref*np.exp(-.5*((pts**2)@inv))
        pred=gram@c
        err=pred-g
        dp=( ((plus-minus)/2)@(pts*c[:,None]) - pred[:,None]*pts + g[:,None]*(pts*inv) )
        return 1+c@(pred-2*g),np.r_[((2*c[:,None])*dp)[1:].ravel(),2*err]
    # Nonnegative weights also bound the cost of the subsequent rejection sampler.
    opt=minimize(fg,np.r_[start.ravel(),weights],jac=True,method='L-BFGS-B',
                 bounds=[(None,None)]*(dim*half)+[(0,None)]*(half+1),
                 options=dict(maxiter=steps,ftol=2e-16,gtol=1e-11,maxcor=24))
    pts=opt.x[:half*dim].reshape(half,dim);w=opt.x[half*dim:]
    pos=np.vstack((np.zeros(dim),pts,-pts));coeff=np.r_[w[0],w[1:]/2,w[1:]/2]
    use=coeff>1e-14
    pos=pos[use];coeff=coeff[use]
    gram=np.exp(-.5*cdist(pos,pos,'sqeuclidean'))
    coeff/=np.sqrt(coeff@gram@coeff)
    overlap=coeff@(pref*np.exp(-.5*((pos*pos)@inv)))
    info=dict(branches=len(coeff),budget=count,iterations=int(opt.nit),state_bound=float(np.sqrt(max(0,1-overlap**2))),
              acceptance=float(1/coeff.sum()**2))
    return pos,coeff,gram,info,(pts,w)
