"""Positive coherent-state quadrature in the input source space."""
import itertools
import math
import numpy as np
from scipy.optimize import nnls, minimize
from scipy.linalg import solve_triangular


def positive(s,budget):
    # Each center denotes the equal-weight orbit over independent sign flips.
    # Normalized Hilbert-space overlaps can be evaluated without the output optics.
    r=len(s)
    levels=[np.r_[0,np.linspace(.42,3.7,10)*np.sqrt(np.tanh(v)/.45)] for v in s]
    coords=np.array(list(itertools.product(*levels)))
    cost=2**np.count_nonzero(coords,axis=1)
    norm=np.prod((1+np.exp(-2*coords**2))/2,axis=1)
    target=np.exp(-.5*(coords**2)@(1-np.tanh(s)))/np.sqrt(np.cosh(s)).prod()
    indices=[];basis=np.empty((len(coords),0));remain=budget
    coeff=np.empty(0)
    while remain:
        delta=target-basis@coeff
        # Residual norm for an orthogonal projection. The final restricted fit
        # enforces positive coefficients, keeping the rejection envelope small.
        if indices:
            lower=np.linalg.cholesky(basis[indices]+np.eye(len(indices))*1e-14)
            wh=solve_triangular(lower,basis.T,lower=True)
            residual_norm=np.maximum(1e-14,norm-(wh*wh).sum(0))
        else:residual_norm=norm
        score=np.maximum(delta,0)**2/(residual_norm*cost)
        score[cost>remain]=0
        score[indices]=0
        j=int(score.argmax())
        if score[j]<1e-19:break
        indices.append(j)
        remain-=int(cost[j])
        a=coords[j]
        overlap=np.prod(.5*(np.exp(-.5*(coords-a)**2)+np.exp(-.5*(coords+a)**2)),axis=1)
        basis=np.column_stack([basis,overlap])
        lower=np.linalg.cholesky(basis[indices]+np.eye(len(indices))*1e-14)
        coeff=nnls(lower.T,solve_triangular(lower,target[indices],lower=True))[0]
    centers=coords[indices].copy()
    active=centers>0
    count=len(coeff)
    def objective(v):
        c=v[:count]
        centers[active]=v[count:]
        pair=centers[:,None,:]*centers[None,:,:]
        log_norm=(centers**2).sum(1)
        gram=np.exp(-.5*(log_norm[:,None]+log_norm)+np.log(np.cosh(pair)).sum(2))
        g=np.exp(-.5*(centers**2)@(1-np.tanh(s)))/np.sqrt(np.cosh(s)).prod()
        residual=gram@c-g
        dc=2*residual
        direction=(gram[:,:,None]*c[None,:,None]*(-centers[:,None,:]+centers[None,:,:]*np.tanh(pair))).sum(1)
        da=2*c[:,None]*(direction+g[:,None]*(1-np.tanh(s))*centers)
        return 1+c@residual-c@g,np.r_[dc,da[active]]
    fit=minimize(objective,np.r_[coeff,centers[active]],method='L-BFGS-B',jac=True,
        bounds=[(0.,3.)]*count+[(.01,5.)]*int(active.sum()),
        options=dict(ftol=1e-15,gtol=2e-10,maxiter=1200,maxcor=30))
    coeff=fit.x[:count];centers[active]=fit.x[count:]
    pos=[];weights=[]
    for a,c in zip(centers,coeff):
        if c<=0:continue
        signs=list(itertools.product(*[(v,-v) if v else (0.,) for v in a]))
        pos.extend(signs);weights.extend([c/len(signs)]*len(signs))
    pos=np.asarray(pos);weights=np.asarray(weights)
    gram=np.exp(-.5*((pos[:,None]-pos[None])**2).sum(2))
    weights/=np.sqrt(weights@gram@weights)
    overlap=np.exp(-.5*(pos**2)@(1-np.tanh(s)))/np.sqrt(np.cosh(s)).prod()
    fidelity=(weights@overlap)**2
    return pos,weights,dict(branches=len(weights),budget=budget,state_bound=math.sqrt(max(0.,1-fidelity)),
                            acceptance=float(1/weights.sum()**2))
