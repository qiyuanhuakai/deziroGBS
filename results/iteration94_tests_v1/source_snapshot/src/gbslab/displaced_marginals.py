"""Exact finite-input local vacuum marginals after Gaussian displacement.

The finite Fock basis belongs to the input MPS. No post-displacement photon
cutoff is introduced. Small Gaussian Wick moments integrate the channel.
"""
import functools
import math
import numpy as np


def local_reduced_states(coherent_mps,adjacent_starts=()):
    """Single-site and selected adjacent-pair density matrices of a normalized MPS."""
    tensors=coherent_mps.tensors; m=len(tensors); starts=set(adjacent_starts)
    if any(not isinstance(i,(int,np.integer)) or not 0<=i<m-1 for i in starts): raise ValueError('Invalid adjacent pair')
    left=np.ones((1,1),complex); singles=[]; pairs={}
    def reduced(A):
        work=(left@A.reshape(A.shape[0],-1)).reshape(A.shape)
        density=np.einsum('anb,amb->nm',work,A.conj())
        if abs(np.trace(density)-1)>1e-9 or np.max(abs(density-density.conj().T))>1e-9: raise ArithmeticError('Unnormalized local reduced state')
        return density,work
    for i,A in enumerate(tensors):
        single,work=reduced(A); singles.append(single)
        if i in starts:
            following=tensors[i+1]
            joined=(A.reshape(-1,A.shape[2])@following.reshape(following.shape[0],-1)).reshape(A.shape[0],A.shape[1]*following.shape[1],following.shape[2])
            pairs[i]=reduced(joined)[0]
        left=A.reshape(-1,A.shape[2]).conj().T@work.reshape(-1,A.shape[2])
    return singles,pairs


def displacement_vacuum_effect(noise_factor,dimensions):
    """Finite-input block of E[D(-alpha)|0><0|D(alpha)], via Wick recurrence."""
    dimensions=tuple(dimensions); k=len(dimensions)
    if not k or any(not isinstance(d,(int,np.integer)) or d<1 for d in dimensions) or math.prod(dimensions)>256: raise ValueError('Invalid finite input dimensions')
    raw=np.asarray(noise_factor)
    if np.iscomplexobj(raw) and np.any(raw.imag!=0): raise ValueError('Noise factor must be real')
    F=np.asarray(raw.real,float)
    if F.ndim!=2 or F.shape[0]!=2*k or not np.all(np.isfinite(F)): raise ValueError('Invalid noise factor')
    C=F@F.T
    if np.any(F!=0) and not np.any(C!=0): raise ArithmeticError('Positive noise variance underflowed')
    K=np.eye(2*k)+C/2; cholesky=np.linalg.cholesky(K); log_prefactor=-float(np.log(np.diag(cholesky)).sum())
    prefactor=math.exp(log_prefactor)
    if prefactor==0: raise ArithmeticError('Positive local vacuum prefactor underflowed')
    tilted=C@np.linalg.solve(K,np.eye(2*k)); tilted=(tilted+tilted.T)/2
    H=np.column_stack([np.eye(k),1j*np.eye(k)])/2
    linear=np.vstack([H,H.conj()]); covariance=linear@tilted@linear.T
    @functools.lru_cache(None)
    def moment(powers):
        degree=sum(powers)
        if degree==0: return 1.+0j
        if degree%2: return 0j
        first=next(i for i,p in enumerate(powers) if p); rest=list(powers); rest[first]-=1; terms=[]
        for j,count in enumerate(rest):
            if count:
                child=rest.copy(); child[j]-=1; terms.append(count*covariance[first,j]*moment(tuple(child)))
        return complex(math.fsum(z.real for z in terms),math.fsum(z.imag for z in terms))
    occupations=list(np.ndindex(dimensions)); factors=[math.sqrt(math.prod(math.factorial(i) for i in n)) for n in occupations]
    effect=np.array([[prefactor*moment(a+b)/(factors[i]*factors[j]) for j,b in enumerate(occupations)] for i,a in enumerate(occupations)])
    if not np.all(np.isfinite(effect)) or np.max(abs(effect-effect.conj().T))>1e-10: raise ArithmeticError('Invalid averaged vacuum effect')
    return effect


def vacuum_expectation(density,effect):
    rho=np.asarray(density,complex); effect=np.asarray(effect,complex)
    if rho.shape!=effect.shape or rho.ndim!=2 or rho.shape[0]!=rho.shape[1]: raise ValueError('Mismatched density and effect')
    value=np.einsum('ij,ji->',rho,effect)
    if not np.isfinite(value) or abs(value.imag)>1e-10 or not 0<=value.real<=1+1e-10: raise ArithmeticError('Invalid averaged vacuum expectation; no clipping applied')
    return float(value.real)


def two_mode_threshold(first_vacuum,second_vacuum,both_vacuum):
    p=np.array([both_vacuum,second_vacuum-both_vacuum,first_vacuum-both_vacuum,1-first_vacuum-second_vacuum+both_vacuum])
    if not np.all(np.isfinite(p)) or np.any(p<0) or abs(p.sum()-1)>1e-12: raise ArithmeticError('Invalid two-mode threshold law; no clipping applied')
    return p
