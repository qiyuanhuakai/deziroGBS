"""Explicit higher-degree Meixner threshold laws via shifted Gaussian Wick moments.

Small-system reference implementation: coefficient count C(m+D,D), output 2**m.
Not the paper's implicit overlap/rejection sampler. Signed laws and repair exposed.
"""
from functools import lru_cache
import math
from time import perf_counter
import numpy as np
from .meixner_probe import uniform_parameters


def occupations(modes,total):
    """Nonnegative multi-indices of one total degree, in deterministic order."""
    if modes==1:
        yield (total,); return
    for first in range(total+1):
        for rest in occupations(modes-1,total-first): yield (first,)+rest


class DiagonalNormalWick:
    """Normally ordered moments with <a_i† a_j>=n delta_ij, possibly signed n.

    Creation/annihilation counts are independent intermediate states. This is a
    formal Wick functional when the covariance is not a physical state.
    """
    def __init__(self,normal_mean,anomalous,*,max_states=1_000_000):
        M=np.asarray(anomalous,complex)
        if M.ndim!=2 or M.shape[0]!=M.shape[1] or not np.all(np.isfinite(M)) or not np.allclose(M,M.T,atol=1e-12,rtol=0):
            raise ValueError('Require finite symmetric anomalous matrix')
        if not math.isfinite(normal_mean) or not isinstance(max_states,int) or max_states<1: raise ValueError('Invalid mean or state cap')
        self.modes=len(M); self.states=0; self.max_states=max_states
        self.matrix=np.block([[M.conj(),normal_mean*np.eye(len(M))],[normal_mean*np.eye(len(M)),M]])
        @lru_cache(None)
        def wick(counts):
            self.states+=1
            if self.states>self.max_states: raise MemoryError('Wick state cap exceeded')
            if not any(counts): return 1.+0j
            i=next(k for k,n in enumerate(counts) if n)
            remaining=list(counts); remaining[i]-=1; value=0j
            for j,multiplicity in enumerate(remaining):
                covariance=self.matrix[i,j]
                if not multiplicity or covariance==0: continue
                remaining[j]-=1
                value+=multiplicity*covariance*wick(tuple(remaining))
                remaining[j]+=1
            return value
        self._wick=wick

    def binomial_moment(self,index):
        index=tuple(index)
        if len(index)!=self.modes or any(isinstance(x,bool) or not isinstance(x,(int,np.integer)) or x<0 for x in index):
            raise ValueError('Invalid photon multi-index')
        return self._wick(index+index)/math.prod(math.factorial(x) for x in index)

    def clear(self): self._wick.cache_clear()


def threshold_transform(subset_coefficients,b):
    """Evaluate sum_S A_S prod_{i in S} h1(bit_i), h1(0)=-b,h1(1)=1."""
    out=np.array(subset_coefficients,float,copy=True)
    if out.ndim!=1 or not len(out) or len(out)&(len(out)-1) or not np.all(np.isfinite(out)) or not math.isfinite(b) or b<=0:
        raise ValueError('Invalid subset table or reference mean')
    step=1
    while step<len(out):
        blocks=out.reshape(-1,2*step); low=blocks[:,:step].copy(); high=blocks[:,step:].copy()
        blocks[:,:step]=low-b*high; blocks[:,step:]=low+high; step*=2
    return out


def projection_prefixes(U,squeezing,transmissivity,max_degree,*,max_modes=16,max_coefficients=100_000,max_states=1_000_000):
    """All degrees 0..D, using c_k = Wick(nout-b,M;k)/prod(k_i!)."""
    start=perf_counter(); u=np.asarray(U,complex)
    if u.ndim!=2 or u.shape[0]!=u.shape[1] or not 2<=len(u)<=max_modes or not np.all(np.isfinite(u)) or not np.allclose(u.conj().T@u,np.eye(len(u)),atol=1e-11,rtol=0):
        raise ValueError('Require unitary U within mode cap')
    if isinstance(max_degree,bool) or not isinstance(max_degree,(int,np.integer)) or not 0<=max_degree<=8:
        raise ValueError('Degree must be in [0,8]')
    m=len(u); count=math.comb(m+max_degree,max_degree)
    if count>max_coefficients: raise MemoryError(f'Coefficient cap: {count}>{max_coefficients}')
    parameters=uniform_parameters(m,squeezing,transmissivity)
    b=parameters['b']; v=parameters['v']; n=parameters['nout']
    M=-transmissivity*math.sinh(squeezing)*math.cosh(squeezing)*(u@u.T)
    engine=DiagonalNormalWick(n-b,M,max_states=max_states)
    bits=((np.arange(1<<m)[:,None]>>np.arange(m))&1).astype(bool)
    base=np.prod(np.where(bits,b/(1+b),1/(1+b)),axis=1)
    cumulative=np.zeros(1<<m); indices=[]; coefficients=[]; rows=[]; laws=[]; max_imag=0.
    try:
        for degree in range(max_degree+1):
            mark=perf_counter(); contribution=np.zeros(1<<m); degree_count=0
            for index in occupations(m,degree):
                coefficient=engine.binomial_moment(index)
                max_imag=max(max_imag,abs(coefficient.imag))
                if abs(coefficient.imag)>1e-10*max(1,abs(coefficient.real)): raise ArithmeticError('Nonreal Wick moment')
                c=float(coefficient.real)
                mask=sum(1<<i for i,k in enumerate(index) if k); support=mask.bit_count()
                contribution[mask]+=c*(-b)**(degree-support)/v**degree
                indices.append(index); coefficients.append(c); degree_count+=1
            cumulative+=contribution
            raw=base*threshold_transform(cumulative,b)
            mass=float(math.fsum(raw)); negative=float(-raw[raw<0].sum())
            if not np.all(np.isfinite(raw)) or abs(mass-1)>1e-9: raise ArithmeticError(f'Signed mass residual: {mass-1}')
            repaired=np.maximum(raw,0); repaired/=repaired.sum()
            rows.append({'degree':degree,'coefficient_count':len(indices),'new_coefficients':degree_count,
                'wick_states':engine.states,'raw_mass':mass,'negative_mass':negative,
                'repair_l1':float(np.abs(repaired-raw).sum()),'degree_seconds':perf_counter()-mark,
                'prefix_seconds':perf_counter()-start})
            laws.append({'raw':raw,'distribution':repaired,'subset_coefficients':cumulative.copy()})
        return {'rows':rows,'laws':laws,'indices':np.asarray(indices,dtype=np.int16),
            'coefficients':np.asarray(coefficients),'parameters':parameters,'max_imaginary_moment':max_imag,
            'elapsed_seconds':perf_counter()-start,'wick_states':engine.states}
    finally: engine.clear()
