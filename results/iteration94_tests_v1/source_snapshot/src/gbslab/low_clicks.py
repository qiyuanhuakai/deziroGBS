"""Exact low-click events of specified states, evaluated in floating point.

All modes outside the listed click pattern are vacuum. Finite MPS events sum
every nonzero local occupation, not just the two-photon sector. No sampling,
conditional renormalization, probability floor, or experimental-state claim.
"""
import math
import numpy as np
from .banded_gaussian_mps import pairing_log_normalization
from .mps_overlap import normalized_right_mps


def _finish(patterns,probabilities,**info):
    p=np.asarray(probabilities,float); total=math.fsum(p)
    if not np.all(np.isfinite(p)) or np.any(p<0) or total>1+1e-10:
        raise ArithmeticError('Invalid low-click probabilities')
    return dict(patterns=np.asarray(patterns,int),probabilities=p,selected_mass=total,
        remainder_probability=max(0.,1-total),remainder_roundoff_clipped=total>1,
        arithmetic='Float64/complex128 diagnostic, not an interval certificate',**info)


def _square(z):
    result=abs(complex(z))**2
    if z!=0 and result==0: raise ArithmeticError('Positive squared amplitude underflowed')
    return result


def _probability(amplitudes):
    result=float(np.sum(np.abs(amplitudes)**2))
    if result==0 and np.any(np.asarray(amplitudes)!=0):
        raise ArithmeticError('Positive finite-MPS event mass underflowed')
    return result


def gaussian_low_clicks(pairing):
    """Vacuum, all exactly-one-click and exactly-two-click probabilities."""
    B=np.asarray(pairing,complex); logp0=2*pairing_log_normalization(B); m=len(B)
    p0=math.exp(logp0)
    if p0==0: raise ArithmeticError('Positive Gaussian vacuum mass underflowed')
    diagonal=np.diag(B); powers=np.array([_square(a) for a in diagonal]); logs=-.5*np.log1p(-powers)
    single_factors=np.expm1(logs); patterns=[(-1,-1)]; probs=[p0]
    for i in range(m):
        value=p0*single_factors[i]
        if diagonal[i]!=0 and value==0: raise ArithmeticError('Positive one-click mass underflowed')
        patterns.append((i,-1)); probs.append(value)
    for i in range(m):
        for j in range(i+1,m):
            a,b,c=diagonal[i],diagonal[j],B[i,j]; c2=_square(c)
            # det(I-B_SS B_SS^dagger)=product(single determinants)-delta.
            # Factor c^2 before evaluation so weak edges do not cancel against 1.
            phase=0j if c==0 else complex(c/abs(c))
            delta=c2*(2+2*(a*b*phase.conjugate()**2).real-c2)
            ratio=delta/((1-powers[i])*(1-powers[j]))
            if not 0<=ratio<1: raise ArithmeticError('Invalid pair determinant ratio')
            correction=-.5*math.log1p(-ratio)
            factor=single_factors[i]*single_factors[j]+math.exp(logs[i]+logs[j])*math.expm1(correction)
            value=p0*factor
            if (c!=0 or (a!=0 and b!=0)) and value==0: raise ArithmeticError('Positive two-click mass underflowed')
            patterns.append((i,j)); probs.append(value)
    return _finish(patterns,probs,modes=m,log_vacuum_probability=logp0)


def mps_low_clicks(tensors):
    """Normalize the whole input MPS, then sum all local occupied outcomes."""
    tensors,lognorm=normalized_right_mps(tensors); m=len(tensors)
    suffix=[None]*(m+1); suffix[m]=np.ones(1,complex)
    for i in range(m-1,-1,-1): suffix[i]=tensors[i][:,0,:]@suffix[i+1]
    patterns=[(-1,-1)]; probs=[_probability(suffix[0])]; prefix=np.ones(1,complex)
    singles=[]; pairs=[]
    for i,A in enumerate(tensors):
        work=np.einsum('a,anb->nb',prefix,A[:,1:,:])
        singles.append(_probability(work@suffix[i+1]))
        for j in range(i+1,m):
            T=tensors[j]; ends=np.einsum('anb,b->an',T[:,1:,:],suffix[j+1])
            pairs.append(((i,j),_probability(work@ends)))
            work=work@T[:,0,:]
        prefix=prefix@A[:,0,:]
    patterns.extend((i,-1) for i in range(m)); probs.extend(singles)
    patterns.extend(pair for pair,_ in pairs); probs.extend(value for _,value in pairs)
    return _finish(patterns,probs,modes=m,input_log_norm_squared=2*lognorm)


def low_click_tv(left,right):
    """TV on selected click patterns plus one remainder is a full-TV lower bound."""
    if not np.array_equal(left['patterns'],right['patterns']): raise ValueError('Mismatched click events')
    p=np.asarray(left['probabilities']); q=np.asarray(right['probabilities'])
    if p.shape!=q.shape: raise ValueError('Mismatched probability arrays')
    for value in (left,right):
        _finish(value['patterns'],value['probabilities'])
        if abs(math.fsum(value['probabilities'])-value['selected_mass'])>1e-13: raise ValueError('Inconsistent selected mass')
    return dict(threshold_coarse_tv_lower_estimate=.5*(math.fsum(np.abs(p-q))+abs(math.fsum(p)-math.fsum(q))),
        vacuum_probability_gap=float(abs(p[0]-q[0])),left_selected_mass=math.fsum(p),right_selected_mass=math.fsum(q),
        patterns=len(p),scope='Specified normalized zero-displacement states. Threshold-output lower bound evaluated in floating point; not an interval certificate or unknown experimental-mixture claim.')
