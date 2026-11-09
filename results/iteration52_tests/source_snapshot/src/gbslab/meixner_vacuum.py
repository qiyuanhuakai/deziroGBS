"""Meixner threshold projections via subset vacuum Taylor series.

Exponential in modes (all subsets), polynomial in degree. Not a scalable
sampler. Mean-matched reference gives positive even vacuum coefficients.
"""
import math
from time import perf_counter
import numpy as np
from .meixner_probe import uniform_parameters


def vacuum_to_click(vacuum):
    """Batched inclusion-exclusion, last axis indexed by no-click subset."""
    out=np.array(vacuum,float,copy=True)
    size=out.shape[-1]
    if size<1 or size&(size-1): raise ValueError('Require power-of-two last axis')
    step=1
    while step<size:
        blocks=out.reshape(-1,size//(2*step),2*step)
        blocks[:,:,:step]-=blocks[:,:,step:]
        step*=2
    return out[...,::-1].copy()


def determinant_series(singular_values,normal_mean,reference_mean,max_degree):
    """Coefficients of det(I+x(C-bI)/(1+b))**(-1/2) on one subset."""
    s=np.asarray(singular_values,float); n=normal_mean; b=reference_mean
    coefficients=np.zeros(max_degree+1); coefficients[0]=1.
    if b==n:
        t=(s/(1+b))**2; powers=np.ones_like(t); log_derivative=[]
        for k in range(1,max_degree//2+1):
            powers*=t; log_derivative.append(.5*math.fsum(powers))
            coefficients[2*k]=np.dot(log_derivative,coefficients[2*(k-1)::-2])/k
    else:
        eigenvalues=np.concatenate(((n-b+s)/(1+b),(n-b-s)/(1+b)))
        powers=np.ones_like(eigenvalues); log_derivative=[]
        for k in range(1,max_degree+1):
            powers*=eigenvalues; log_derivative.append(.5*(-1)**k*math.fsum(powers))
            coefficients[k]=np.dot(log_derivative,coefficients[k-1::-1])/k
    return coefficients


def vacuum_projection_prefixes(U,squeezing,transmissivity,max_degree,*,reference_mean=None,max_modes=14):
    """Return all total-degree laws; default b is the physical mean n.

    Tail estimates use exact subset vacuum values in floating point. They
    are diagnostic bounds in ideal arithmetic, not numerical certificates.
    """
    start=perf_counter(); u=np.asarray(U,complex)
    if u.ndim!=2 or u.shape[0]!=u.shape[1] or not 2<=len(u)<=max_modes or not np.all(np.isfinite(u)) or not np.allclose(u.conj().T@u,np.eye(len(u)),atol=1e-11,rtol=0):
        raise ValueError('Require finite unitary within mode cap')
    if isinstance(max_degree,bool) or not isinstance(max_degree,(int,np.integer)) or not 0<=max_degree<=128: raise ValueError('Degree must be in [0,128]')
    pars=uniform_parameters(len(u),squeezing,transmissivity); n=pars['nout']
    b=n if reference_mean is None else float(reference_mean)
    if not math.isfinite(b) or b<0: raise ValueError('Require finite nonnegative reference mean')
    m=len(u); size=1<<m; M=-transmissivity*math.sinh(squeezing)*math.cosh(squeezing)*(u@u.T)
    vacuum_coefficients=np.zeros((max_degree+1,size)); exact_vacuum=np.ones(size); radii=np.zeros(size)
    cardinality=np.array([mask.bit_count() for mask in range(size)])
    vacuum_coefficients[0,0]=1.
    for mask in range(1,size):
        ids=[i for i in range(m) if (mask>>i)&1]
        s=np.linalg.svd(M[np.ix_(ids,ids)],compute_uv=False)
        radii[mask]=max(abs(n-b+s[0]),abs(n-b-s[0]))/(1+b)
        vacuum_coefficients[:,mask]=determinant_series(s,n,b,max_degree)/(1+b)**len(ids)
        exact_vacuum[mask]=math.exp(-len(ids)*math.log1p(n)-.5*math.fsum(np.log1p(-(s/(1+n))**2)))
    prefix=np.cumsum(vacuum_coefficients,axis=0); raw=vacuum_to_click(prefix)
    if not np.all(np.isfinite(raw)): raise ArithmeticError('Nonfinite signed law')
    repaired=np.maximum(raw,0); repaired/=repaired.sum(axis=1)[:,None]
    rows=[]
    for degree in range(max_degree+1):
        mass=math.fsum(raw[degree]); negative=-raw[degree,raw[degree]<0].sum()
        if abs(mass-1)>1e-8: raise ArithmeticError(f'Signed mass at D{degree}: {mass}')
        row=dict(degree=degree,raw_mass=mass,negative_mass=float(negative),repair_l1=float(np.abs(repaired[degree]-raw[degree]).sum()))
        if b==n:
            residual=exact_vacuum-prefix[degree]
            row['vacuum_residual_min']=float(residual.min())
            row['signed_half_l1_bound_estimate']=float(.5*np.dot(2.**cardinality,np.maximum(residual,0)))
            row['repaired_tv_bound_estimate']=min(1.,row['signed_half_l1_bound_estimate']+float(negative))
        rows.append(row)
    return dict(rows=rows,raw=raw,repaired=repaired,vacuum_coefficients=vacuum_coefficients,vacuum_prefix=prefix,
        exact_vacuum=exact_vacuum,subset_radii=radii,reference_mean=b,physical_mean=n,
        mean_matched=b==n,spectral_radius=float(radii.max()),elapsed_seconds=perf_counter()-start)
