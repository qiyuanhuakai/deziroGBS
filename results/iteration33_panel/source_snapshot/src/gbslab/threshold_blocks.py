"""Deterministic finite-input block laws and explicit small Gaussian laws.

No clipping, local renormalization, or full-system error upper certificate.
"""
import math
import numpy as np
from .displaced_marginals import displacement_vacuum_effect,vacuum_expectation


def contiguous_density(coherent_mps,start,length,*,maximum_elements=20_000_000):
    tensors=coherent_mps.tensors
    if not isinstance(start,(int,np.integer)) or not isinstance(length,(int,np.integer)) or length<1 or start<0 or start+length>len(tensors):raise ValueError('Invalid contiguous block')
    dimensions=tuple(A.shape[1] for A in tensors[start:start+length]);physical=math.prod(dimensions)
    if physical>256:raise ValueError('Finite block dimension exceeds 256')
    left=np.ones((1,1),complex)
    for A in tensors[:start]:
        work=(left@A.reshape(A.shape[0],-1)).reshape(A.shape)
        left=A.reshape(-1,A.shape[2]).conj().T@work.reshape(-1,A.shape[2])
    joined=tensors[start]
    for A in tensors[start+1:start+length]:
        proposed=joined.shape[0]*joined.shape[1]*A.shape[1]*A.shape[2]
        if proposed>maximum_elements:raise MemoryError('Joined block exceeds explicit element cap')
        joined=(joined.reshape(-1,joined.shape[2])@A.reshape(A.shape[0],-1)).reshape(joined.shape[0],joined.shape[1]*A.shape[1],A.shape[2])
    work=(left@joined.reshape(joined.shape[0],-1)).reshape(joined.shape)
    rho=work.transpose(1,0,2).reshape(physical,-1)@joined.transpose(1,0,2).reshape(physical,-1).conj().T
    if not np.all(np.isfinite(rho)) or abs(np.trace(rho)-1)>1e-9 or np.max(abs(rho-rho.conj().T))>1e-9:raise ArithmeticError('Invalid finite block density')
    return rho,dimensions


def reduce_density(rho,dimensions,sites):
    dimensions=tuple(dimensions);sites=tuple(sites);size=math.prod(dimensions);rho=np.asarray(rho,complex)
    if rho.shape!=(size,size) or tuple(sorted(set(sites)))!=sites or any(i<0 or i>=len(dimensions) for i in sites):raise ValueError('Invalid density restriction')
    work=rho.reshape(dimensions+dimensions);remaining=len(dimensions)
    for mode in reversed(range(len(dimensions))):
        if mode not in sites:work=np.trace(work,axis1=mode,axis2=mode+remaining);remaining-=1
    d=math.prod(dimensions[i] for i in sites);return work.reshape(d,d)


def subset_effects(noise,dimensions):
    k=len(dimensions);noise=np.asarray(noise,float)
    if noise.ndim!=2 or noise.shape[0]!=2*k:raise ValueError('Noise/block dimension mismatch')
    effects={}
    for mask in range(1,1<<k):
        sites=[i for i in range(k) if mask&(1<<i)]
        effects[mask]=displacement_vacuum_effect(noise[sites+[i+k for i in sites]],tuple(dimensions[i] for i in sites))
    return effects


def threshold_from_vacua(vacua):
    """Little-endian click masks from all no-click subsets, no clipping."""
    q=np.asarray(vacua,float);count=len(q)
    if q.ndim!=1 or count<2 or count&(count-1) or not np.all(np.isfinite(q)) or q[0]!=1 or np.any((q<0)|(q>1)):raise ValueError('Invalid vacuum-subset array')
    probabilities=q[::-1].copy();k=count.bit_length()-1
    for bit in range(k):
        for mask in range(count):
            if mask&(1<<bit):probabilities[mask]-=probabilities[mask^(1<<bit)]
    if np.any(probabilities<0) or abs(math.fsum(probabilities)-1)>1e-11:raise ArithmeticError('Invalid threshold law; no clipping')
    return probabilities


def finite_block_law(rho,dimensions,effects):
    k=len(dimensions);vacua=np.ones(1<<k)
    for mask in range(1,1<<k):
        sites=[i for i in range(k) if mask&(1<<i)];vacua[mask]=vacuum_expectation(reduce_density(rho,dimensions,sites),effects[mask])
    return threshold_from_vacua(vacua),vacua


def gaussian_block_law(covariance):
    V=np.asarray(covariance,float);k=len(V)//2
    if V.shape!=(2*k,2*k) or k<1 or k>12 or not np.all(np.isfinite(V)) or not np.array_equal(V,V.T):raise ValueError('Invalid small Gaussian covariance')
    Q=(V+np.eye(2*k))/2;vacua=np.ones(1<<k)
    for mask in range(1,1<<k):
        sites=[i for i in range(k) if mask&(1<<i)];indices=sites+[i+k for i in sites];L=np.linalg.cholesky(Q[np.ix_(indices,indices)]);vacua[mask]=math.exp(-float(np.log(np.diag(L)).sum()))
    return threshold_from_vacua(vacua),vacua


def prefix_marginal(law,length):
    p=np.asarray(law,float);m=len(p).bit_length()-1
    if p.ndim!=1 or len(p)!=(1<<m) or not isinstance(length,(int,np.integer)) or not 1<=length<=m:raise ValueError('Invalid prefix length')
    return p.reshape(-1,1<<length).sum(axis=0)


def distance_diagnostics(target,proxy):
    p=np.asarray(target,float);q=np.asarray(proxy,float)
    if p.shape!=q.shape or np.any(p<0) or np.any(q<0) or abs(math.fsum(p)-1)>1e-10 or abs(math.fsum(q)-1)>1e-10:raise ValueError('Invalid probability laws')
    tv=.5*float(np.sum(abs(p-q)));root_sum=np.sqrt(p)+np.sqrt(q);active=root_sum>0
    # Rationalized root differences avoid subtracting nearly identical roots.
    squared_difference=math.fsum(((p[active]-q[active])/root_sum[active])**2)
    upper=.5*math.sqrt(squared_difference*math.fsum(root_sum**2))
    return dict(tv=tv,hellinger_squared=squared_difference,hellinger_cauchy_upper_for_this_block=upper,target_mass=math.fsum(p),proxy_mass=math.fsum(q))
