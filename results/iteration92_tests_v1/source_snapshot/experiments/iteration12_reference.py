"""Small rectangular Gaussian coefficients for an exact infinite-state overlap.

The returned vector is NOT normalized: omitted Gaussian Fock occupations are
outside the box. Its inner product with a normalized box-supported MPS equals
the inner product with the full untruncated Gaussian state.
"""
from functools import lru_cache
import math
import numpy as np


def gaussian_box_coefficients(unitary, core_squeezing, dimension, max_elements=100_000):
    u=np.asarray(unitary,complex); m=len(u); d=dimension
    if u.shape!=(m,m) or not np.allclose(u.conj().T@u,np.eye(m),atol=1e-11,rtol=0):
        raise ValueError('Require unitary matrix')
    if not isinstance(d,int) or d<1 or d**m>max_elements:
        raise MemoryError('Gaussian coefficient box resource cap')
    B=-math.tanh(core_squeezing)*(u@u.T)
    @lru_cache(None)
    def coefficient(n):
        if not any(n): return complex(math.cosh(core_squeezing)**(-m/2))
        if sum(n)%2: return 0j
        i=next(j for j in range(m) if n[j])
        value=0j
        for j in range(m):
            multiplicity=n[j]-int(i==j)
            if multiplicity>0:
                previous=list(n); previous[i]-=1; previous[j]-=1
                value+=B[i,j]*math.sqrt(multiplicity)*coefficient(tuple(previous))
        return value/math.sqrt(n[i])
    x=np.empty((d,)*m,complex)
    for n in np.ndindex(x.shape): x[n]=coefficient(n)
    return x
