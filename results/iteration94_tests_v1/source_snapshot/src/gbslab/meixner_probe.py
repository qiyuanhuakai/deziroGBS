"""Degree <= 2 Meixner projection, analytically coarse-grained to threshold bits.

Special case: all inputs equally squeezed; uniform pure loss; zero mean. This
is a small full-law probe, not the randomized rejection sampler of
Angrisani and Mhiri, arXiv:2611.08444v1. Signed laws and repair are exposed.
"""
import math
import numpy as np


def uniform_parameters(modes, squeezing, transmissivity):
    if (isinstance(modes, bool) or not isinstance(modes, (int, np.integer)) or modes < 2
            or not np.isfinite(squeezing) or squeezing < 0
            or not np.isfinite(transmissivity) or not 0 < transmissivity < 1):
        raise ValueError("Require m>=2, r>=0 and 0<eta<1")
    nbar = math.sinh(squeezing)**2
    a = nbar + (1+nbar)/modes
    b = transmissivity*a
    nout = transmissivity*nbar
    return {'nbar':nbar, 'a':a, 'b':b, 'nout':nout, 'delta':b-nout,
        'v':b*(1+b), 'kappa':transmissivity**2*nbar*(1+nbar),
        'q':transmissivity*(1+a)/(1+transmissivity*a),
        'C':math.e*(1+(modes+1)*nbar)-1}


def signed_degree_budget(modes, squeezing, transmissivity, l1_error=.1, haar_failure=.05):
    """G.40 sufficient signed-law L1 degree, not end-to-end sampling degree."""
    if not 0 < l1_error < 1 or not 0 < haar_failure < 1:
        raise ValueError("Require errors in (0,1)")
    p = uniform_parameters(modes, squeezing, transmissivity)
    d = max(0, math.ceil(math.log(p['C']/(haar_failure*l1_error**2))/math.log(1/p['q']))-1)
    return dict(p, degree=d, l1_error=l1_error, haar_failure=haar_failure,
        naive_coefficient_count=str(math.comb(modes+d, d)),
        markov_bound=p['C']*p['q']**(d+1)/l1_error**2)


def threshold_projection(U, squeezing, transmissivity, degree, *, max_modes=16):
    """Sum the infinite PNR signed projection over each threshold cell exactly.

    Under geometric reference mean b, H_1=x-b and
    H_2=x(x-1)/2-2bx+b^2. Conditional expectations are (-b,1)
    and (b^2,-b) for no-click/click. Gaussian Wick moments fix coefficients.
    """
    u = np.asarray(U, dtype=complex)
    if (u.ndim != 2 or u.shape[0] != u.shape[1] or not 2 <= len(u) <= max_modes
            or not np.all(np.isfinite(u))
            or not np.allclose(u.conj().T@u, np.eye(len(u)), atol=1e-11, rtol=0)):
        raise ValueError("Require unitary U within mode cap")
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree not in (0,1,2):
        raise ValueError("Only degrees 0,1,2 are implemented")
    m = len(u)
    p = uniform_parameters(m, squeezing, transmissivity)
    b, delta, v = p['b'], p['delta'], p['v']
    bits = ((np.arange(1 << m)[:, None] >> np.arange(m)) & 1).astype(bool)
    base = np.prod(np.where(bits, b/(1+b), 1/(1+b)), axis=1)
    density = np.ones(1 << m)
    h1 = np.where(bits, 1., -b)
    if degree >= 1:
        density -= delta/v*h1.sum(axis=1)
    if degree >= 2:
        anomalous2 = p['kappa'] * np.abs(u@u.T)**2
        h2 = np.where(bits, -b, b*b)
        density += h2 @ (delta*delta + .5*np.diag(anomalous2))/v**2
        for i in range(m):
            for j in range(i+1, m):
                density += (delta*delta+anomalous2[i,j])/v**2*h1[:,i]*h1[:,j]
    raw = base*density
    if abs(raw.sum()-1) > 1e-10:
        raise ArithmeticError("Signed law mass is not one")
    negative = float(-raw[raw < 0].sum())
    repaired = np.maximum(raw, 0)
    repaired /= repaired.sum()
    return {'raw':raw, 'distribution':repaired, 'negative_mass':negative,
        'repair_l1':float(np.abs(repaired-raw).sum()), 'parameters':p,
        'raw_mass':float(raw.sum())}
