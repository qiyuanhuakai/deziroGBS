"""Zero-mean mixed Gaussian low-click events, hbar=2 and xxpp ordering.

One global inverse and O(m^2) constant-size blocks. Floating diagnostics, not
interval certificates. No covariance repair, clipping, or probability floors.
Near-zero conditional determinants in float64 require the explicit mpmath path.
The latter evaluates the supplied binary64 covariance, not an unknown exact state.
"""
import math
import numpy as np
from .exact import ThresholdModel
from .low_clicks import _finish


def _to_probability(value):
    result=float(value)
    if value>0 and result==0: raise ArithmeticError('Positive mixed-Gaussian event mass underflowed')
    if result<0 or not math.isfinite(result): raise ArithmeticError('Invalid mixed-Gaussian event mass')
    return result


def mixed_gaussian_low_clicks(covariance,*,backend='float64',decimal_digits=80):
    """Vacuum, exactly one click, and exactly two clicks on every mode pair.

Let Q=(V+I)/2, K=Q^-1, A_i=K[{i,i+m},{i,i+m}], F_i=det(A_i)^-1/2.
The event 'vacuum outside S' equals p0/sqrt(det K_SS), by Jacobi's identity.
For a pair, R=L_i^-1 K_ij L_j^-T and C=-log det(I-R.T R)/2 give
    p_ij=p0*((F_i-1)*(F_j-1)+F_i*F_j*expm1(C)).
This factors the positive correlation contribution instead of subtracting four
nearly equal vacuum probabilities. Exactly isolated vacuum modes are recognized
from input entries. Other near-zero float64 singles are rejected, not rounded.
"""
    if backend not in ('float64','mpmath'): raise ValueError('Unknown arithmetic backend')
    if not isinstance(decimal_digits,int) or decimal_digits<40: raise ValueError('Need at least 40 decimal digits')
    raw=np.asarray(covariance)
    if np.iscomplexobj(raw) and np.any(raw.imag!=0): raise ValueError('Covariance must be real')
    model=ThresholdModel(np.asarray(raw.real,float))
    # Require exact symmetry rather than silently alter the supplied matrix.
    V=np.asarray(raw.real,float)
    if not np.array_equal(V,V.T): raise ValueError('Covariance must be exactly symmetric')
    m=model.m; eye=np.eye(2*m); modes=[(i,i+m) for i in range(m)]
    isolated=[bool(np.array_equal(V[list(ix)],eye[list(ix)])) for ix in modes]
    if backend=='mpmath':
        return _multiprecision(V,modes,isolated,decimal_digits,model.minimum_uncertainty_eigenvalue)
    Q=(V+eye)/2; L=np.linalg.cholesky(Q); logp0=-float(np.log(np.diag(L)).sum()); p0=_to_probability(math.exp(logp0))
    if p0==0: raise ArithmeticError('Positive mixed-Gaussian vacuum mass underflowed')
    K=np.linalg.solve(Q,eye); K=(K+K.T)/2
    residual=float(np.linalg.norm(Q@K-eye,ord=np.inf)); condition=float(np.linalg.cond(Q))
    # This is an engineering refusal threshold, not a rigorous error enclosure.
    guard=64*np.finfo(float).eps*max(1.,condition,residual/np.finfo(float).eps)
    single=[]; logs=[]; factors=[]; roots=[]; minimum_delta=math.inf
    for i,ix in enumerate(modes):
        A=K[np.ix_(ix,ix)]; roots.append(np.linalg.cholesky(A))
        delta=math.fsum([1.,-float(A[0,0]*A[1,1]),float(A[0,1]*A[1,0])])
        if isolated[i]: delta=0.
        elif delta<=guard:
            raise ArithmeticError(f'Unresolved conditional determinant at mode {i}; use mpmath (delta={delta:g}, guard={guard:g})')
        if not 0<=delta<1: raise ArithmeticError('Invalid conditional vacuum determinant')
        minimum_delta=min(minimum_delta,delta); l=-.5*math.log1p(-delta); f=math.expm1(l)
        logs.append(l); factors.append(f); value=p0*f
        if f>0 and value==0: raise ArithmeticError('Positive one-click mass underflowed')
        single.append(value)
    pairs=[]
    for i in range(m):
        for j in range(i+1,m):
            C=K[np.ix_(modes[i],modes[j])]
            R=np.linalg.solve(roots[j],np.linalg.solve(roots[i],C).T).T
            s=np.linalg.svd(R,compute_uv=False); powers=s*s
            if np.any((s!=0)&(powers==0)): raise ArithmeticError('Positive pair-correlation square underflowed')
            if np.any(powers>=1): raise ArithmeticError('Invalid pair Schur complement')
            correction=-.5*float(np.log1p(-powers).sum())
            factor=factors[i]*factors[j]+math.exp(logs[i]+logs[j])*math.expm1(correction)
            value=p0*factor
            if ((factors[i]>0 and factors[j]>0) or np.any(C!=0)) and value==0:
                raise ArithmeticError('Positive two-click mass underflowed')
            pairs.append(value)
    patterns=[(-1,-1)]+[(i,-1) for i in range(m)]+[(i,j) for i in range(m) for j in range(i+1,m)]
    return _finish(patterns,[p0]+single+pairs,modes=m,backend=backend,log_vacuum_probability=logp0,
        minimum_uncertainty_eigenvalue=model.minimum_uncertainty_eigenvalue,inverse_residual_inf=residual,
        condition_number=condition,conditional_determinant_refusal_indicator=guard,
        minimum_conditional_delta=minimum_delta,isolated_vacuum_modes=[i for i,x in enumerate(isolated) if x],
        probability_corrections=0)


def _multiprecision(V,modes,isolated,digits,uncertainty):
    import mpmath as mp
    m=len(modes)
    with mp.workdps(digits):
        Q=mp.matrix([[ (mp.mpf(float(V[i,j]))+(1 if i==j else 0))/2 for j in range(2*m)] for i in range(2*m)])
        K=Q**-1; p0=1/mp.sqrt(mp.det(Q)); logp0=mp.log(p0)
        _to_probability(p0)
        inverses=[]; determinants=[]; factors=[]; single=[]
        for i,ix in enumerate(modes):
            A=mp.matrix([[K[a,b] for b in ix] for a in ix]); determinant=mp.det(A)
            delta=mp.mpf(0) if isolated[i] else 1-determinant
            if not 0<=delta<=1 or determinant<=0: raise ArithmeticError(f'Negative/invalid conditional mass at mode {i} in supplied covariance; no repair applied')
            l=-mp.log1p(-delta)/2 if delta<mp.mpf('.5') else -mp.log(determinant)/2
            f=mp.expm1(l)
            inverses.append(A**-1); determinants.append(determinant); factors.append(f); single.append(_to_probability(p0*f))
        pairs=[]
        for i in range(m):
            for j in range(i+1,m):
                C=mp.matrix([[K[a,b] for b in modes[j]] for a in modes[i]])
                X=inverses[j]*C.T*inverses[i]*C
                delta=X[0,0]+X[1,1]-mp.det(C)**2/(determinants[i]*determinants[j])
                if not 0<=delta<1: raise ArithmeticError('Invalid high-precision pair determinant ratio')
                correction=-mp.log1p(-delta)/2
                factor=factors[i]*factors[j]+(1+factors[i])*(1+factors[j])*mp.expm1(correction)
                pairs.append(_to_probability(p0*factor))
        patterns=[(-1,-1)]+[(i,-1) for i in range(m)]+[(i,j) for i in range(m) for j in range(i+1,m)]
        return _finish(patterns,[_to_probability(p0)]+single+pairs,modes=m,backend='mpmath',decimal_digits=digits,
            log_vacuum_probability=float(logp0),minimum_uncertainty_eigenvalue=uncertainty,
            isolated_vacuum_modes=[i for i,x in enumerate(isolated) if x],probability_corrections=0,
            input_semantics='Exact values of supplied binary64 entries; final probabilities converted to binary64; not interval certified')
