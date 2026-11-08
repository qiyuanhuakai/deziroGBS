"""Pure Gaussian resource search with independently checked primal/dual residuals.

Conventions: zero mean, hbar=2, xxpp. The weighted-pure oracle and dual follow
the minimum-trace covariance SDP (Oh et al.; Kalra & Kocharovsky 2606.29739).
All diagnostics are floating point, never interval or sampling-cost certificates.
The input and classical remainder are never projected or eigenvalue-clipped.
"""
import numpy as np
from scipy.optimize import minimize
from .exact import ThresholdModel


def omega(m):
    eye=np.eye(m); zero=np.zeros((m,m))
    return np.block([[zero,eye],[-eye,zero]])


def weighted_pure(weight):
    """Minimize tr(A P) over physical P: P=A^-1/2 |A^1/2 Omega A^1/2| A^-1/2."""
    A=np.asarray(weight,float)
    if A.ndim!=2 or A.shape[0]!=A.shape[1] or not len(A) or len(A)%2 or not np.all(np.isfinite(A)) or not np.allclose(A,A.T,atol=1e-12,rtol=0): raise ValueError('Invalid real symmetric weight')
    values,vectors=np.linalg.eigh(A)
    if values[0]<=0: raise ValueError('Weight must be positive definite')
    root=(vectors*np.sqrt(values))@vectors.T; inverse=(vectors/np.sqrt(values))@vectors.T
    skew=root@omega(len(A)//2)@root; skew=(skew-skew.T)/2
    square=-skew@skew; square=(square+square.T)/2
    squared,W=np.linalg.eigh(square)
    if squared[0]<=0: raise ArithmeticError('Unresolved positive symplectic frequency')
    absolute=(W*np.sqrt(squared))@W.T
    P=inverse@absolute@inverse; P=(P+P.T)/2
    return P,float(np.sqrt(squared).sum())


def resource_diagnostics(V,P,S):
    V=np.asarray(V,float); P=np.asarray(P,float); S=np.asarray(S,float); n=len(V); O=omega(n//2)
    A=np.eye(n)+S; oracle,value=weighted_pure(A); remainder=(V-P+V.T-P.T)/2
    primal=float(np.trace(P)); dual=value-float(np.sum(S*V.T)); gap=primal-dual
    return dict(primal_trace=primal,dual_trace=dual,primal_minus_dual=gap,
        pure_mean_photons=(primal-n)/4,dual_photon_lower_estimate=(dual-n)/4,
        minimum_remainder_eigenvalue=float(np.linalg.eigvalsh(remainder)[0]),
        minimum_dual_eigenvalue=float(np.linalg.eigvalsh((S+S.T)/2)[0]),
        minimum_pure_eigenvalue=float(np.linalg.eigvalsh(P)[0]),
        purity_residual=float(np.max(abs(P@O@P-O))),
        riccati_residual=float(np.max(abs(P@A@P-O.T@A@O))),
        oracle_residual=float(np.max(abs(P-oracle))),
        complementary_trace=float(np.sum(S*remainder.T)),
        scope='Float64 primal/dual diagnostic; neither interval certificate nor sampling-complexity lower bound')


def search_resource(covariance,*,max_iterations=2000,gradient_tolerance=1e-9,callback=None):
    """Factor S=L L.T to maintain dual PSD; verify actual final P independently.

This factorized search is not a globally convergent SDP solver. Its termination
flag alone is never acceptance. Full-rank initialization permits all directions;
the independent remainder, purity and dual-gap tests determine numerical quality.
"""
    V=np.asarray(covariance,float); ThresholdModel(V)
    if not np.array_equal(V,V.T): raise ValueError('Covariance must be exactly symmetric')
    if not isinstance(max_iterations,int) or max_iterations<1 or gradient_tolerance<=0: raise ValueError('Invalid search limits')
    n=len(V); values,vectors=np.linalg.eigh(V)
    if values[0]<=0: raise ValueError('Nonpositive covariance')
    # Only initializes a dual variable; never alters the input or its remainder.
    weights=np.maximum(1/values**2-1,1e-6); initial=vectors*np.sqrt(weights)
    evaluations=0; trace=[]; last={}
    def objective(flat):
        nonlocal evaluations,last
        L=flat.reshape(n,n); S=L@L.T; P,value=weighted_pure(np.eye(n)+S)
        negative_dual=float(np.sum(S*V.T))-value; grad=2*(V-P)@L
        evaluations+=1; last=dict(P=P,S=S,negative_dual=negative_dual,gradient_norm=float(np.linalg.norm(grad)))
        return negative_dual,grad.ravel()
    def record(flat):
        row=dict(iteration=len(trace)+1,evaluations=evaluations,dual_trace=-last['negative_dual'],factor_gradient_norm=last['gradient_norm'])
        trace.append(row)
        if callback is not None: callback(row)
    opt=minimize(objective,initial.ravel(),jac=True,method='L-BFGS-B',callback=record,
        options=dict(maxiter=max_iterations,gtol=gradient_tolerance,ftol=1e-15,maxls=40,maxcor=20))
    objective(opt.x); P=last['P']; S=last['S']; diagnostics=resource_diagnostics(V,P,S)
    return dict(pure_covariance=P,dual_multiplier=S,remainder=V-P,diagnostics=diagnostics,
        optimizer_success=bool(opt.success),optimizer_message=str(opt.message),iterations=int(opt.nit),evaluations=evaluations,history=trace)
