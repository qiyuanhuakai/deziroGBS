"""Small-table parity hierarchy with exact projection and true-moment bridge.

Generalizes the round43/44 certificate to declared nonempty features of degree
at most four. This remains exponential in the number of binary variables.
"""
from fractions import Fraction as F
from .pairwise_completion import character,log_interval,sqrt_upper

def feature_masks(n,order):
    if type(n) is not int or not 1<=n<=8 or type(order) is not int or not 1<=order<=min(4,n):raise ValueError('Require1..8 modes and degree1..min(4,n)')
    return tuple(sorted((m for m in range(1,1<<n) if m.bit_count()<=order),key=lambda m:(m.bit_count(),m)))

def moments_from_vacua(vacua,n,order):
    if len(vacua)!=1<<n:raise ValueError('Complete vacuum subset table required for this audit')
    intervals=[]
    for mask in feature_masks(n,order):
        subset=mask;lo=hi=F(0)
        while True:
            a,b=map(F,vacua[subset]);coefficient=(-1)**(mask.bit_count()-subset.bit_count())*(1<<subset.bit_count())
            if coefficient>=0:lo+=coefficient*a;hi+=coefficient*b
            else:lo+=coefficient*b;hi+=coefficient*a
            if subset==0:break
            subset=(subset-1)&mask
        intervals.append((lo,hi))
    return intervals

def fit_candidate(moments,n,order):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    masks=feature_masks(n,order);mu=np.asarray(list(map(float,moments)));X=np.asarray([[character(a,m) for m in masks] for a in range(1<<n)],dtype=float)
    if len(mu)!=len(masks):raise ValueError('Feature/moment mismatch')
    def evaluate(theta):
        energy=X@theta;normalizer=logsumexp(energy);q=np.exp(energy-normalizer);return normalizer-theta@mu,X.T@q-mu,q
    result=minimize(lambda t:evaluate(t)[:2],np.zeros(len(masks)),jac=True,method='BFGS',options={'gtol':1e-11,'maxiter':2000});theta=result.x.copy();refinement=[]
    for _ in range(16):
        value,g,q=evaluate(theta);error=float(np.max(np.abs(g)))
        if error<2e-14:break
        centered=X-(q@X);H=(centered.T*q)@centered
        try:step=np.linalg.solve(H,g)
        except np.linalg.LinAlgError:break
        accepted=False
        for scale in (1.,.5,.25,.125,.0625,.03125):
            candidate=theta-scale*step;v,h,p=evaluate(candidate)
            if np.max(np.abs(h))<error and v<=value+1e-12:theta=candidate;accepted=True;break
        refinement.append(dict(error=error,accepted=accepted))
        if not accepted:break
    value,g,q=evaluate(theta);return dict(theta=[F(float(x)) for x in theta],probabilities=[F(float(x)) for x in q],numerical=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),gradient_max=float(np.max(np.abs(g))),dual_objective=float(value),refinement=refinement))

def project(probabilities,moments,n,order):
    masks=feature_masks(n,order);q=tuple(map(F,probabilities));mu=tuple(map(F,moments))
    if len(q)!=1<<n or len(mu)!=len(masks) or min(q)<=0:raise ValueError('Positive matching table required')
    q=tuple(x/sum(q) for x in q);delta=tuple(w-sum(x*character(a,m) for a,x in enumerate(q)) for m,w in zip(masks,mu));r=tuple(x+sum(d*character(a,m) for d,m in zip(delta,masks))/(1<<n) for a,x in enumerate(q))
    if min(r)<=0:raise ArithmeticError('Declared-moment projection not strictly positive')
    assert sum(r)==1 and all(sum(x*character(a,m) for a,x in enumerate(r))==w for m,w in zip(masks,mu))
    return dict(law=r,normalized_candidate=q,corrections=delta,features=masks,moments=mu,minimum_probability=min(r))

def bridge(law,intervals,theta,n,order):
    masks=feature_masks(n,order);r=tuple(map(F,law));intervals=tuple((F(a),F(b)) for a,b in intervals);theta=tuple(map(F,theta))
    if len(r)!=1<<n or len(intervals)!=len(masks) or len(theta)!=len(masks) or sum(r)!=1 or min(r)<=0 or any(a>b for a,b in intervals):raise ValueError('Matching positive law and feature intervals required')
    mu=tuple(sum(p*character(x,m) for x,p in enumerate(r)) for m in masks)
    if any(not a<=x<=b for x,(a,b) in zip(mu,intervals)):raise ValueError('Rational moments outside interval')
    radii=tuple(max(abs(a-x),abs(b-x)) for x,(a,b) in zip(mu,intervals));rho=sum(radii)/(1<<n);delta=min(F(1),sum(radii)/2)
    if min(r)<=rho:raise ArithmeticError('True-moment positive bridge not certified')
    atoms=tuple((x-rho,x+rho) for x in r);logs=tuple((log_interval(a)[0],log_interval(b)[1]) for a,b in atoms);energy=tuple(sum(t*character(x,m) for t,m in zip(theta,masks)) for x in range(1<<n));residual=tuple((a-e,b-e) for (a,b),e in zip(logs,energy));floor=min(a for a,b in residual);gap=sum(b*(h-floor) for (a,b),(l,h) in zip(atoms,residual))
    return dict(features=masks,rational_moments=mu,moment_intervals=intervals,moment_radii=radii,atom_radius=rho,feasible_atom_intervals=atoms,TV_to_true_feasible_upper=delta,log_intervals=logs,residual_floor=floor,entropy_gap_upper_for_true_feasible=gap,TV_to_true_maxent_upper=min(F(1),delta+sqrt_upper(gap/2)),theta=theta,scope='Maximum entropy under the declared true feature moments, not physical-target accuracy')
