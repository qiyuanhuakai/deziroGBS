"""Bridge rational moment completions to unknown exact moments in intervals.

Certifies existence of a nearby positive table at each enclosed true moment
vector. It does not claim the rational output exactly equals those moments.
"""
from fractions import Fraction as F
from .pairwise_completion import feature_masks,character,log_interval,sqrt_upper

def moment_bridge(law,moment_intervals,theta,n):
    masks=feature_masks(n);r=tuple(map(F,law));intervals=tuple((F(a),F(b)) for a,b in moment_intervals);theta=tuple(map(F,theta))
    if len(r)!=1<<n or len(intervals)!=len(masks) or len(theta)!=len(masks) or min(r)<=0 or sum(r)!=1 or any(a>b for a,b in intervals):raise ValueError('Positive normalized table and valid feature intervals required')
    mu=tuple(sum(p*character(x,m) for x,p in enumerate(r)) for m in masks)
    if any(not a<=x<=b for x,(a,b) in zip(mu,intervals)):raise ValueError('Rational moments must lie inside enclosures')
    radii=tuple(max(abs(a-x),abs(b-x)) for x,(a,b) in zip(mu,intervals));atom_radius=sum(radii)/(1<<n);tv_bridge=min(F(1),sum(radii)/2)
    if min(r)<=atom_radius:raise ArithmeticError('Moment uncertainty cannot certify positive feasible bridge')
    atom_intervals=tuple((p-atom_radius,p+atom_radius) for p in r);logs=tuple((log_interval(a)[0],log_interval(b)[1]) for a,b in atom_intervals);energies=tuple(sum(t*character(x,m) for t,m in zip(theta,masks)) for x in range(1<<n));residuals=tuple((a-e,b-e) for (a,b),e in zip(logs,energies));floor=min(a for a,b in residuals)
    # Each residual minus floor is nonnegative, so upper probability endpoints
    # bound its expectation for every exactly feasible bridged distribution.
    gap=sum(b*(hi-floor) for (a,b),(lo,hi) in zip(atom_intervals,residuals));assert gap>=0
    maxent_tv=min(F(1),tv_bridge+sqrt_upper(gap/2))
    return dict(rational_moments=mu,moment_intervals=intervals,moment_radii=radii,atom_radius=atom_radius,feasible_atom_intervals=atom_intervals,TV_to_true_feasible_upper=tv_bridge,log_intervals=logs,residual_floor=floor,entropy_gap_upper_for_true_feasible=gap,TV_to_true_maxent_upper=maxent_tv,theta=theta,scope='R is rational; a nearby positive law exists at every enclosed true moment vector. Maximum entropy is not the physical target.')

def physical_tv_interval(law,target):
    r=tuple(map(F,law));p=tuple((F(a),F(b)) for a,b in target)
    if len(r)!=len(p) or min(r)<0 or sum(r)!=1 or any(not 0<=a<=b<=1 for a,b in p):raise ValueError('Normalized law and probability enclosures required')
    return (sum(max(F(0),a-x,x-b) for x,(a,b) in zip(r,p))/2,min(F(1),sum(max(abs(x-a),abs(x-b)) for x,(a,b) in zip(r,p))/2))
