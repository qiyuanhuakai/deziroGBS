"""Exact one-dimensional fiber preserving every proper marginal of a binary law.

This is a small-table diagnostic, not a scalable GBS sampler. A signed table
may have an empty nonnegative fiber. Target-aware median optimization is an
oracle benchmark and must not be described as a target-free algorithm.
"""
from fractions import Fraction as F
from itertools import combinations

def _table(values):
    q=tuple(map(F,values));size=len(q)
    if size<2 or size&(size-1) or sum(q)!=1:raise ValueError('Normalized power-of-two signed table required')
    return q,size.bit_length()-1

def shift(values,delta):
    q,n=_table(values);d=F(delta)
    return tuple(x+(-1)**i.bit_count()*d for i,x in enumerate(q))

def marginal(values,subset):
    q,n=_table(values);subset=tuple(subset)
    if len(set(subset))!=len(subset) or any(type(j) is not int or not 0<=j<n for j in subset):raise ValueError('Invalid marginal subset')
    out=[F(0)]*(1<<len(subset))
    for mask,x in enumerate(q):out[sum(((mask>>j)&1)<<k for k,j in enumerate(subset))]+=x
    return tuple(out)

def feasible_interval(values):
    q,n=_table(values);evens=[i for i in range(len(q)) if i.bit_count()%2==0];odds=[i for i in range(len(q)) if i.bit_count()%2]
    e=min(evens,key=lambda i:q[i]);o=min(odds,key=lambda i:q[i]);lo,hi=-q[e],q[o]
    return dict(lower=lo,upper=hi,feasible=lo<=hi,even_witness=e,odd_witness=o,pair_mass=q[e]+q[o])

def feasible_correction(values,desired_delta):
    interval=feasible_interval(values)
    if not interval['feasible']:return dict(interval=interval,delta=None,law=None)
    delta=max(interval['lower'],min(interval['upper'],F(desired_delta)));law=shift(values,delta);assert min(law)>=0 and sum(law)==1
    return dict(interval=interval,delta=delta,law=law)

def oracle_tv_projection(values,target):
    q,n=_table(values);p,m=_table(target)
    if m!=n or min(p)<0:raise ValueError('Matching physical probability table required')
    interval=feasible_interval(q)
    if not interval['feasible']:return dict(interval=interval,median_interval=None,delta=None,law=None,TV=None)
    residuals=sorted((-1)**i.bit_count()*(p[i]-q[i]) for i in range(len(q)));mid=len(q)//2;median=(residuals[mid-1],residuals[mid]);delta=max(interval['lower'],min(interval['upper'],median[0]));law=shift(q,delta);tv=sum((abs(x-y) for x,y in zip(law,p)),F(0))/2
    return dict(interval=interval,median_interval=median,delta=delta,law=law,TV=tv)

def first_negative_marginal(values):
    q,n=_table(values)
    for size in range(1,n):
        for subset in combinations(range(n),size):
            p=marginal(q,subset)
            for mask,x in enumerate(p):
                if x<0:return dict(subset=subset,atom=mask,mass=x,order=size)
    return None
