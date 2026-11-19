"""Exact bounded-mean confidence witnesses and integer-support count bounds.

For iid X_i in [0,1], E(prod X_i) = (E X)^n. Markov gives a
one-sided lower confidence bound (alpha * prod X_i) ** (1/n).
The two tails below each cost alpha; supplied samples do not prove iid.
This is a standard product/Markov witness, not a new statistical theorem.
"""
from fractions import Fraction as F
from math import prod


def dyadic_root_lower(value,degree,*,bits=40):
    """Floor of the nonnegative root on a 2**(-bits) grid, using integers."""
    value=F(value)
    if not 0<=value<=1 or type(degree) is not int or degree<1 or type(bits) is not int or not 8<=bits<=256:
        raise ValueError('Need value in [0,1], positive integer degree and 8..256 bits')
    if value in (0,1):return value
    scale=1<<bits;lo=0;hi=scale;right=value.numerator*scale**degree
    while hi-lo>1:
        mid=(lo+hi)//2
        if mid**degree*value.denominator<=right:lo=mid
        else:hi=mid
    return F(lo,scale)


def product_mean_interval(values,alpha,*,bits=40):
    """Two conservative mean limits; joint noncoverage at most 2*alpha under iid.

    The caller must predefine every value, including values assigned to failed
    sampling attempts. Never discard failures or select alpha after inspection.
    Independent inputs with unequal means do not give this iid mean statement.
    """
    values=tuple(map(F,values));alpha=F(alpha)
    if not values or any(not 0<=x<=1 for x in values) or not 0<alpha<1:
        raise ValueError('Nonempty bounded observations and alpha in (0,1) required')
    forward=prod(values,start=F(1));reverse=prod((1-x for x in values),start=F(1));n=len(values)
    lower=dyadic_root_lower(alpha*forward,n,bits=bits);upper=1-dyadic_root_lower(alpha*reverse,n,bits=bits)
    assert 0<=lower<=upper<=1
    return dict(lower=lower,upper=upper,forward_product=forward,reverse_product=reverse,attempts=n,alpha_per_tail=alpha,bits=bits)


def integer_count_tail_upper(threshold,modes,mean_interval,second_moment_interval):
    """Quadratic majorants on integer counts, conditional on valid moment bounds.

    For a=-1,...,K-2, (N-a)(N-a-1)/((K-a)(K-a-1)) majorizes
    1[N>=K] for all integer N. These are NOT moment-physicality certificates.
    """
    if type(threshold) is not int or type(modes) is not int or modes<1:raise ValueError('Integer threshold and positive mode count required')
    lo,hi=map(F,mean_interval);slo,shi=map(F,second_moment_interval)
    if not 0<=lo<=hi<=modes or not 0<=slo<=shi<=modes*modes:raise ValueError('Invalid moment bounds')
    if threshold<=0:return dict(upper=F(1),root=None,candidates=())
    if threshold>modes:return dict(upper=F(0),root=None,candidates=())
    candidates=[]
    for a in range(-1,threshold-1):
        coefficient=-(2*a+1);numerator=shi+coefficient*(hi if coefficient>=0 else lo)+a*(a+1);denominator=(threshold-a)*(threshold-a-1);bound=numerator/denominator
        if bound<0:raise ValueError('Moment intervals contradict a nonnegative integer polynomial')
        candidates.append(dict(root=a,upper=bound))
    best=min(candidates,key=lambda r:r['upper'])
    return dict(upper=min(F(1),best['upper']),root=best['root'] if best['upper']<1 else None,candidates=tuple(candidates))
