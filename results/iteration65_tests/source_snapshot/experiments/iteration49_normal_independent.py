"""Positive integral expansion and an independent pi identity."""
from fractions import Fraction as F
from functools import lru_cache
from iteration48_independent import bisect_sqrt


def atan_series(x):
    positive=sum(x**(4*k+1)/F(4*k+1) for k in range(200));negative=sum(x**(4*k+3)/F(4*k+3) for k in range(200));lo=positive-negative;return lo,lo+x**801/801


@lru_cache(None)
def pi_bounds():
    a,b=atan_series(F(1,2));c,d=atan_series(F(1,3));return 4*(a+c),4*(b+d)


def exp_negative(x):
    assert 0<=x<=8;term=total=F(1)
    for k in range(1,513):term*=x/k;total+=term
    tail=(term*x/513)/(1-x/F(514));return 1/(total+tail),1/total


@lru_cache(None)
def cdf(x):
    x=F(x);assert abs(x)<=4
    if x<0:
        a,b=cdf(-x);return 1-b,1-a
    if not x:return F(1,2),F(1,2)
    # I(x)=exp(-x^2/2) sum_{k>=0} x^(2k+1)/(2k+1)!!.
    term=total=x
    for k in range(1,301):term*=x*x/F(2*k+1);total+=term
    omitted=term*x*x/F(603);tail=omitted/(1-x*x/F(605));el,eh=exp_negative(x*x/2);pl,ph=pi_bounds();rl=bisect_sqrt(1/(2*ph))[0];rh=bisect_sqrt(1/(2*pl))[1];return F(1,2)+el*total*rl,F(1,2)+eh*(total+tail)*rh
