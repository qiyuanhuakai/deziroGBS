"""Exact integer binomial tails with an optional, untrusted quantile proposal."""
import math
from fractions import Fraction as F


def tail_ratio(n,k,p):
    if type(n) is not int or type(k) is not int or n<1 or not 0<=k<=n:raise ValueError('Need 0<=k<=n and n>=1')
    p=F(p)
    if not 0<=p<=1:raise ValueError('Probability outside [0,1]')
    a,b=p.numerator,p.denominator;c=b-a;den=b**n
    if k==0 or a==b:return den,den
    if a==0:return 0,den
    if k<=n//2:
        term=c**n;lower=term
        for j in range(k-1):
            numerator=term*(n-j)*a;divisor=(j+1)*c;term,remainder=divmod(numerator,divisor);assert remainder==0;lower+=term
        return den-lower,den
    term=a**n;upper=term
    for j in range(n,k,-1):
        numerator=term*j*c;divisor=(n-j+1)*a;term,remainder=divmod(numerator,divisor);assert remainder==0;upper+=term
    return upper,den


def lower_bound(n,k,alpha,*,bits=40,proposal=True):
    alpha=F(alpha)
    if not 0<alpha<1 or type(bits) is not int or not 8<=bits<=128:raise ValueError('Invalid confidence budget/grid')
    tail_ratio(n,k,0)
    if k==0:return F(0)
    scale=1<<bits
    def accepted(word):
        a,b=tail_ratio(n,k,F(word,scale));return a*alpha.denominator<=b*alpha.numerator
    if proposal:
        from scipy.stats import beta
        x=float(beta.ppf(float(alpha),k,n-k+1))
        if math.isfinite(x) and 0<=x<=1:
            candidate=min(scale-1,max(0,math.floor(x*scale)))
            if accepted(candidate) and not accepted(candidate+1):return F(candidate,scale)
    lo,hi=0,scale
    while hi-lo>1:
        mid=(lo+hi)//2
        if accepted(mid):lo=mid
        else:hi=mid
    return F(lo,scale)
