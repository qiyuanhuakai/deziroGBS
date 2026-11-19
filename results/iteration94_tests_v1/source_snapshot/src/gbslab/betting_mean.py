"""Exact fixed-stake mixture tests for a bounded iid mean.

For E[X]<=m and 0<t<=1, E[1-t+t*X/m]<=1. Products and a
predeclared convex mixture are e-values. Markov gives one-sided level alpha.
This is an implementation of standard betting/mixture reasoning, not a new
confidence theorem. The fixed finite stake grid need not yield consistent CIs.
"""
from collections import Counter
from fractions import Fraction as F
from math import lcm,prod,log,exp,floor,isfinite

DEFAULT_STAKES=tuple(F(1,1<<e) for e in range(6,-1,-1))


def _prepare(values,stakes):
    values=tuple(map(F,values));stakes=tuple(map(F,stakes))
    if not values or any(not 0<=x<=1 for x in values):raise ValueError('Nonempty observations in [0,1] required')
    if not stakes or any(not 0<t<=1 for t in stakes) or len(set(stakes))!=len(stakes):raise ValueError('Distinct fixed stakes in (0,1] required')
    denominator=lcm(*(x.denominator for x in values));base=lcm(*(t.denominator for t in stakes));counts=Counter(int(x*denominator) for x in values);numerators=tuple(int(t*base) for t in stakes)
    return values,stakes,denominator,base,counts,numerators


def mixture_ratio(values,mean,*,stakes=DEFAULT_STAKES):
    """Exact mixture numerator/denominator; mean must be positive."""
    values,stakes,D,B,counts,A=_prepare(values,stakes);m=F(mean)
    if not 0<m<=1:raise ValueError('Candidate mean in (0,1] required')
    word,scale=m.numerator,m.denominator;n=len(values)
    numerator=sum(prod((((B-a)*word*D+a*v*scale)**frequency for v,frequency in counts.items()),start=1) for a in A)
    denominator=len(A)*(B*word*D)**n
    return numerator,denominator


def _proposal(values,stakes,alpha):
    """Untrusted floating root proposal; exact endpoint checks are mandatory."""
    from scipy.optimize import brentq
    target=-log(float(alpha));counts=Counter(map(float,values));ts=tuple(map(float,stakes))
    def objective(m):
        if m<=0:return 1e300
        terms=[]
        for t in ts:
            factors=[(1-t+t*x/m,f) for x,f in counts.items()]
            terms.append(float('-inf') if any(a<=0 for a,f in factors) else sum(f*log(a) for a,f in factors))
        top=max(terms)
        if top==float('-inf'):return -1e300
        return top+log(sum(exp(x-top) for x in terms))-log(len(ts))-target
    return brentq(objective,0.,1.,xtol=1e-14,rtol=1e-14)


def lower_bound(values,alpha,*,stakes=DEFAULT_STAKES,bits=40,proposal=True):
    values,stakes,D,B,counts,A=_prepare(values,stakes);alpha=F(alpha)
    if not 0<alpha<1 or type(bits) is not int or not 8<=bits<=128:raise ValueError('Invalid error budget or grid')
    if not any(values):return F(0)
    scale=1<<bits;n=len(values)
    def rejected(word):
        if word==0:return True
        numerator=sum(prod((((B-a)*word*D+a*v*scale)**frequency for v,frequency in counts.items()),start=1) for a in A);denominator=len(A)*(B*word*D)**n
        return numerator*alpha.numerator>=denominator*alpha.denominator
    if proposal:
        try:x=float(_proposal(values,stakes,alpha))
        except (ValueError,OverflowError,ZeroDivisionError,RuntimeError):x=float('nan')
        if isfinite(x) and 0<=x<=1:
            candidate=min(scale-1,max(0,floor(x*scale)))
            if rejected(candidate) and not rejected(candidate+1):return F(candidate,scale)
    lo,hi=0,scale
    while hi-lo>1:
        mid=(lo+hi)//2
        if rejected(mid):lo=mid
        else:hi=mid
    return F(lo,scale)


def mean_interval(values,alpha,*,stakes=DEFAULT_STAKES,bits=40,proposal=True):
    values=tuple(map(F,values));stakes=tuple(map(F,stakes));lo=lower_bound(values,alpha,stakes=stakes,bits=bits,proposal=proposal);hi=1-lower_bound((1-x for x in values),alpha,stakes=stakes,bits=bits,proposal=proposal)
    assert 0<=lo<=sum(values)/len(values)<=hi<=1
    return dict(lower=lo,upper=hi,attempts=len(values),alpha_per_tail=F(alpha),stakes=stakes,weights=tuple(F(1,len(stakes)) for _ in stakes),bits=bits)
