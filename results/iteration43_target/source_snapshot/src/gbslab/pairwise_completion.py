"""Exactly feasible small binary tables with certified max-entropy gap bounds.

Numerical optimization only proposes parameters. The returned rational table
is checked for positivity and exact first/pair moments; rational log intervals
bound its distance to the constrained maximum-entropy law, not the GBS target.
Enumeration is exponential and is not a scalable full-system sampler.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import isqrt

def feature_masks(n):
    if type(n) is not int or not 1<=n<=12:raise ValueError('Small table dimension1..12 required')
    return tuple([1<<j for j in range(n)]+[(1<<i)|(1<<j) for j in range(n) for i in range(j)])

def character(atom,mask):return -1 if (atom&mask).bit_count()%2 else 1

def exact_moment_projection(probabilities,moments,n):
    features=feature_masks(n);q=tuple(map(F,probabilities));mu=tuple(map(F,moments))
    if len(q)!=(1<<n) or len(mu)!=len(features) or min(q)<=0 or sum(q)<=0:raise ValueError('Positive small table and matching moment vector required')
    total=sum(q);q=tuple(x/total for x in q);observed=tuple(sum((x*character(a,m) for a,x in enumerate(q)),F(0)) for m in features);delta=tuple(x-y for x,y in zip(mu,observed));r=tuple(x+sum((d*character(a,m) for d,m in zip(delta,features)),F(0))/(1<<n) for a,x in enumerate(q))
    if min(r)<=0:raise ArithmeticError('Moment correction did not retain strict positivity')
    assert sum(r)==1 and all(sum((x*character(a,m) for a,x in enumerate(r)),F(0))==wanted for m,wanted in zip(features,mu))
    return dict(law=r,normalized_candidate=q,corrections=delta,features=features,moments=mu,minimum_probability=min(r))

@lru_cache(None)
def _unit_log(y,bits,terms):
    y=F(y)
    if not 1<=y<=2:raise ValueError('Reduced log argument outside[1,2]')
    if y==1:return F(0),F(0)
    scale=1<<bits;z=(y-1)/(y+1);scaled=z*scale;zl=scaled.numerator//scaled.denominator;zh=-((-scaled.numerator)//scaled.denominator)
    square=(zl*zl//scale,-((-zh*zh)//scale));power=(zl,zh);lower=upper=0
    for k in range(terms):
        divisor=2*k+1;lower+=2*(power[0]//divisor);upper+=2*(-((-power[1])//divisor));power=(power[0]*square[0]//scale,-((-power[1]*square[1])//scale))
    remainder=F(9,4*(2*terms+1)*3**(2*terms+1))*scale;upper+=-((-remainder.numerator)//remainder.denominator)
    return F(lower,scale),F(upper,scale)

def log_interval(value,*,bits=160,terms=80):
    x=F(value)
    if x<=0 or type(bits) is not int or bits<64 or type(terms) is not int or terms<16:raise ValueError('Positive argument and adequate integer precision required')
    exponent=x.numerator.bit_length()-x.denominator.bit_length();y=x/(F(2)**exponent)
    if y<1:exponent-=1;y*=2
    if y>2:exponent+=1;y/=2
    a,b=_unit_log(y,bits,terms);c,d=_unit_log(F(2),bits,terms)
    return (a+exponent*c,b+exponent*d) if exponent>=0 else (a+exponent*d,b+exponent*c)

def sqrt_upper(value,bits=80):
    x=F(value)
    if x<0:raise ValueError('Nonnegative radicand required')
    scale=1<<bits;v=x*scale*scale;word=isqrt(v.numerator//v.denominator)
    if word*word*v.denominator<v.numerator:word+=1
    return F(word,scale)

def entropy_certificate(law,moments,theta,n):
    r=tuple(map(F,law));mu=tuple(map(F,moments));theta=tuple(map(F,theta));masks=feature_masks(n)
    if len(r)!=(1<<n) or len(mu)!=len(theta) or len(mu)!=len(masks) or min(r)<=0 or sum(r)!=1:raise ValueError('Positive normalized feasible law required')
    if any(sum((p*character(a,m) for a,p in enumerate(r)),F(0))!=wanted for m,wanted in zip(masks,mu)):raise ValueError('Exact moments required')
    logs=tuple(log_interval(x) for x in r);energy=tuple(sum((t*character(a,m) for t,m in zip(theta,masks)),F(0)) for a in range(1<<n));residuals=tuple((a-e,b-e) for (a,b),e in zip(logs,energy));gap=sum((p*b for p,(a,b) in zip(r,residuals)),F(0))-min(a for a,b in residuals);assert gap>=0
    entropy=(-sum((p*b for p,(a,b) in zip(r,logs)),F(0)),-sum((p*a for p,(a,b) in zip(r,logs)),F(0)))
    return dict(log_intervals=logs,entropy_interval=entropy,entropy_gap_upper=gap,TV_to_exact_maximum_entropy_upper=min(F(1),sqrt_upper(gap/2)),theta=theta,scope='Maximum entropy under exact first/pair constraints; not a physical-target TV guarantee')

def fit_candidate(moments,n):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    masks=feature_masks(n);mu=np.asarray(list(map(float,moments)));X=np.asarray([[character(a,m) for m in masks] for a in range(1<<n)],dtype=float)
    def evaluate(theta):
        energy=X@theta;normalizer=logsumexp(energy);q=np.exp(energy-normalizer);gradient=X.T@q-mu
        return normalizer-theta@mu,gradient,q
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
    value,g,q=evaluate(theta)
    return dict(theta=[F(float(x)) for x in theta],probabilities=[F(float(x)) for x in q],numerical=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),gradient_max=float(np.max(np.abs(g))),dual_objective=float(value),refinement=refinement))
