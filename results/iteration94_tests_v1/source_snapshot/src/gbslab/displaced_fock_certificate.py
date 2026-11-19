"""Rigorous displaced threshold law for small stored dimension-two MPS.

Certifies the normalized finite input state and exact displacement measurement,
including all output photons. No Gaussian target or displacement average claim.
"""
from fractions import Fraction as F
import numpy as np
from .exact_mps_law import integer_tensors

def exp_minus_interval(x,terms=160):
    x=F(x)
    if not 0<=x<=4 or type(terms) is not int or terms<16 or terms%2:raise ValueError('Require x in [0,4] and even terms >=16')
    term=F(1);total=F(1)
    for k in range(1,terms+1):term*=-x/k;total+=term
    lower=total+term*(-x)/(terms+1)
    return max(F(0),lower),min(F(1),total)

def mul(a,b):return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])

def displaced_dimension_two_law(tensors,alpha):
    n=len(tensors);alpha=np.asarray(alpha,complex)
    if not 1<=n<=4 or alpha.shape!=(n,) or not np.all(np.isfinite(alpha)) or any(a.shape[1]!=2 for a in tensors):raise ValueError('Require1..4 finite dimension-two sites and one displacement each')
    integer=integer_tensors(tensors,max_states=16);amplitudes=[]
    for mask in range(1<<n):
        re=np.ones(1,dtype=object);im=np.zeros(1,dtype=object)
        for j,(ar,ai,scale) in enumerate(integer):
            k=(mask>>j)&1;re,im=re@ar[:,k,:]-im@ai[:,k,:],re@ai[:,k,:]+im@ar[:,k,:]
        amplitudes.append((int(re[0]),int(im[0])))
    norm=sum(a*a+b*b for a,b in amplitudes)
    if norm<=0:raise ArithmeticError('Zero represented state')
    z=[(F(float(a.real)),F(float(a.imag))) for a in alpha];vac=[];records=[]
    for mask in range(1<<n):
        exponent=sum(a*a+b*b for j,(a,b) in enumerate(z) if mask>>j&1);environment={}
        for occupation,amplitude in enumerate(amplitudes):
            value=amplitude
            for j,(a,b) in enumerate(z):
                if mask>>j&1 and occupation>>j&1:value=mul(value,(-a,b))
            key=occupation&~mask;old=environment.get(key,(F(0),F(0)));environment[key]=(old[0]+value[0],old[1]+value[1])
        coefficient=F(sum(a*a+b*b for a,b in environment.values()),norm);lo,hi=exp_minus_interval(exponent);p=(coefficient*lo,coefficient*hi);vac.append(p);records.append(dict(mask=mask,coefficient=coefficient,exponent=exponent,exponential_interval=(lo,hi)))
    raw=[];law=[]
    for clicks in range(1<<n):
        zeros=((1<<n)-1)^clicks;lo=hi=F(0);subset=clicks
        while True:
            a,b=vac[zeros|subset]
            if subset.bit_count()%2:lo-=b;hi-=a
            else:lo+=a;hi+=b
            if not subset:break
            subset=(subset-1)&clicks
        raw.append((lo,hi));law.append((max(F(0),lo),min(F(1),hi)))
    assert all(0<=a<=b<=1 for a,b in law) and sum(a for a,b in law)<=1<=sum(b for a,b in law)
    return dict(modes=n,displacements=z,integer_amplitudes=amplitudes,integer_norm=norm,vacuum_records=records,vacuum_intervals=vac,raw_intervals=raw,intervals=law,scope='Normalized stored dimension-two state, exact displaced threshold effects, no output-photon truncation; not an untruncated Gaussian target')
