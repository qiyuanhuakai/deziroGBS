"""Exactly physical classical-noise Gaussian family and rational click laws."""
from fractions import Fraction as F

def coefficients(weights,intensity):
    w=tuple(map(F,weights));a=F(intensity)
    if not w or a<=0:raise ValueError('Nonempty positive-intensity family required')
    p=[1/(1+a*x*x) for x in w]
    return p,{(j,i):1/(1+a*(w[j]*w[j]+w[i]*w[i]))-p[j]*p[i] for i in range(len(w)) for j in range(i)}

def target_law(weights,intensity):
    w=tuple(map(F,weights));a=F(intensity);n=len(w);law=[]
    for mask in range(1<<n):
        zeros=sum((w[j]**2 for j in range(n) if not(mask>>j)&1),F(0));mass=F(0);sub=mask
        while True:
            mass+=(-1)**sub.bit_count()/(1+a*(zeros+sum((w[j]**2 for j in range(n) if (sub>>j)&1),F(0))))
            if sub==0:break
            sub=(sub-1)&mask
        law.append(mass)
    assert sum(law)==1 and min(law)>=0
    return law

def covariance(weights,intensity,total=10):
    w=tuple(map(F,weights))+(F(0),)*(total-len(weights));a=F(intensity)
    if len(w)!=total:raise ValueError('Invalid padding')
    return [[F(i==j)+(2*a*w[i%total]*w[j%total] if i//total==j//total else F(0)) for j in range(2*total)] for i in range(2*total)]

def half_l1(a,b):return sum((abs(x-y) for x,y in zip(a,b)),F(0))/2
