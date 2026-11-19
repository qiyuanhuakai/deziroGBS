"""Small finite-Fock displaced threshold certificates with factorial radicals."""
from fractions import Fraction as F
from math import factorial,isqrt,prod
import numpy as np
from .exact_mps_law import integer_tensors
from .displaced_fock_certificate import exp_minus_interval

def add(a,b):return a[0]+b[0],a[1]+b[1]
def neg(a):return -a[1],-a[0]
def multiply(a,b):
    values=[x*y for x in a for y in b];return min(values),max(values)
def square(a):return (F(0) if a[0]<=0<=a[1] else min(a[0]**2,a[1]**2),max(a[0]**2,a[1]**2))
def cmul(a,b):return add(multiply(a[0],b[0]),neg(multiply(a[1],b[1]))),add(multiply(a[0],b[1]),multiply(a[1],b[0]))
def cadd(a,b):return add(a[0],b[0]),add(a[1],b[1])

def reciprocal_root_integer(value,bits=192):
    if type(value) is not int or value<1 or type(bits) is not int or not 32<=bits<=512:raise ValueError('Positive integer and32..512 bits required')
    scale=1<<bits;low=isqrt(scale*scale//value);high=low if low*low*value==scale*scale else low+1;return F(low,scale),F(high,scale)

def displaced_finite_law(tensors,alpha,*,bits=192,max_states=4096):
    n=len(tensors);alpha=np.asarray(alpha,complex)
    if not 1<=n<=4 or alpha.shape!=(n,) or not np.all(np.isfinite(alpha)) or any(a.ndim!=3 or not 1<=a.shape[1]<=8 for a in tensors):raise ValueError('Require1..4 sites with local dimensions1..8')
    integer=integer_tensors(tensors,max_states=max_states);dims=[a.shape[1] for a in tensors];occupations=list(np.ndindex(*dims));amplitudes=[]
    for occupation in occupations:
        re=np.ones(1,dtype=object);im=np.zeros(1,dtype=object)
        for k,(ar,ai,scale) in zip(occupation,integer):re,im=re@ar[:,k,:]-im@ai[:,k,:],re@ai[:,k,:]+im@ar[:,k,:]
        amplitudes.append((int(re[0]),int(im[0])))
    norm=sum(a*a+b*b for a,b in amplitudes)
    if norm<=0:raise ArithmeticError('Zero represented state')
    z=[(F(float(a.real)),F(float(a.imag))) for a in alpha];roots={k:reciprocal_root_integer(factorial(k),bits) for k in range(max(dims))};weights=[]
    for (a,b),d in zip(z,dims):
        power=(F(1),F(0));site=[]
        for k in range(d):
            site.append((multiply((power[0],power[0]),roots[k]),multiply((power[1],power[1]),roots[k])));power=(-a*power[0]-b*power[1],b*power[0]-a*power[1])
        weights.append(site)
    vac=[];records=[]
    for mask in range(1<<n):
        exponent=sum(a*a+b*b for j,(a,b) in enumerate(z) if mask>>j&1);environment={}
        for occupation,(ar,ai) in zip(occupations,amplitudes):
            value=((F(ar),F(ar)),(F(ai),F(ai)))
            for j,k in enumerate(occupation):
                if mask>>j&1:value=cmul(value,weights[j][k])
            key=tuple(k for j,k in enumerate(occupation) if not mask>>j&1);environment[key]=cadd(environment.get(key,((F(0),F(0)),(F(0),F(0)))),value)
        c=(F(0),F(0))
        for a,b in environment.values():c=add(c,add(square(a),square(b)))
        coefficient=(c[0]/norm,c[1]/norm);exp=exp_minus_interval(exponent);raw=multiply(coefficient,exp);interval=(max(F(0),raw[0]),min(F(1),raw[1]));vac.append(interval);records.append(dict(mask=mask,coefficient_interval=coefficient,exponent=exponent,exponential_interval=exp,raw_vacuum_interval=raw))
    raw=[];law=[]
    for clicks in range(1<<n):
        zeros=((1<<n)-1)^clicks;subset=clicks;value=(F(0),F(0))
        while True:
            value=add(value,neg(vac[zeros|subset]) if subset.bit_count()%2 else vac[zeros|subset])
            if not subset:break
            subset=(subset-1)&clicks
        raw.append(value);law.append((max(F(0),value[0]),min(F(1),value[1])))
    assert all(0<=a<=b<=1 for a,b in law) and sum(a for a,b in law)<=1<=sum(b for a,b in law)
    return dict(modes=n,dimensions=dims,fock_states=prod(dims),displacements=z,radical_bits=bits,factorial_root_intervals=roots,integer_amplitudes=amplitudes,integer_norm=norm,vacuum_records=records,vacuum_intervals=vac,raw_intervals=raw,intervals=law,scope='Exact normalized finite input state and displaced threshold POVM; no output truncation, Gaussian tail or displacement integration claim')
