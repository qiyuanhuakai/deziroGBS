"""Independent exact density double sums grouped by square-free radicals."""
from fractions import Fraction as F
from functools import lru_cache
from math import factorial
import numpy as np

def mul(a,b):return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
def conj(a):return a[0],-a[1]

@lru_cache(None)
def squarefree(value):
    square=1;radical=1;prime=2
    while prime*prime<=value:
        power=0
        while value%prime==0:value//=prime;power+=1
        square*=prime**(power//2)
        if power%2:radical*=prime
        prime+=1
    return square,radical*value

@lru_cache(None)
def bisection_root(value,bits=256):
    scale=1<<bits;lo=0;hi=scale
    while lo<hi:
        mid=(lo+hi+1)//2
        if mid*mid*value<=scale*scale:lo=mid
        else:hi=mid-1
    return F(lo,scale),F(lo if lo*lo*value==scale*scale else lo+1,scale)

def density_radicals(tensors,alpha):
    n=len(tensors);dims=[a.shape[1] for a in tensors];converted=[]
    for a in tensors:converted.append((np.array([F(float(z.real)) for z in a.flat],dtype=object).reshape(a.shape),np.array([F(float(z.imag)) for z in a.flat],dtype=object).reshape(a.shape)))
    occupations=list(np.ndindex(*dims));amplitudes=[]
    for occ in occupations:
        r=np.ones(1,dtype=object);i=np.zeros(1,dtype=object)
        for k,(ar,ai) in zip(occ,converted):r,i=r@ar[:,k,:]-i@ai[:,k,:],r@ai[:,k,:]+i@ar[:,k,:]
        amplitudes.append((F(r[0]),F(i[0])))
    norm=sum(a*a+b*b for a,b in amplitudes);assert norm>0;powers=[]
    for z,d in zip(alpha,dims):
        values=[(F(1),F(0))];base=(-F(float(z.real)),-F(float(z.imag)))
        for k in range(1,d):values.append(mul(values[-1],base))
        powers.append(values)
    records=[]
    for mask in range(1<<n):
        groups={};imaginary={}
        for oa,aa in zip(occupations,amplitudes):
            for ob,ab in zip(occupations,amplitudes):
                if any(oa[j]!=ob[j] for j in range(n) if not mask>>j&1):continue
                value=mul(conj(aa),ab);denominator=1
                for j in range(n):
                    if mask>>j&1:value=mul(value,mul(powers[j][oa[j]],conj(powers[j][ob[j]])));denominator*=factorial(oa[j])*factorial(ob[j])
                sq,rad=squarefree(denominator);groups[rad]=groups.get(rad,F(0))+value[0]/(sq*norm);imaginary[rad]=imaginary.get(rad,F(0))+value[1]/(sq*norm)
        assert all(x==0 for x in imaginary.values());lo=hi=F(0)
        for rad,c in groups.items():
            a,b=bisection_root(rad);lo+=c*(a if c>=0 else b);hi+=c*(b if c>=0 else a)
        records.append(dict(mask=mask,radical_coefficients=groups,coefficient_interval=(max(F(0),lo),hi)))
    return records
