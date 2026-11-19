"""Independent polynomial coefficients and radical-group density overlap."""
from fractions import Fraction as F
from math import factorial,prod
import numpy as np
from iteration47_radicals import mul,conj,squarefree


def bisect_sqrt(x,bits=256):
    x=F(x);assert x>=0;scale=1<<bits;lo=0;hi=scale
    while hi*hi*x.denominator<x.numerator*scale*scale:hi*=2
    while lo<hi:
        mid=(lo+hi+1)//2
        if mid*mid*x.denominator<=x.numerator*scale*scale:lo=mid
        else:hi=mid-1
    return F(lo,scale),F(lo if lo*lo*x.denominator==x.numerator*scale*scale else lo+1,scale)


def polynomial_hermite(real,imag,dims):
    n=len(dims);zero=(0,)*n;p={zero:(F(1),F(0))}
    for i in range(n):
        for j in range(i,n):
            b=(real[i][j]/(2 if i==j else 1),imag[i][j]/(2 if i==j else 1));terms=[];power=(F(1),F(0));k=0
            while True:
                occ=[0]*n;occ[i]+=k;occ[j]+=k
                if any(x>=d for x,d in zip(occ,dims)):break
                terms.append((tuple(occ),(power[0]/factorial(k),power[1]/factorial(k))));power=mul(power,b);k+=1
            result={}
            for x,a in p.items():
                for y,b in terms:
                    key=tuple(v+w for v,w in zip(x,y))
                    if any(v>=d for v,d in zip(key,dims)):continue
                    c=mul(a,b);old=result.get(key,(F(0),F(0)));result[key]=(old[0]+c[0],old[1]+c[1])
            p=result
    return [(p.get(o,(F(0),F(0)))[0]*prod(factorial(k) for k in o),p.get(o,(F(0),F(0)))[1]*prod(factorial(k) for k in o)) for o in np.ndindex(*dims)]


def rational_amplitudes(tensors):
    arrays=[[(F(float(z.real)),F(float(z.imag))) for z in a.flat] for a in tensors];arrays=[np.array(a,dtype=object).reshape(t.shape+(2,)) for a,t in zip(arrays,tensors)];amps=[]
    for occ in np.ndindex(*(a.shape[1] for a in tensors)):
        vector=[(F(1),F(0))]
        for k,t in zip(occ,arrays):
            nxt=[]
            for right in range(t.shape[2]):
                terms=[mul(v,tuple(t[left,k,right])) for left,v in enumerate(vector)];nxt.append((sum(x for x,y in terms),sum(y for x,y in terms)))
            vector=nxt
        amps.append(vector[0])
    return amps


def independent_overlap(tensors,real,imag):
    dims=[a.shape[1] for a in tensors];H=polynomial_hermite(real,imag,dims);amps=rational_amplitudes(tensors);norm=sum(a*a+b*b for a,b in amps);groups={};R=F(0)
    for o,h,a in zip(np.ndindex(*dims),H,amps):
        fact=prod(factorial(k) for k in o);sq,rad=squarefree(fact);v=mul(conj(a),h);old=groups.get(rad,(F(0),F(0)));groups[rad]=(old[0]+v[0]/sq,old[1]+v[1]/sq);R+=(h[0]**2+h[1]**2)/fact
    density={};imaginary={}
    for ra,a in groups.items():
        for rb,b in groups.items():
            sq,rad=squarefree(ra*rb);v=mul(conj(a),b);density[rad]=density.get(rad,F(0))+v[0]/(sq*norm);imaginary[rad]=imaginary.get(rad,F(0))+v[1]/(sq*norm)
    assert all(x==0 for x in imaginary.values());lo=hi=F(0)
    for rad,c in density.items():
        a,b=bisect_sqrt(F(1,rad));lo+=c*(a if c>=0 else b);hi+=c*(b if c>=0 else a)
    return dict(hermite=H,amplitudes=amps,norm=norm,projection_unnormalized_norm=R,overlap_density_radicals=density,overlap_squared_over_norm_interval=(max(F(0),lo),hi))
