"""Certified continuous rank-one Gaussian displacement of finite multimode states.

alpha_j = direction_j X, X ~ N(0,sigma^2). Exact polynomial moments,
no quadrature, no displaced output-photon cutoff. Small-box reference only.
"""
from fractions import Fraction as F
from math import factorial,prod
import numpy as np
from .exact_mps_law import integer_tensors
from .extended_displaced_fock import add,multiply,cmul,cadd,reciprocal_root_integer
from .gaussian_fock_bridge import root_interval,physical_pairing,positive_inverse

def direction_values(direction,n):
    if len(direction)!=n:raise ValueError('One rational complex pair per mode required')
    z=[(F(a),F(b)) for a,b in direction]
    if not any(a or b for a,b in z):raise ValueError('Nonzero displacement direction required')
    return z

def interval_law(vac):
    size=len(vac);raw=[];law=[]
    for clicks in range(size):
        zero=(size-1)^clicks;subset=clicks;a=b=F(0)
        while True:
            l,h=vac[zero|subset]
            if subset.bit_count()%2:a-=h;b-=l
            else:a+=l;b+=h
            if not subset:break
            subset=(subset-1)&clicks
        raw.append((a,b));law.append((max(F(0),a),min(F(1),b)))
    if not all(0<=a<=b<=1 for a,b in law) or not sum(a for a,b in law)<=1<=sum(b for a,b in law):raise ArithmeticError('Invalid law enclosure')
    return raw,law

def integrated_finite_law(tensors,direction,sigma,*,bits=192):
    n=len(tensors);sigma=F(sigma)
    if not 1<=n<=4 or not 0<=sigma<=F(1,2) or any(a.ndim!=3 or not 1<=a.shape[1]<=8 for a in tensors):raise ValueError('Require1..4 modes, dimensions1..8, sigma0..1/2')
    z=direction_values(direction,n);dims=[int(a.shape[1]) for a in tensors];integer=integer_tensors(tensors,max_states=4096);occ=list(np.ndindex(*dims));amps=[]
    for o in occ:
        re=np.ones(1,dtype=object);im=np.zeros(1,dtype=object)
        for k,(ar,ai,s) in zip(o,integer):re,im=re@ar[:,k,:]-im@ai[:,k,:],re@ai[:,k,:]+im@ar[:,k,:]
        amps.append((int(re[0]),int(im[0])))
    norm=sum(a*a+b*b for a,b in amps)
    if not norm:raise ValueError('Nonzero finite state required')
    powers=[]
    for (a,b),d in zip(z,dims):
        p=[(F(1),F(0))]
        for k in range(1,d):x,y=p[-1];p.append((-a*x-b*y,b*x-a*y))
        powers.append(p)
    records=[];vac=[];zero=((F(0),F(0)),(F(0),F(0)))
    for mask in range(1<<n):
        A=1+2*sigma*sigma*sum(a*a+b*b for j,(a,b) in enumerate(z) if mask>>j&1);env={}
        for o,(ar,ai) in zip(occ,amps):
            value=((F(ar),F(ar)),(F(ai),F(ai)));degree=0;fact=1
            for j,k in enumerate(o):
                if mask>>j&1:
                    a,b=powers[j][k];value=cmul(value,((a,a),(b,b)));degree+=k;fact*=factorial(k)
            root=reciprocal_root_integer(fact,bits);value=(multiply(value[0],root),multiply(value[1],root));key=tuple(k for j,k in enumerate(o) if not mask>>j&1);poly=env.setdefault(key,{});poly[degree]=cadd(poly.get(degree,zero),value)
        coefficients={}
        for poly in env.values():
            for p,(ar,ai) in poly.items():
                for q,(br,bi) in poly.items():coefficients[p+q]=add(coefficients.get(p+q,(F(0),F(0))),add(multiply(ar,br),multiply(ai,bi)))
        moments={k:(F(0) if k%2 else F(prod(range(1,k,2)))*sigma**k/A**(k//2)) for k in coefficients};value=(F(0),F(0))
        for k,c in coefficients.items():value=add(value,multiply(c,(moments[k]/norm,moments[k]/norm)))
        factor=root_interval(1/A,bits);raw=multiply(value,factor);v=(max(F(0),raw[0]),min(F(1),raw[1]));vac.append(v);records.append(dict(mask=mask,tilt=A,polynomial_coefficients=coefficients,scaled_moments=moments,normalization_interval=factor,coefficient_interval=value,raw_vacuum_interval=raw))
    raw,law=interval_law(vac)
    return dict(direction=z,sigma=sigma,dimensions=dims,integer_amplitudes=amps,integer_norm=norm,records=records,vacuum_intervals=vac,raw_intervals=raw,intervals=law,scope='Exact continuous common-normal displacement of normalized stored finite input; complex directions allowed, output untruncated')

def gaussian_rank_one(real,imag,direction,sigma,*,bits=192):
    physical=physical_pairing(real,imag);n=len(real);z=direction_values(direction,n);sigma=F(sigma)
    if not 0<=sigma<=F(1,2):raise ValueError('sigma0..1/2 required')
    w=[a for a,b in z]+[b for a,b in z];V=physical['covariance'];noisy=[[V[i][j]+4*sigma*sigma*w[i]*w[j] for j in range(2*n)] for i in range(2*n)];u=[b for a,b in z]+[-a for a,b in z];variance=sum(u[i]*V[i][j]*u[j] for i in range(2*n) for j in range(2*n));records=[];vac=[]
    for mask in range(1<<n):
        modes=[j for j in range(n) if mask>>j&1];ix=modes+[j+n for j in modes]
        if ix:
            Q=[[(noisy[i][j]+int(i==j))/2 for j in ix] for i in ix];_,det,_=positive_inverse(Q);Q0=[[(V[i][j]+int(i==j))/2 for j in ix] for i in ix];inv,det0,_=positive_inverse(Q0);g=sum(w[i]*inv[a][b]*w[j] for a,i in enumerate(ix) for b,j in enumerate(ix))
        else:det=det0=F(1);g=F(0)
        value=root_interval(1/det,bits);vac.append(value);records.append(dict(mask=mask,determinant=det,pure_determinant=det0,displacement_exponent_coefficient=g,pure_factor_interval=root_interval(1/det0,bits)))
    raw,law=interval_law(vac)
    return dict(physical=physical,direction=z,sigma=sigma,covariance=noisy,generator_variance=variance,records=records,vacuum_intervals=vac,raw_intervals=raw,intervals=law)

def coupling_bound(target,grid):
    if target['sigma']!=grid['sigma']:raise ValueError('Noise scales must agree')
    distance=(1-grid['tail_interval'][0])*grid['interior_half_width']+grid['tail_mean_excess_interval'][1]
    return min(F(1),root_interval(target['generator_variance'])[1]*distance)
