"""Exact small-box overlap certificates against an infinite pure Gaussian state.

B is supplied as two rational symmetric matrices. No covariance rounding,
input-tail truncation, or floating SVD error is omitted from the fidelity.
This is exponential validation, not a scalable Gaussian-to-MPS constructor.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import factorial, isqrt, prod
import numpy as np
from .exact_mps_law import integer_tensors
from .displaced_fock_interval import add, multiply, square, cmul, cadd, reciprocal_root_integer
from .displaced_fock_certificate import exp_minus_interval


def root_interval(x, bits=192):
    x=F(x)
    if x<0 or type(bits) is not int or not 32<=bits<=512:raise ValueError('Nonnegative rational and32..512 bits required')
    scale=1<<bits;k=isqrt(x.numerator*scale*scale//x.denominator)
    return F(k,scale),F(k if k*k*x.denominator==x.numerator*scale*scale else k+1,scale)


def positive_inverse(matrix):
    a=[[F(v) for v in row] for row in matrix];n=len(a)
    if not n or any(len(row)!=n for row in a) or any(a[i][j]!=a[j][i] for i in range(n) for j in range(n)):raise ValueError('Symmetric square matrix required')
    right=[[F(i==j) for j in range(n)] for i in range(n)];det=F(1);pivots=[]
    for k in range(n):
        p=a[k][k]
        if p<=0:raise ValueError('Positive definite matrix required')
        pivots.append(p);det*=p;a[k]=[v/p for v in a[k]];right[k]=[v/p for v in right[k]]
        for i in range(n):
            if i==k:continue
            q=a[i][k];a[i]=[x-q*y for x,y in zip(a[i],a[k])];right[i]=[x-q*y for x,y in zip(right[i],right[k])]
    return right,det,pivots


def physical_pairing(real,imag):
    r=[[F(v) for v in row] for row in real];i=[[F(v) for v in row] for row in imag];n=len(r)
    if not 1<=n<=4 or len(i)!=n or any(len(row)!=n for row in r+i) or any(r[a][b]!=r[b][a] or i[a][b]!=i[b][a] for a in range(n) for b in range(n)):raise ValueError('Symmetric rational pairing with1..4 modes required')
    S=np.block([[np.array(r,dtype=object),np.array(i,dtype=object)],[np.array(i,dtype=object),-np.array(r,dtype=object)]])
    A=[[F(a==b)-S[a,b] for b in range(2*n)] for a in range(2*n)];inverse,det,pivots=positive_inverse(A)
    V=[[2*inverse[a][b]-int(a==b) for b in range(2*n)] for a in range(2*n)]
    return dict(real=r,imag=i,determinant=det,pivots=pivots,covariance=V)


def cm(a,b):return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]


def hermite_box(real,imag,dimensions):
    dims=tuple(dimensions);n=len(real)
    if len(dims)!=n or any(type(d) is not int or not 1<=d<=8 for d in dims) or prod(dims)>4096:raise ValueError('Invalid small Fock box')
    @lru_cache(None)
    def h(occ):
        if not any(occ):return F(1),F(0)
        if sum(occ)%2:return F(0),F(0)
        k=next(j for j,x in enumerate(occ) if x);left=list(occ);left[k]-=1;re=im=F(0)
        for j,m in enumerate(left):
            if not m:continue
            child=left.copy();child[j]-=1;x,y=cm((real[k][j],imag[k][j]),h(tuple(child)));re+=m*x;im+=m*y
        return re,im
    occupations=list(np.ndindex(*dims));return occupations,[h(o) for o in occupations]


def gaussian_fidelity(tensors,real,imag,*,bits=192):
    physical=physical_pairing(real,imag);r=physical['real'];i=physical['imag'];dims=[int(a.shape[1]) for a in tensors]
    occupations,H=hermite_box(r,i,dims);integer=integer_tensors(tensors,max_states=4096);amps=[]
    for occ in occupations:
        ar=np.ones(1,dtype=object);ai=np.zeros(1,dtype=object)
        for k,(tr,ti,scale) in zip(occ,integer):ar,ai=ar@tr[:,k,:]-ai@ti[:,k,:],ar@ti[:,k,:]+ai@tr[:,k,:]
        amps.append((int(ar[0]),int(ai[0])))
    norm=sum(a*a+b*b for a,b in amps)
    if norm<=0:raise ValueError('Nonzero MPS required')
    R=F(0);overlap=((F(0),F(0)),(F(0),F(0)))
    for occ,(hr,hi),(ar,ai) in zip(occupations,H,amps):
        fact=prod(factorial(k) for k in occ);R+=(hr*hr+hi*hi)/fact;cr,ci=cm((ar,-ai),(hr,hi));root=reciprocal_root_integer(fact,bits);overlap=cadd(overlap,(multiply((cr,cr),root),multiply((ci,ci),root)))
    if R<=0:raise ArithmeticError('Projection must contain vacuum')
    squared=add(square(overlap[0]),square(overlap[1]));normalization=root_interval(physical['determinant'],bits)
    projection=tuple(x*R for x in normalization);projected_fidelity=tuple(x/(norm*R) for x in squared);full=multiply(tuple(x/norm for x in squared),normalization)
    clip=lambda v:(max(F(0),v[0]),min(F(1),v[1]))
    projection=clip(projection);projected_fidelity=clip(projected_fidelity);full=clip(full)
    if any(a>b for a,b in (projection,projected_fidelity,full)):raise ArithmeticError('Invalid fidelity enclosure')
    distance=lambda v:(root_interval(1-v[1],bits)[0],root_interval(1-v[0],bits)[1])
    return dict(physical=physical,dimensions=dims,occupations=occupations,hermite=H,projection_unnormalized_norm=R,integer_amplitudes=amps,integer_norm=norm,normalization_probability_interval=normalization,projection_mass_interval=projection,tail_mass_interval=(1-projection[1],1-projection[0]),overlap_squared_interval=squared,projected_fidelity_interval=projected_fidelity,full_fidelity_interval=full,tail_trace_interval=distance(projection),compression_trace_interval=distance(projected_fidelity),full_trace_interval=distance(full),bits=bits,scope='Exact infinite pure Gaussian B versus exactly normalized stored binary64 MPS; arbitrary common quantum channel contracts trace distance')


def gaussian_displaced_law(real,imag,alpha,*,bits=192):
    physical=physical_pairing(real,imag);V=physical['covariance'];n=len(real);alpha=np.asarray(alpha,complex)
    if alpha.shape!=(n,) or not np.all(np.isfinite(alpha)):raise ValueError('Finite displacement vector required')
    z=[F(float(a.real)) for a in alpha]+[F(float(a.imag)) for a in alpha];vac=[];records=[]
    for mask in range(1<<n):
        modes=[j for j in range(n) if mask>>j&1];ix=modes+[j+n for j in modes]
        if not ix:det=F(1);exponent=F(0)
        else:
            Q=[[(V[a][b]+int(a==b))/2 for b in ix] for a in ix];inv,det,pivots=positive_inverse(Q);exponent=sum(z[a]*inv[j][k]*z[b] for j,a in enumerate(ix) for k,b in enumerate(ix))
        factor=root_interval(1/det,bits);exponential=exp_minus_interval(exponent);value=multiply(factor,exponential);vac.append(value);records.append(dict(mask=mask,determinant=det,exponent=exponent,factor_interval=factor,exponential_interval=exponential))
    raw=[];law=[]
    for clicks in range(1<<n):
        zeros=((1<<n)-1)^clicks;subset=clicks;lo=hi=F(0)
        while True:
            a,b=vac[zeros|subset]
            if subset.bit_count()%2:lo-=b;hi-=a
            else:lo+=a;hi+=b
            if not subset:break
            subset=(subset-1)&clicks
        raw.append((lo,hi));law.append((max(F(0),lo),min(F(1),hi)))
    assert all(0<=a<=b<=1 for a,b in law) and sum(a for a,b in law)<=1<=sum(b for a,b in law)
    return dict(physical=physical,records=records,vacuum_intervals=vac,raw_intervals=raw,intervals=law,scope='Infinite pure Gaussian input and fixed displacement, threshold outputs untruncated')
