"""Exact ideal local-Fock projection, independent of any MPS constructor.

Integer Hermite recurrence after clearing the rational pairing denominator.
Exponential validation only: at most eight modes and65536 occupations.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import lcm,prod,factorial
import numpy as np
from .gaussian_fock_bridge import positive_inverse,root_interval


def gaussian_box_reference(real,imag,dimensions):
    r=[[F(x) for x in row] for row in real];im=[[F(x) for x in row] for row in imag];n=len(r);dims=tuple(dimensions)
    if not 1<=n<=8 or len(im)!=n or any(len(row)!=n for row in r+im) or any(r[i][j]!=r[j][i] or im[i][j]!=im[j][i] for i in range(n) for j in range(n)):raise ValueError('Symmetric rational pairing with1..8 modes required')
    if len(dims)!=n or any(type(d) is not int or not 1<=d<=8 for d in dims) or prod(dims)>65536:raise ValueError('Fock box cap exceeded or invalid dimensions')
    S=np.block([[np.array(r,dtype=object),np.array(im,dtype=object)],[np.array(im,dtype=object),-np.array(r,dtype=object)]]);A=[[F(i==j)-S[i,j] for j in range(2*n)] for i in range(2*n)];inverse,det,pivots=positive_inverse(A);V=[[2*inverse[i][j]-int(i==j) for j in range(2*n)] for i in range(2*n)];D=lcm(*(x.denominator for row in r+im for x in row));C=[[(int(r[i][j]*D),int(im[i][j]*D)) for j in range(n)] for i in range(n)]
    @lru_cache(None)
    def coefficient(occ):
        if not any(occ):return 1,0
        if sum(occ)%2:return 0,0
        i=next(i for i,k in enumerate(occ) if k);left=list(occ);left[i]-=1;re=iv=0
        for j,m in enumerate(left):
            if not m:continue
            child=left.copy();child[j]-=1;a,b=coefficient(tuple(child));c,d=C[i][j];re+=m*(a*c-b*d);iv+=m*(a*d+b*c)
        return re,iv
    H=[];masses=[F(0)]*(1<<n);R=F(0)
    for occ in np.ndindex(*dims):
        a,b=coefficient(occ);H.append((a,b));w=F(a*a+b*b,D**sum(occ)*prod(factorial(k) for k in occ));R+=w;masses[sum(1<<j for j,k in enumerate(occ) if k)]+=w
    norm=root_interval(det);p=(norm[0]*R,min(F(1),norm[1]*R));assert 0<p[0]<=p[1]<=1
    law=[x/R for x in masses];distance=(root_interval(1-p[1])[0],root_interval(1-p[0])[1]);assert sum(law)==1
    return dict(dimensions=dims,occupations=prod(dims),pairing_denominator=D,integer_pairing=C,integer_hermite=H,unnormalized_threshold_masses=masses,unnormalized_norm=R,distribution=law,normalization_squared=det,normalization_interval=norm,projection_probability_interval=p,tail_trace_interval=distance,covariance=V,positive_pivots=pivots,scope='Ideal normalized physical local-box projection, no MPS bond truncation or construction rounding; exponential reference')
