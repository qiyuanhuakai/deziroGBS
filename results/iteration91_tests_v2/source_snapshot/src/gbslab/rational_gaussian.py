"""Exact integer residual enclosures for vacuum determinant functions.

The binary64 input is interpreted exactly, including (V+I)/2 without rounding.
This proves positive definiteness of Q, NOT quantum validity of V. Probability
interpretation requires a valid zero-mean covariance with hbar=2, xxpp order.
"""
from fractions import Fraction as F
from itertools import permutations,combinations
from math import isqrt,prod
import numpy as np


def outward(lo,hi,bits=128):
    lo,hi=F(lo),F(hi)
    if lo>hi or not isinstance(bits,int) or not 16<=bits<=512:raise ValueError('Invalid enclosure')
    scale=1<<bits
    return F((lo.numerator*scale)//lo.denominator,scale),F(-((-hi.numerator*scale)//hi.denominator),scale)


def reciprocal_sqrt(interval,bits=128):
    lo,hi=interval
    if not 0<lo<=hi:raise ArithmeticError('Unresolved positive determinant')
    scale=1<<bits
    def root_bounds(x):
        t=isqrt((x.numerator*scale*scale)//x.denominator)
        return F(t,scale),F(t if t*t*x.denominator==x.numerator*scale*scale else t+1,scale)
    return root_bounds(1/hi)[0],root_bounds(1/lo)[1]


def integer_matrix(values):
    a=np.asarray(values,dtype=object);fractions=[F(x) for x in a.flat];den=max(x.denominator for x in fractions)
    if den&(den-1):raise ValueError('Expected dyadic matrix')
    return np.array([x.numerator*(den//x.denominator) for x in fractions],dtype=object).reshape(a.shape),den


def interval_determinant(matrix):
    n=len(matrix);lo=F(0);hi=F(0)
    for perm in permutations(range(n)):
        a=b=F(1)
        for i,j in enumerate(perm):
            c,d=matrix[i][j];products=(a*c,a*d,b*c,b*d);a,b=min(products),max(products)
        if sum(perm[i]>perm[j] for i in range(n) for j in range(i+1,n))%2:lo-=b;hi-=a
        else:lo+=a;hi+=b
    return lo,hi


class VacuumEnclosure:
    def __init__(self,covariance,*,bits=128):
        raw=np.asarray(covariance)
        if np.iscomplexobj(raw) or raw.ndim!=2 or raw.shape[0]!=raw.shape[1] or not raw.shape[0] or raw.shape[0]%2 or raw.shape[0]>512:raise ValueError('Invalid real covariance shape')
        V=np.asarray(raw,dtype=float)
        if not np.all(np.isfinite(V)) or not np.array_equal(V,V.T):raise ValueError('Require finite exact symmetry')
        outward(0,1,bits);self.bits=bits;self.m=len(V)//2;n=len(V);Qexact=np.array([[(F(float(V[i,j]))+int(i==j))/2 for j in range(n)] for i in range(n)],dtype=object)
        # Floating linear algebra proposes a witness only; all verification below
        # uses exact integer products of its stored binary64 entries.
        L=np.linalg.cholesky((V+np.eye(n))/2);R=np.linalg.solve(L,np.eye(n));R=np.tril(R)
        if not np.all(np.isfinite(R)) or np.any(np.diag(R)<=0):raise ArithmeticError('Invalid triangular witness')
        A,denq=integer_matrix(Qexact);T,denr=integer_matrix(R);scale=denq*denr**2;B=T@A@T.T;residual=B-np.eye(n,dtype=object)*scale;maxrow=max(sum(abs(int(x)) for x in row) for row in residual);delta=F(maxrow,scale)
        if delta>=1:raise ArithmeticError('Whitening residual cannot certify positive definiteness')
        detR=prod((F(float(x)) for x in np.diag(R)),start=F(1));self.vacuum=outward(detR/(1+delta)**self.m,detR/(1-delta)**self.m,bits);self.R=R;self.delta=delta;self.inverse_center=T.T@T;self.inverse_denominator=denr**2;self.error_factor=delta/(1-delta);self._complements={():self.vacuum};self._determinants={}
        self.diagnostics=dict(whitening_residual_inf=delta,whitening_residual_float=float(delta),bits=bits,dimension=n,Q_positive_definite_proved=True,quantum_uncertainty_not_certified=True,integer_Q_denominator=denq,integer_R_denominator=denr)

    def complement(self,removed):
        removed=tuple(removed)
        if tuple(sorted(set(removed)))!=removed or len(removed)>2 or any(not isinstance(j,int) or j<0 or j>=self.m for j in removed):raise ValueError('Need zero, one, or two removed modes')
        if removed not in self._complements:
            ix=removed+tuple(j+self.m for j in removed);matrix=[];Y=self.inverse_center;den=self.inverse_denominator
            for i in ix:
                row=[]
                for j in ix:
                    center=F(int(Y[i,j]),den);error=self.error_factor*F(int(Y[i,i])+int(Y[j,j]),2*den);row.append((center-error,center+error))
                matrix.append(row)
            det=interval_determinant(matrix);factor=reciprocal_sqrt(det,self.bits);self._determinants[removed]=det;self._complements[removed]=outward(self.vacuum[0]*factor[0],self.vacuum[1]*factor[1],self.bits)
        return self._complements[removed]

    def atom(self,clicks):
        clicks=tuple(clicks);self.complement(clicks);lo=hi=F(0)
        for k in range(len(clicks)+1):
            for removed in combinations(clicks,k):
                a,b=self.complement(removed)
                if (len(clicks)-k)%2:lo-=b;hi-=a
                else:lo+=a;hi+=b
        # Do not silently project signs into a probability domain: return the
        # determinant-function interval even if quantum validity is unresolved.
        return outward(lo,hi,self.bits)
