"""Exact dyadic squeezed thermal inputs, passive mixing and uniform loss."""
from fractions import Fraction as F
from math import lcm
import numpy as np
from gbslab.rational_gaussian import reciprocal_sqrt

def bareiss_determinant(matrix):
    values=[[F(x) for x in row] for row in matrix];n=len(values);den=lcm(*(x.denominator for row in values for x in row));a=[[x.numerator*(den//x.denominator) for x in row] for row in values];previous=1
    for k in range(n-1):
        pivot=a[k][k]
        if pivot<=0:raise ArithmeticError('Expected positive definite leading pivot')
        for i in range(k+1,n):
            for j in range(k+1,n):
                v=a[i][j]*pivot-a[i][k]*a[k][j];assert v%previous==0;a[i][j]=v//previous
            a[i][k]=0
        previous=pivot
    return F(a[-1][-1],den**n)

def physical_input(n,strength,transmission,order):
    H=np.array([[(-1)**((i&j).bit_count()) for j in range(n)] for i in range(n)],dtype=object);phases=([1,1j,-1,1] if n==4 else [1,1j,-1,1,1j,-1j,1,-1]);U=np.asarray(H,dtype=complex)@np.diag(phases)@np.asarray(H,dtype=complex)/n;A=np.array([[F(float(x)) for x in row] for row in U.real],dtype=object);B=np.array([[F(float(x)) for x in row] for row in U.imag],dtype=object);S=np.block([[A,-B],[B,A]])
    if strength=='moderate':q=[F(1,2),F(3,4),F(1,2),F(3,4)]*(n//4);p=[F(5,2),F(3,2),F(3),F(2)]*(n//4)
    elif strength=='strong':q=[F(1,4),F(1,2),F(1,4),F(1,2)]*(n//4);p=[F(5),F(3),F(6),F(5,2)]*(n//4)
    else:raise ValueError('Unknown fixed input strength')
    eta=F(transmission);D=np.diag(np.array(q+p,dtype=object));V=eta*(S@D@S.T)+(1-eta)*np.eye(2*n,dtype=object);ix=list(order)+[n+j for j in order];V=V[np.ix_(ix,ix)];w=S[:,q.index(min(q))][ix];nonclassical=sum(w[i]*(V[i,j]-int(i==j))*w[j] for i in range(2*n) for j in range(2*n));assert nonclassical<0 and all(F(float(x))==x for x in V.flat)
    return dict(covariance=V.tolist(),unitary_real=A.tolist(),unitary_imag=B.tolist(),input_q=q,input_p=p,transmission=eta,order=order,nonclassical_witness=w.tolist(),nonclassical_rayleigh=nonclassical)

def exact_target(covariance,bits=192):
    V=covariance;n=len(V)//2;vac=[(F(1),F(1))];determinants=[]
    for mask in range(1,1<<n):
        modes=[j for j in range(n) if mask>>j&1];ix=modes+[j+n for j in modes];Q=[[(V[i][j]+int(i==j))/2 for j in ix] for i in ix];det=bareiss_determinant(Q);assert det>=1;interval=reciprocal_sqrt((det,det),bits);vac.append(interval);determinants.append(det)
    raw=[];target=[]
    for clicks in range(1<<n):
        zeros=((1<<n)-1)^clicks;subset=clicks;lo=hi=F(0)
        while True:
            a,b=vac[zeros|subset]
            if subset.bit_count()%2:lo-=b;hi-=a
            else:lo+=a;hi+=b
            if not subset:break
            subset=(subset-1)&clicks
        raw.append((lo,hi));target.append((max(F(0),lo),min(F(1),hi)));assert target[-1][0]<=target[-1][1]
    mu=[(2*vac[1<<i][0]-1,2*vac[1<<i][1]-1) for i in range(n)]
    for j in range(n):
        for i in range(j):
            a,b=vac[(1<<i)|(1<<j)];c,d=vac[1<<i];e,f=vac[1<<j];mu.append((4*a-2*d-2*f+1,4*b-2*c-2*e+1))
    return dict(determinants=determinants,vacuum_intervals=vac,raw_intervals=raw,target_intervals=target,moment_intervals=mu,bits=bits)
