"""Exact-model marginal enclosures and fixed-event sampling confidence bounds.

The Gaussian model is physically certified by ExactPairingModel. All numerical
enclosures below are rational. Binomial coverage additionally assumes independent
identically distributed attempts; a fixed PRNG seed is not a proof of that premise.
"""
from fractions import Fraction as F
from math import comb
import numpy as np
from .certified_gaussian import ExactPairingModel,HermitianCertificate,CertificationFailure
from .rational_gaussian import outward,reciprocal_sqrt


class GaussianEnvelope:
    def __init__(self,model,*,bits=128,output_bits=256):
        if not isinstance(model,ExactPairingModel):raise TypeError('A certified exact B,F model is required')
        self.model=model;self.m=model.m;self.bits=bits;self.output_bits=output_bits;self.vacua={(): (F(1),F(1))};self.certificates={}

    def vacuum(self,modes):
        modes=tuple(modes)
        if tuple(sorted(set(modes)))!=modes or any(not isinstance(j,(int,np.integer)) or not 0<=j<self.m for j in modes):raise ValueError('Sorted distinct model modes required')
        if modes not in self.vacua:
            ix=modes+tuple(j+self.m for j in modes);C=self.model.center_numerators[np.ix_(ix,ix)];den=self.model.center_denominator;n=len(ix);Q=np.array([[F(int(C[i,j])+int(i==j)*den,2*den) for j in range(n)] for i in range(n)],dtype=object);cert=HermitianCertificate(Q,bits=self.bits);eta=self.model.operator_radius/(2*cert.minimum_eigenvalue_lower)
            if eta>=1:raise CertificationFailure('Local relative perturbation is unresolved')
            a,b=cert.determinant_interval(self.output_bits);det=outward(a*(1-eta)**n,b*(1+eta)**n,self.output_bits);lo,hi=reciprocal_sqrt(det,self.output_bits);self.vacua[modes]=(max(F(0),lo),min(F(1),hi));self.certificates[modes]=dict(certificate=cert,relative_perturbation=eta,perturbed_determinant_interval=det,raw_vacuum_interval=(lo,hi))
        return self.vacua[modes]

    def law(self,modes):
        modes=tuple(modes)
        if not 1<=len(modes)<=12:raise ValueError('Complete laws limited to 1..12 modes')
        self.vacuum(modes);n=len(modes);vac=[self.vacuum(tuple(modes[j] for j in range(n) if mask>>j&1)) for mask in range(1<<n)];raw=[];intervals=[]
        for clicks in range(1<<n):
            zeros=((1<<n)-1)^clicks;subset=clicks;lo=hi=F(0)
            while True:
                a,b=vac[zeros|subset]
                if subset.bit_count()%2:lo-=b;hi-=a
                else:lo+=a;hi+=b
                if not subset:break
                subset=(subset-1)&clicks
            raw.append((lo,hi));intervals.append((max(F(0),lo),min(F(1),hi)))
            if intervals[-1][0]>intervals[-1][1]:raise ArithmeticError('Inconsistent physical atom interval')
        return dict(modes=modes,raw_intervals=raw,intervals=intervals)

    def click_moments(self):
        singles=[self.vacuum((j,)) for j in range(self.m)];mean=(sum(1-b for a,b in singles),sum(1-a for a,b in singles));e2lo,e2hi=mean;pairs=[]
        for i in range(self.m):
            for j in range(i+1,self.m):
                a,b=self.vacuum((i,j));lo=max(F(0),1-singles[i][1]-singles[j][1]+a);hi=min(F(1),1-singles[i][0]-singles[j][0]+b);e2lo+=2*lo;e2hi+=2*hi;pairs.append(dict(modes=(i,j),click_pair_interval=(lo,hi)))
        return dict(mean_interval=mean,second_moment_interval=(e2lo,e2hi),variance_interval=(max(F(0),e2lo-mean[1]**2),max(F(0),e2hi-mean[0]**2)),pairs=pairs)


def binomial_tail(n,k,p):
    if not isinstance(n,int) or not isinstance(k,int) or not 0<=k<=n or n<1:raise ValueError('Need 0 <= successes <= attempts')
    p=F(p)
    if not 0<=p<=1:raise ValueError('Probability outside [0,1]')
    a,b=p.numerator,p.denominator;return F(sum(comb(n,j)*a**j*(b-a)**(n-j) for j in range(k,n+1)),b**n)


def binomial_lower(n,k,alpha,*,bits=40):
    alpha=F(alpha)
    if not 0<alpha<1 or not isinstance(bits,int) or not 8<=bits<=128:raise ValueError('Invalid confidence budget or output bits')
    binomial_tail(n,k,F(0))
    if k==0:return F(0)
    scale=1<<bits;lo=0;hi=scale
    while hi-lo>1:
        mid=(lo+hi)//2
        if binomial_tail(n,k,F(mid,scale))<=alpha:lo=mid
        else:hi=mid
    return F(lo,scale)


def target_count_tail_upper(threshold,modes,mean_interval,variance_interval):
    if not isinstance(threshold,int) or not isinstance(modes,int) or modes<1:raise ValueError('Integer threshold and positive mode count required')
    lo,mu=map(F,mean_interval);vlo,var=map(F,variance_interval)
    if not 0<=lo<=mu<=modes or not 0<=vlo<=var:raise ValueError('Invalid target moments')
    if threshold<=0:return F(1)
    if threshold>modes:return F(0)
    bound=min(F(1),mu/threshold)
    if threshold>mu:bound=min(bound,var/(var+(threshold-mu)**2))
    return bound
