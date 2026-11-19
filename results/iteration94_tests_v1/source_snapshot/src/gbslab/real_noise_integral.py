"""Exact continuous real-normal displacement average on one finite input mode."""
from fractions import Fraction as F
from math import factorial,prod
import numpy as np
from .displaced_fock_interval import reciprocal_root_integer,multiply,add
from .gaussian_fock_bridge import root_interval


def integrated_vacuum(tensor,sigma,*,bits=192):
    t=np.asarray(tensor,complex);sigma=F(sigma)
    if t.ndim!=3 or t.shape[0]!=1 or t.shape[2]!=1 or not 1<=t.shape[1]<=8 or not np.all(np.isfinite(t)) or not 0<=sigma<=F(1,2):raise ValueError('One finite mode,dimension1..8 and0<=sigma<=1/2 required')
    amplitudes=[(F(float(z.real)),F(float(z.imag))) for z in t.ravel()];norm=sum(a*a+b*b for a,b in amplitudes)
    if not norm:raise ValueError('Nonzero input state required')
    A=1+2*sigma*sigma;coefficient=(F(0),F(0));moments={};terms=[]
    for power in range(2*len(amplitudes)-1):
        moments[power]=F(0) if power%2 else F(prod(range(1,power,2)))*sigma**power/A**(power//2)
    for a,(ar,ai) in enumerate(amplitudes):
        for b,(br,bi) in enumerate(amplitudes):
            c=(ar*br+ai*bi)*((-1)**(a+b))*moments[a+b]/norm;root=reciprocal_root_integer(factorial(a)*factorial(b),bits);coefficient=add(coefficient,multiply((c,c),root));terms.append(dict(first=a,second=b,coefficient=c,factorial_product=factorial(a)*factorial(b)))
    normalization=root_interval(1/A,bits);raw=multiply(coefficient,normalization);p=(max(F(0),raw[0]),min(F(1),raw[1]));assert 0<=p[0]<=p[1]<=1
    return dict(sigma=sigma,normalization_squared=1/A,normalization_interval=normalization,scaled_even_moments=moments,coefficient_interval=coefficient,terms=terms,vacuum_interval=p,law=[p,(1-p[1],1-p[0])],scope='Exact continuous real Gaussian displacement of normalized stored finite one-mode state; no quadrature or output cutoff')
