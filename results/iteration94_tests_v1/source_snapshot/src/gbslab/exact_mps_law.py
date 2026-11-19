"""Exact integer zero-displacement threshold law of stored binary64 MPS.

This enumerates a small Fock box for audit. It certifies the represented
finite state after exact normalization, not the original Gaussian state.
"""
from fractions import Fraction as F
import math
import numpy as np

def integer_tensors(tensors,max_states=100000):
    if not tensors:raise ValueError('Nonempty tensor chain required')
    tensors=[np.asarray(a,dtype=complex) for a in tensors];left=1;total=1;result=[]
    for a in tensors:
        if a.ndim!=3 or min(a.shape)<1 or a.shape[0]!=left or not np.all(np.isfinite(a)):raise ValueError('Finite compatible MPS required')
        total*=a.shape[1];left=a.shape[2]
        if total>max_states:raise MemoryError('Exact Fock-box audit cap')
        real=[F(float(x)) for x in a.real.flat];imag=[F(float(x)) for x in a.imag.flat];scale=max(x.denominator for x in real+imag);r=np.array([x.numerator*(scale//x.denominator) for x in real],dtype=object).reshape(a.shape);i=np.array([x.numerator*(scale//x.denominator) for x in imag],dtype=object).reshape(a.shape);result.append((r,i,scale))
    if left!=1:raise ValueError('Scalar right boundary required')
    return result

def represented_threshold_law(tensors,max_states=100000):
    integer=integer_tensors(tensors,max_states);real=np.ones((1,1),dtype=object);imag=np.zeros((1,1),dtype=object);scale=1;dims=[]
    for ar,ai,den in integer:
        d=ar.shape[1];nr=[];ni=[]
        for occupation in range(d):nr.append(real@ar[:,occupation,:]-imag@ai[:,occupation,:]);ni.append(real@ai[:,occupation,:]+imag@ar[:,occupation,:])
        real=np.stack(nr,axis=1).reshape(-1,ar.shape[2]);imag=np.stack(ni,axis=1).reshape(-1,ar.shape[2]);scale*=den;dims.append(d)
    weights=[int(a)**2+int(b)**2 for a,b in zip(real[:,0],imag[:,0])];norm=sum(weights)
    if norm<=0:raise ArithmeticError('Cannot normalize zero represented state')
    masses=[0]*(1<<len(dims))
    for occupation,weight in zip(np.ndindex(*dims),weights):masses[sum(1<<j for j,n in enumerate(occupation) if n>0)]+=weight
    law=[F(x,norm) for x in masses];assert sum(law)==1 and min(law)>=0
    return dict(distribution=law,integer_threshold_masses=masses,integer_norm=norm,amplitude_denominator=scale,represented_norm=F(norm,scale**2),fock_dimensions=dims,fock_states=math.prod(dims),scope='Exact normalized stored tensor state; zero displacement only; exponential audit')
