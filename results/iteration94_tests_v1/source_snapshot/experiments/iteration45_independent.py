"""Independent rational/density and interval audit primitives for round45."""
from fractions import Fraction as F
import numpy as np

def inverse(matrix):
    n=len(matrix);a=[[F(x) for x in row]+[F(i==j) for j in range(n)] for i,row in enumerate(matrix)]
    for j in range(n):
        k=next(k for k in range(j,n) if a[k][j]);a[j],a[k]=a[k],a[j];v=a[j][j];a[j]=[x/v for x in a[j]]
        for k in range(n):
            if k!=j:
                v=a[k][j];a[k]=[x-v*y for x,y in zip(a[k],a[j])]
    return [r[n:] for r in a]

def integer_density_law(tensors):
    """Double-layer transfer, independently of enumerated Fock amplitudes."""
    converted=[];scale=1
    for a in tensors:
        pairs=[float(x).as_integer_ratio() for z in a.flat for x in (z.real,z.imag)]
        exponent=max(d.bit_length()-1 for _,d in pairs);s=2**exponent;scale*=s
        values=[p*(s//q) for p,q in pairs]
        re=np.array(values[::2],dtype=object).reshape(a.shape);im=np.array(values[1::2],dtype=object).reshape(a.shape);converted.append((re,im))
    masses=[]
    for mask in range(1<<len(tensors)):
        er=np.ones((1,1),dtype=object);ei=np.zeros((1,1),dtype=object)
        for j,(ar,ai) in enumerate(converted):
            nr=np.zeros((ar.shape[2],ar.shape[2]),dtype=object);ni=nr.copy()
            for k in (range(1,ar.shape[1]) if mask>>j&1 else (0,)):
                r=ar[:,k,:];i=ai[:,k,:];ur=er@r-ei@i;ui=er@i+ei@r;nr+=r.T@ur+i.T@ui;ni+=r.T@ui-i.T@ur
            er,ei=nr,ni
        assert ei[0,0]==0 and er[0,0]>=0;masses.append(int(er[0,0]))
    norm=sum(masses)
    if norm<=0:raise ValueError('Zero state')
    return masses,norm,scale,[F(x,norm) for x in masses]

def grid_below(edge,bits):
    """Count grid points strictly below a rational edge by integer bisection."""
    edge=F(edge);lo=0;hi=1<<bits
    while lo<hi:
        mid=(lo+hi)//2
        if F(mid,1<<bits)<edge:lo=mid+1
        else:hi=mid
    return lo

def mobius(vacua,n):
    size=1<<n;v=[tuple(p) for p in vacua]
    for bit in range(n):
        for mask in range(size):
            if not mask>>bit&1:
                a,b=v[mask];c,d=v[mask|(1<<bit)];v[mask]=(a-d,b-c)
    raw=[v[(size-1)^mask] for mask in range(size)]
    return raw,[(max(F(0),a),min(F(1),b)) for a,b in raw]

def tv(law,target):
    assert len(law)==len(target)
    return (sum(max(F(0),a-x,x-b) for x,(a,b) in zip(law,target))/2,min(F(1),sum(max(abs(x-a),abs(x-b)) for x,(a,b) in zip(law,target))/2))
