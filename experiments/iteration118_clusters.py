"""Finite Gaussian contractions of two and three nonquadratic local factors."""
from functools import lru_cache
import itertools
import math
import numpy as np

F=[math.factorial(k) for k in range(24)]


def mul(a,b,n):
    return [sum(a[j]*b[k-j] for j in range(k+1)) for k in range(n+1)]


def two_sites(cap,J,powers,residual):
    n=len(J);cache={}
    if n<2:return cache
    # At ten core photons, at most two orders above the leading two-site covariance enter.
    extra=cap-3
    covariance=[(J@p)@J.T for p in powers[:extra+1]]
    diagonal=[c.diagonal() for c in covariance]
    def build_power(series):
        ans=[[1.]+[0.]*extra]
        for _ in range(cap):ans.append(mul(ans[-1],series,extra))
        return ans
    cp=build_power(covariance);vp=build_power(diagonal)
    def col(a):return a[:,None] if isinstance(a,np.ndarray) else a
    def row(a):return a[None,:] if isinstance(a,np.ndarray) else a
    for p in range(3,2*cap-2):
        for q in range(3,2*cap-p+1):
            marked=np.triu(residual[:,p,None]*residual[None,:,q],1)
            for c in range(min(p,q)+1):
                for a in range((p-c)//2+1):
                    for b in range((q-c)//2+1):
                        d=p-c-2*a;e=q-c-2*b;ell=a+b+c
                        room=cap-ell-d-e
                        if room<0:continue
                        factor=F[p]*F[q]/(F[d]*F[e]*F[c]*2**(a+b)*F[a]*F[b])
                        for order in range(room+1):
                            coeff=sum(col(vp[a][i])*row(vp[b][j])*cp[c][order-i-j]
                                      for i in range(order+1) for j in range(order-i+1))
                            term=marked*(factor*coeff)
                            if not d and not e:term=term.sum()
                            elif not e:term=term.sum(1)
                            elif not d:term=term.sum(0)
                            key=(d,e,ell+order)
                            cache[key]=cache.get(key,0)+term
    return cache


@lru_cache(None)
def sparse_wick(orders,cap):
    # Enumerate matchings of a labeled multiset up to symmetry at each site.
    # Only diagrams with at most cap Gaussian contractions or means can survive.
    n=len(orders)
    edges=list(itertools.combinations(range(n),2));out=[]
    def walk(idx,degrees,counts):
        if idx<len(edges):
            i,j=edges[idx]
            for v in range(min(degrees[i],degrees[j])+1):
                ds=list(degrees);ds[i]-=v;ds[j]-=v
                walk(idx+1,ds,counts+(v,))
            return
        for loops in itertools.product(*[range(d//2+1) for d in degrees]):
            singles=tuple(d-2*a for d,a in zip(degrees,loops))
            lead=sum(singles)+sum(counts)+sum(loops)
            if lead>cap:continue
            factor=(math.prod(F[p] for p in orders)/math.prod(F[x] for x in counts+singles+loops))/2**sum(loops)
            out.append((counts,loops,singles,lead,factor))
    walk(0,list(orders),())
    return out


class Triple:
    """At cap five the three-marked-site correction is a constant plus a linear form."""
    def __init__(self):
        self.constant=0j;self.linear=None
    def append(self,J,S,R):
        row=len(J)-1
        if row<2:return
        x=J@S
        if self.linear is None:self.linear=np.zeros(len(S),complex)
        k=x@J.T;v=k.diagonal()
        cov=[k[:row,:row],k[:row,row,None],k[None,:row,row]]
        diag=[v[:row,None],v[None,:row],v[row]]
        for orders in [(3,3,3),(3,3,4),(3,4,3),(4,3,3)]:
            p,q,r=orders
            marked=np.triu(R[:row,p,None]*R[None,:row,q],1)*R[row,r]
            for edges,loops,singles,lead,factor in sparse_wick(orders,5):
                assert lead==5
                term=marked*factor
                for v0,n in zip(cov,edges):
                    if n:term=term*v0**n
                for v0,n in zip(diag,loops):
                    if n:term=term*v0**n
                if not any(singles):self.constant+=term.sum()
                elif singles[0]:self.linear+=term.sum(1)@x[:row]
                elif singles[1]:self.linear+=term.sum(0)@x[:row]
                else:self.linear+=term.sum()*x[row]
