"""Independent multinomial density polynomials and pairing-count normal moments."""
from fractions import Fraction as F
from math import factorial,comb,prod
from functools import lru_cache
import numpy as np
from iteration47_radicals import mul,conj,squarefree,bisection_root
from iteration48_independent import rational_amplitudes,bisect_sqrt
from iteration45_independent import inverse,mobius
from verify_iteration34 import exact_determinant

def pmul(a,b):
    result={}
    for p,x in a.items():
        for q,y in b.items():
            k=tuple(i+j for i,j in zip(p,q));v=mul(x,y);old=result.get(k,(F(0),F(0)));result[k]=(old[0]+v[0],old[1]+v[1])
    return {k:v for k,v in result.items() if v!=(0,0)}

def power(z,k):
    value=(F(1),F(0))
    for _ in range(k):value=mul(value,z)
    return value

def compositions(total,parts):
    if parts==1:
        yield (total,);return
    for first in range(total+1):
        for rest in compositions(total-first,parts-1):yield (first,)+rest


def density_polynomials(tensors,directions):
    r=len(directions);assert r in (1,2,3,4);dims=[t.shape[1] for t in tensors];n=len(dims);occ=list(np.ndindex(*dims));amps=rational_amplitudes(tensors);norm=sum(a*a+b*b for a,b in amps);assert norm>0;powers=[]
    for j,d in enumerate(dims):
        p=[]
        for k in range(d):
            terms={}
            for degree in compositions(k,r):
                value=(F(1),F(0));count=factorial(k)
                for axis,exponent in enumerate(degree):
                    value=mul(value,power(tuple(-F(x) for x in directions[axis][j]),exponent));count//=factorial(exponent)
                terms[degree]=(value[0]*count,value[1]*count)
            p.append(terms)
        powers.append(p)
    pair_polys=[{(a,b):pmul(p[a],{k:conj(v) for k,v in p[b].items()}) for a in range(d) for b in range(d)} for p,d in zip(powers,dims)];records=[]
    for mask in range(1<<n):
        groups={};imaginary={}
        for oa,aa in zip(occ,amps):
            if aa==(0,0):continue
            for ob,ab in zip(occ,amps):
                if ab==(0,0) or any(oa[j]!=ob[j] for j in range(n) if not mask>>j&1):continue
                p={(0,)*r:mul(conj(aa),ab)};fact=1
                for j in range(n):
                    if mask>>j&1:p=pmul(p,pair_polys[j][oa[j],ob[j]]);fact*=factorial(oa[j])*factorial(ob[j])
                sq,rad=squarefree(fact);group=groups.setdefault(rad,{});imag=imaginary.setdefault(rad,{})
                for k,(a,b) in p.items():group[k]=group.get(k,F(0))+a/(sq*norm);imag[k]=imag.get(k,F(0))+b/(sq*norm)
        assert all(v==0 for g in imaginary.values() for v in g.values());records.append(dict(mask=mask,groups={rad:{k:v for k,v in p.items() if v} for rad,p in groups.items()}))
    return dict(dimensions=dims,directions=directions,records=records)

def evaluate_density(polynomial,xs):
    records=[]
    for r in polynomial['records']:
        groups={rad:sum(c*prod(x**k for x,k in zip(xs,p)) for p,c in terms.items()) for rad,terms in r['groups'].items()};lo=hi=F(0)
        for rad,c in groups.items():
            a,b=bisection_root(rad);lo+=c*(a if c>=0 else b);hi+=c*(b if c>=0 else a)
        records.append(dict(mask=r['mask'],radical_coefficients=groups,coefficient_interval=(max(F(0),lo),hi)))
    return records

def closed_moment(p,C):
    return pairing_moment(tuple(p),tuple(tuple(F(x) for x in row) for row in C))

@lru_cache(None)
def pairing_moment(p,C):
    if sum(p)%2:return F(0)
    r=len(p);edges=[(i,j) for i in range(r) for j in range(i+1,r) if C[i][j]]
    @lru_cache(None)
    def configurations(k,remaining):
        if k==len(edges):
            if any(x%2 for x in remaining):return F(0)
            return prod((C[i][i]/2)**(x//2)/factorial(x//2) for i,x in enumerate(remaining))
        i,j=edges[k];total=F(0)
        for cross in range(min(remaining[i],remaining[j])+1):
            rest=list(remaining);rest[i]-=cross;rest[j]-=cross
            total+=C[i][j]**cross/factorial(cross)*configurations(k+1,tuple(rest))
        return total
    return prod(factorial(x) for x in p)*configurations(0,p)


def integrate_density(polynomial,sigmas):
    v=polynomial['directions'];r=len(v);n=len(polynomial['dimensions']);sigmas=[F(s) for s in sigmas];assert all(s>0 for s in sigmas);records=[];vac=[]
    for item in polynomial['records']:
        mask=item['mask'];G=[[sum(F(v[a][j][0])*v[b][j][0]+F(v[a][j][1])*v[b][j][1] for j in range(n) if mask>>j&1) for b in range(r)] for a in range(r)];precision=[[F(a==b)/sigmas[a]**2+2*G[a][b] for b in range(r)] for a in range(r)];C=inverse(precision);ratio=1/(exact_determinant(precision)*prod(s*s for s in sigmas));groups={rad:sum(c*closed_moment(p,C) for p,c in terms.items()) for rad,terms in item['groups'].items()};lo=hi=F(0)
        for rad,c in groups.items():
            a,b=bisect_sqrt(ratio/rad);lo+=c*(a if c>=0 else b);hi+=c*(b if c>=0 else a)
        value=(max(F(0),lo),min(F(1),hi));vac.append(value);records.append(dict(mask=mask,tilted_covariance=C,normalization_squared=ratio,integrated_radicals=groups,vacuum_interval=value))
    raw,law=mobius(vac,n);return dict(records=records,vacuum_intervals=vac,raw_intervals=raw,intervals=law)

def serialize_polynomial(p):
    return dict(dimensions=p['dimensions'],directions=p['directions'],records=[dict(mask=r['mask'],groups=[dict(radical=rad,terms=[dict(degree=k,coefficient=c) for k,c in sorted(terms.items())]) for rad,terms in sorted(r['groups'].items())]) for r in p['records']])
