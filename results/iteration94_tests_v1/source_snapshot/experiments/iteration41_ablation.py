"""Exact rational controls for K2 truncation, marginal approximation, and clamps.

These real-arithmetic controls are not substitutes for the official floating
sampler. Negative signed masses are retained, never silently normalized.
"""
from fractions import Fraction as F
from functools import lru_cache


def validate(pvac,cov,prefix):
    n=len(pvac)
    if not n or len(prefix)>n or set(prefix)-{'0','1'}:raise ValueError('Invalid dimension or prefix')
    p=tuple(map(F,pvac));c={tuple(k):F(v) for k,v in cov.items()}
    if any(not 0<=x<=1 for x in p) or set(c)!={(j,i) for i in range(n) for j in range(i)}:raise ValueError('Invalid coefficients')
    return p,c


def matching_mass(prefix,pvac,cov):
    """All singleton/pair partitions, exactly: K2 cumulant truncation alone."""
    p,c=validate(pvac,cov,prefix);sign=[1 if b=='0' else -1 for b in prefix]
    @lru_cache(None)
    def mass(indices):
        if not indices:return F(1)
        i,*rest=indices;rest=tuple(rest)
        return (p[i] if sign[i]==1 else 1-p[i])*mass(rest)+sum((sign[i]*sign[j]*c[min(i,j),max(i,j)]*mass(tuple(k for k in rest if k!=j)) for j in rest),F(0))
    return mass(tuple(range(len(prefix))))


def recurrence(prefix,pvac,cov,*,clamps=True):
    """Rational official K2 recurrence with optional absolute-mass clamps."""
    p,c=validate(pvac,cov,prefix);sign=[1 if b=='0' else -1 for b in prefix]
    ps=[p[i] if sign[i]==1 else 1-p[i] for i in range(len(prefix))]
    K={(j,i):sign[j]*c[j,i] for i in range(len(prefix)) for j in range(i)}
    records={};clipped=[]
    def clip(value,label):
        if not clamps:return value
        out=max(F(0),min(F(1),value))
        if out!=value:clipped.append((label,value,out))
        return out
    def combine(previous,total,last,label):return clip(previous*ps[last]+sign[last]*total,label)
    @lru_cache(None)
    def main(end):
        if end==0:return F(1)
        last=end-1;previous=main(last)
        zero=clip(previous*p[last]+sum((K[j,last]*elided(j,last) for j in range(last)),F(0)),('main',end))
        records[last]=dict(mode=last,previous=previous,zero=zero,threshold=None if previous==0 else zero/previous)
        return zero if sign[last]==1 else previous-zero
    @lru_cache(None)
    def partial(begin,end):
        if begin==end:return F(1)
        if begin==0:return main(end)
        last=end-1;total=sum((K[j,last]*partial_elided(begin,j,last) for j in range(begin,last)),F(0))
        return combine(partial(begin,last),total,last,('partial',begin,end))
    @lru_cache(None)
    def elided(e,end):
        if e==0:return partial(1,end)
        if e==end-1:return main(end-1)
        last=end-1;total=sum((K[j,last]*double_elided(min(j,e),max(j,e),last) for j in range(last) if j!=e),F(0))
        return combine(elided(e,last),total,last,('elided',e,end))
    @lru_cache(None)
    def partial_elided(begin,e,end):
        if begin==e:return partial(begin+1,end)
        if begin==0:return elided(e,end)
        if e==end-1:return partial(begin,end-1)
        last=end-1;total=F(0)
        for j in range(begin,last):
            if j!=e:
                a,b=min(j,e),max(j,e)
                total+=K[j,last]*partial_elided(begin,a,b)*partial(b+1,last)
        return combine(partial_elided(begin,e,last),total,last,('partial_elided',begin,e,end))
    @lru_cache(None)
    def double_elided(a,b,end):
        if a==0 and b==1:return partial(2,end)
        if a==0 and b==end-1:return partial(1,end-1)
        if a==end-2 and b==end-1:return main(end-2)
        if b==end-1:return elided(a,end-1)
        last=end-1;total=F(0)
        for j in range(last):
            if j!=a and j!=b:
                x,y,z=sorted((a,b,j));total+=K[j,last]*double_elided(x,y,z)*partial(z+1,last)
        return combine(double_elided(a,b,last),total,last,('double_elided',a,b,end))
    mass=main(len(prefix))
    return dict(mass=mass,records=[records[j] for j in range(len(prefix))],clipped=clipped)


def enumerate_laws(pvac,cov):
    """Full signed controls and normalized real-uniform official recurrence law."""
    n=len(pvac);matching=[];unclamped=[];clamped=[];tree={}
    for mask in range(1<<n):
        prefix=''.join(str((mask>>j)&1) for j in range(n))
        matching.append(matching_mass(prefix,pvac,cov))
        unclamped.append(recurrence(prefix,pvac,cov,clamps=False)['mass'])
        record=recurrence(prefix,pvac,cov)
        clamped.append(record['mass'])
        for j,row in enumerate(record['records']):
            key=prefix[:j]
            if key in tree:assert tree[key]==row
            else:tree[key]=row
    law=[F(0)]*(1<<n);pending=[('',F(1),0)];reachable=[];saturated=[]
    while pending:
        prefix,mass,mask=pending.pop();j=len(prefix)
        if j==n:law[mask]=mass;continue
        if not mass:continue
        row=tree[prefix];assert row['previous']>0
        q=row['threshold'];reachable.append(prefix)
        if not 0<=q<=1:saturated.append(dict(prefix=prefix,threshold=q,output_prefix_probability=mass))
        q=max(F(0),min(F(1),q))
        pending.extend(((prefix+'0',mass*q,mask),(prefix+'1',mass*(1-q),mask|(1<<j))))
    assert sum(matching)==sum(unclamped)==sum(clamped)==sum(law)==1 and all(x>=0 for x in law)
    return dict(matching=matching,unclamped=unclamped,clamped=clamped,sampler=law,tree=tree,reachable=reachable,saturated=saturated)


def common_thermal(n,intensity):
    """V=I+2 intensity diag(J_n,J_n); classical Gaussian displacement of vacuum."""
    a=F(intensity)
    if type(n) is not int or n<1 or a<0:raise ValueError('Invalid thermal input')
    p=1/(1+a);cov=1/(1+2*a)-p*p
    return [p]*n,{(j,i):cov for i in range(n) for j in range(i)}


def common_thermal_law(n,intensity):
    from math import comb
    a=F(intensity);by_clicks=[sum(((-1)**j*F(comb(k,j),1)/(1+a*(n-k+j)) for j in range(k+1)),F(0)) for k in range(n+1)]
    law=[by_clicks[mask.bit_count()] for mask in range(1<<n)]
    assert sum(law)==1 and all(x>=0 for x in law)
    return law
