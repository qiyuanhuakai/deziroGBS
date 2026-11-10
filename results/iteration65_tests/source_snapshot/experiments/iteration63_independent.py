"""Independent edge-combination/pair counting; no production graph imports."""
from collections import Counter
from fractions import Fraction as F
from itertools import combinations
from math import comb,lcm

def reconstruct(n,edges):
    matchings=[]; pm=[0]*(1<<n)
    for k in range(n//2+1):
        for selected in combinations(edges,k):
            vertices=[v for edge in selected for v in edge]
            if len(set(vertices))==2*k:
                mask=sum(1<<v for v in vertices); pm[mask]+=1; matchings.append((mask,selected))
    hist=Counter(); witnesses=set(); full=(1<<n)-1
    for s,a in matchings:
        for t,b in matchings:
            h=len(a)+len(b); unused=full ^ (s|t); u=unused.bit_count()
            for v in range(u+1): hist[h+v,h]+=comb(u,v)
            if s==t:
                w=tuple(sorted((*map(tuple,a),*((i+n,j+n) for i,j in b),*((i,i+n) for i in range(n) if unused & (1<<i)))))
                witnesses.add((s,w))
    return pm,hist,witnesses,len(matchings)

def integer_determinant(a):
    n=len(a); denominator=lcm(*(x.denominator for row in a for x in row)); b=[[int(x*denominator) for x in row] for row in a]; previous=1; sign=1
    for k in range(n-1):
        pivot=next((j for j in range(k,n) if b[j][k]),None)
        if pivot is None: return F(0)
        if pivot!=k: b[k],b[pivot]=b[pivot],b[k];sign=-sign
        for i in range(k+1,n):
            for j in range(k+1,n):
                value=b[i][j]*b[k][k]-b[i][k]*b[k][j]; assert value%previous==0; b[i][j]=value//previous
        previous=b[k][k]
        for i in range(k+1,n): b[i][k]=0
    return F(sign*b[-1][-1],denominator**n)

def sqrt_interval(x,bits=192):
    scale=1<<bits; lo=0; hi=scale
    while F(hi*hi,scale*scale)<x: hi*=2
    while lo<hi:
        mid=(lo+hi+1)//2
        if F(mid*mid,scale*scale)<=x: lo=mid
        else: hi=mid-1
    return F(lo,scale),F(lo if F(lo*lo,scale*scale)==x else lo+1,scale)

def word_cells(p,bits=64):
    # Count integer words using floor and an explicit exact-boundary branch.
    size=1<<bits; total=F(0); previous=0; counts=[]
    for value in p:
        total+=value; a=total*size; end=a.numerator//a.denominator + int(a.denominator!=1)
        counts.append(end-previous); previous=end
    return counts

def locate(p,word,bits=64):
    # Sequential inverse CDF, independent of production binary search.
    cumulative=F(0)
    for i,value in enumerate(p):
        cumulative+=value
        if F(word,1<<bits)<cumulative: return i
    raise AssertionError('word outside CDF')
