"""Positive rational Chow-Liu control; edge selection is numerical only.

Once the tree is fixed, factors and enumerated law are exact. Enumeration is
for small-table audit; the stored root/conditional factors have linear size.
"""
from fractions import Fraction as F
from math import log
from .pairwise_completion import feature_masks

def tree_control(moments,n):
    features=feature_masks(n);mu=dict(zip(features,map(F,moments)))
    if len(mu)!=len(features):raise ValueError('Complete low-order moments required')
    singles=[((1+mu[1<<i])/2,(1-mu[1<<i])/2) for i in range(n)]
    if any(min(p)<=0 for p in singles):raise ArithmeticError('Strictly positive singles required')
    pairs={};edges=[]
    for j in range(n):
        for i in range(j):
            pair=tuple((1+(-1)**a*mu[1<<i]+(-1)**b*mu[1<<j]+(-1)**(a+b)*mu[(1<<i)|(1<<j)])/4 for a in range(2) for b in range(2))
            if min(pair)<=0:raise ArithmeticError('Strictly positive pair table required')
            assert sum(pair)==1 and tuple(sum(pair[2*a+b] for b in range(2)) for a in range(2))==singles[i] and tuple(sum(pair[2*a+b] for a in range(2)) for b in range(2))==singles[j]
            pairs[i,j]=pair;score=sum(float(pair[2*a+b])*log(float(pair[2*a+b]/(singles[i][a]*singles[j][b]))) for a in range(2) for b in range(2));edges.append((score,i,j))
    components=list(range(n));chosen=[]
    def root(i):
        while components[i]!=i:i=components[i]
        return i
    for score,i,j in sorted(edges,key=lambda x:(-x[0],x[1],x[2])):
        a,b=root(i),root(j)
        if a!=b:components[b]=a;chosen.append((i,j,score))
    parent=[None]*n;parent[0]=-1;order=[0]
    for i in order:
        for a,b,_ in chosen:
            j=b if a==i else a if b==i else None
            if j is not None and parent[j] is None:parent[j]=i;order.append(j)
    assert len(order)==n;conditional={}
    for child in order[1:]:
        par=parent[child];i,j=sorted((par,child));pair=pairs[i,j];conditional[child]=tuple((pair[2*a+b] if par==i else pair[2*b+a])/singles[par][a] for a in range(2) for b in range(2))
    law=[]
    for mask in range(1<<n):
        p=singles[0][mask&1]
        for child in order[1:]:p*=conditional[child][2*((mask>>parent[child])&1)+((mask>>child)&1)]
        law.append(p)
    assert min(law)>0 and sum(law)==1
    return dict(law=law,singles=singles,pairs=pairs,selected_edges=chosen,parent=parent,order=order,root=singles[0],conditional=conditional,scope='Numerically selected tree; exact positive factors. Only tree-edge pairs are preserved, not all pair moments.')
