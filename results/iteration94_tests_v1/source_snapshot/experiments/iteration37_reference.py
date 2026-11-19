"""Independent dense Gaussian selection and Kruskal graph checks (no sampler imports)."""
import math,itertools
import numpy as np


def vacuum(V,modes):
    if not modes:return 1.0
    m=len(V)//2;ix=list(modes)+[j+m for j in modes];Q=(V[np.ix_(ix,ix)]+np.eye(len(ix)))/2;s,d=np.linalg.slogdet(Q);assert s>0;return math.exp(-d/2)


def mi(V,i,j):
    a=vacuum(V,(i,));b=vacuum(V,(j,));c=vacuum(V,(i,j));p=np.array([c,b-c,a-c,1-a-b+c]);assert np.min(p)>0
    product=np.array([a*b,(1-a)*b,a*(1-b),(1-a)*(1-b)]);d=(p-product)/product
    # The linear terms sum to zero algebraically; use a convergent series near independence.
    def h(x):
        if abs(x)<.01:return sum((-1)**k*x**k/(k*(k-1)) for k in range(2,16))
        return (1+x)*math.log1p(x)-x
    return sum(float(w)*h(float(x)) for w,x in zip(product,d))


def weights(V):
    m=len(V)//2;W=np.zeros((m,m))
    for i in range(m):
        for j in range(i):W[i,j]=W[j,i]=mi(V,i,j)
    return W


def retained(V,W,order,prefix,selection):
    clicks=tuple(order[i] for i,b in enumerate(prefix) if b=='1');mode=order[len(prefix)]
    if len(clicks)<=4:return clicks
    if selection=='recent':return clicks[-4:]
    if selection=='pairwise':return tuple(sorted(clicks,key=lambda j:(-W[mode,j],j))[:4])
    zeros=tuple(order[i] for i,b in enumerate(prefix) if b=='0');m=len(V)//2;active=tuple(sorted(clicks+(mode,)));ix=active+tuple(j+m for j in active);Z=zeros+tuple(j+m for j in zeros);C=V[np.ix_(ix,ix)].copy()
    if Z:C-=V[np.ix_(ix,Z)]@np.linalg.solve(V[np.ix_(Z,Z)]+np.eye(len(Z)),V[np.ix_(Z,ix)])
    index={j:i for i,j in enumerate(active)};next_mode=index[mode]
    if selection=='conditional_mi':return tuple(sorted(clicks,key=lambda j:(-mi(C,next_mode,index[j]),j))[:4])
    assert selection=='click_influence';memo={():1.0}
    def vac(modes):
        key=tuple(sorted(modes))
        if key not in memo:memo[key]=vacuum(C,key)
        return memo[key]
    def click(selected):
        bits=tuple(index[j] for j in selected);den=[];num=[]
        for k in range(len(bits)+1):
            for subset in itertools.combinations(bits,k):den.append((-1)**k*vac(subset));num.append((-1)**k*vac(subset+(next_mode,)))
        return 1-math.fsum(num)/math.fsum(den)
    chosen=[];available=set(clicks);base=click(())
    for _ in range(4):
        p={j:click(chosen+[j]) for j in available};best=min(available,key=lambda j:(-abs(p[j]-base),j));chosen.append(best);available.remove(best);base=p[best]
    return tuple(chosen)


def kruskal(W,vertices):
    vertices=tuple(vertices);parent={j:j for j in vertices};edges=[]
    def find(j):
        while parent[j]!=j:j=parent[j]
        return j
    for _,i,j in sorted((-W[i,j],min(i,j),max(i,j)) for k,i in enumerate(vertices) for j in vertices[k+1:]):
        a,b=find(i),find(j)
        if a!=b:parent[a]=b;edges.append((i,j))
        if len(edges)==len(vertices)-1:break
    return set(edges)
