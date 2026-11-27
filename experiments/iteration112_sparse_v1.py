"""Adaptive lower sets of orthogonalized even coherent-state pairs."""
import heapq
import itertools
import math
import mpmath as mp
import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist


def axis(s, levels=5):
    # The grid is only a search over dictionary centers. Projection of the final
    # dictionary is evaluated again at 65 decimal places.
    grid=np.linspace(.18,6.4,260)
    targets=np.exp(-.5*(1-np.tanh(s))*grid**2)/np.sqrt(np.cosh(s))
    diag=(1+np.exp(-2*grid**2))/2
    centers=[0.]
    def k(a,b):return .5*(np.exp(-.5*(a-b)**2)+np.exp(-.5*(a+b)**2))
    for n in range(levels-1):
        a=np.array(centers)
        l=np.linalg.cholesky(k(a[:,None],a))
        g=np.exp(-.5*(1-np.tanh(s))*a*a)/np.sqrt(np.cosh(s))
        w=solve_triangular(l,k(a[:,None],grid),lower=True)
        residual=targets-solve_triangular(l,g,lower=True)@w
        den=diag-(w*w).sum(0)
        usable=(den>1e-8)&(abs(grid[:,None]-a).min(1)>.16)
        gain=np.where(usable,residual**2/np.maximum(den,1e-15),-1)
        centers.append(float(grid[np.argmax(gain)]))
    with mp.workdps(65):
        xx=list(map(mp.mpf,centers))
        gram=mp.matrix([[(mp.exp(-(a-b)**2/2)+mp.exp(-(a+b)**2/2))/2 for a in xx] for b in xx])
        lower=mp.cholesky(gram)
        transform=lower**-1
        overlap=mp.matrix([mp.exp(-(1-mp.tanh(s))*a*a/2)/mp.sqrt(mp.cosh(s)) for a in xx])
        amplitude=transform*overlap
        return np.array(centers),np.array(transform.tolist(),float),np.array(list(amplitude),float)


def select(dictionaries,budget):
    amps=[v[2] for v in dictionaries]
    root=(0,)*len(amps)
    def cost(key):return 2**sum(j>0 for j in key)
    def priority(key):return -math.prod(a[j]**2 for a,j in zip(amps,key))/cost(key)
    queue=[(priority(root),root)]
    queued={root};selected=set();size=0
    while queue:
        score,key=heapq.heappop(queue)
        if size+cost(key)>budget:continue
        selected.add(key);size+=cost(key)
        for d in range(len(key)):
            if key[d]+1==len(amps[d]):continue
            nxt=key[:d]+(key[d]+1,)+key[d+1:]
            if nxt not in queued and all(nxt[:i]+(j-1,)+nxt[i+1:] in selected for i,j in enumerate(nxt) if j):
                queued.add(nxt);heapq.heappush(queue,(priority(nxt),nxt))
    return sorted(selected)


def project(s, dictionaries, budget):
    chosen=select(dictionaries,budget)
    coeff={index:0. for index in chosen}
    captured=[]
    for index in chosen:
        amp=math.prod(table[2][i] for table,i in zip(dictionaries,index))
        captured.append(amp**2)
        for index2 in itertools.product(*(range(i+1) for i in index)):
            coeff[index2]+=amp*math.prod(t[1][i,j] for t,i,j in zip(dictionaries,index,index2))
    positions=[];weight=[]
    for idx,v in coeff.items():
        scales=[t[0][i] for t,i in zip(dictionaries,idx)]
        signs=list(itertools.product(*[(x,-x) if x else (0.,) for x in scales]))
        positions.extend(signs)
        weight.extend([v/len(signs)]*len(signs))
    pos=np.array(positions)
    c=np.array(weight)
    gram=np.exp(-.5*cdist(pos,pos,'sqeuclidean'))
    norm=float(c@gram@c)
    g=np.exp(-.5*((pos*pos)@(1-np.tanh(s))))/np.sqrt(np.cosh(s)).prod()
    captured=math.fsum(captured)
    assert abs(c@g-captured)<3e-11 and abs(norm-captured)<3e-11
    c/=np.sqrt(norm)
    return pos,c,gram,dict(branches=len(c),blocks=len(chosen),budget=budget,
            state_bound=math.sqrt(max(0.,1-(c@g)**2)),acceptance=float(1/abs(c).sum()**2))
