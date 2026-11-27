"""Compile connected Wick contractions into a shared arithmetic program.

Matrix IDs have an orientation. A transposed nonsymmetric factor reverses the
two site labels. Each operation is interned once per cutoff, independently of
the source parameters, displacement, detection instrument and number of sites.
"""
from collections import defaultdict
from functools import lru_cache
from itertools import combinations,permutations
import math
import numpy as np

from iteration118_auxiliary import series_exp
from iteration119_matching import graphs,compositions,Evaluation,product,eliminate


class Program:
    def __init__(self,cap):
        self.cap=cap
        self.ops=[('one',())]
        self.ids={}
        self.symmetric=set()
        self.reverse={0:0}
        totals=defaultdict(lambda:np.zeros(cap+1))
        for (degree,(nodes,cross)),coef in graphs(cap):
            edges=list(zip(combinations(range(len(nodes)),2),cross))
            labels=[('v',i,p) for i,p in enumerate(nodes) if any(p)]
            labels += [('e',(i,j),(count,)) for (i,j),count in edges if count]
            for orders in compositions(len(labels),cap-degree):
                v={i:0 for i in range(len(nodes))};e={}
                for (kind,site,powers),d in zip(labels,orders):
                    token=self.op((kind,*powers,d))
                    if kind=='v':v[site]=token
                    else:e[site]=token;self.symmetric.add(token)
                total=self.contract(v,e)
                totals[total][degree+sum(orders)]+=coef
        self.outputs=dict(totals)
        self.last_use={}
        for i,(key,deps) in enumerate(self.ops):
            for d in deps:self.last_use[d]=i

    def op(self,key,*deps):
        uid=self.ids.get((key,deps))
        if uid is None:
            uid=len(self.ops);self.ids[key,deps]=uid;self.ops.append((key,deps))
        return uid

    def trans(self,a):
        if a in self.symmetric:return a
        if a not in self.reverse:
            other=self.op('T',a);self.reverse[a]=other;self.reverse[other]=a
        return self.reverse[a]

    def times(self,a,b):
        if not a:return b
        if not b:return a
        a,b=sorted((a,b))
        result=self.op('times',a,b)
        if a in self.symmetric and b in self.symmetric:self.symmetric.add(result)
        return result

    def chain(self,a,w,b):
        # a and b point from the eliminated site to their surviving neighbor.
        if a>b:return self.trans(self.chain(b,w,a))
        result=self.op('chain',a,w,b)
        if a==b:self.symmetric.add(result)
        return result

    def contract(self,v,e):
        def edge(i,j):
            return e[i,j] if i<j else self.trans(e[j,i])
        while len(v)>1:
            adj={i:[] for i in v}
            for i,j in e:adj[i].append(j);adj[j].append(i)
            i=min(adj,key=lambda j:(len(adj[j]),bool(v[j]),j))
            if len(adj[i])>=3:
                if len(v)==4:
                    # Choose the external edge of a tetrahedron so that the
                    # five-factor cavity, usually homogeneous, can be shared.
                    best=None
                    for a,b,c,d in permutations(v):
                        factors=(edge(a,c),edge(a,d),edge(b,c),edge(b,d),edge(c,d),v[c],v[d])
                        rank=(sum(bool(x) for x in factors[-2:])+len(set(factors[:5])),factors)
                        if best is None or rank<best[0]:best=(rank,(a,b,c,d),factors)
                    _,(a,b,c,d),factors=best
                    cavity=self.op('cavity',*factors)
                    pair=self.times(edge(a,b),cavity)
                    return self.op('pair',v[a],pair,v[b])
                verts=list(v)
                links=[(verts.index(i),verts.index(j)) for i,j in e]
                return self.op(('dense',tuple(links),len(v)),*v.values(),*e.values())
            neighbors=adj[i];factors=[edge(i,j) for j in neighbors]
            for j in neighbors:del e[tuple(sorted((i,j)))]
            w=v.pop(i)
            if len(neighbors)==1:
                j=neighbors[0];v[j]=self.times(v[j],self.op('leaf',w,factors[0]))
            else:
                j,k=neighbors;ab=self.chain(factors[0],w,factors[1])
                if j>k:j,k=k,j;ab=self.trans(ab)
                e[j,k]=self.times(e.get((j,k),0),ab)
        return self.op('sum',next(iter(v.values())))

    def evaluate(self,cov,mean,device=None,compensated=False):
        base=Evaluation(cov,mean,self.cap)
        if compensated:
            terms=[[] for _ in range(self.cap+1)]
        if device:
            from iteration120_dense import Dense
            bath=Dense(device);deferred=[]
        n=len(mean[0]);values={0:None};log=np.zeros(self.cap+1,complex)
        for i,(key,deps) in enumerate(self.ops[1:],1):
            a=[values[d] for d in deps]
            if isinstance(key,tuple):
                if key[0]=='v':
                    _,loops,free,k=key
                    ans=product(base.powers('var',loops,k),base.powers('mean',free,k),k)[k]
                elif key[0]=='e':
                    _,p,k=key;ans=base.powers('edge',p,k)[k]
                else:
                    _,links,h=key
                    if device:
                        val=bath.evaluate(links,h,a,n)
                        deferred.append((self.outputs[i],val))
                        ans=0.
                    else:ans=eliminate(dict(enumerate(a[:h])),dict(zip(links,a[h:])),n)
            elif key=='times':ans=a[0]*a[1]
            elif key=='T':ans=a[0].T
            elif key=='chain':
                left,w,right=a
                ans=left.T@right if w is None else left.T@(w[:,None]*right)
            elif key=='leaf':ans=a[1].sum(0) if a[0] is None else a[0]@a[1]
            elif key=='sum':ans=n if a[0] is None else a[0].sum()
            elif key=='pair':
                left,mat,right=a
                if left is None and right is None:ans=mat.sum()
                elif left is None:ans=mat.sum(0)@right
                elif right is None:ans=left@mat.sum(1)
                else:ans=left@mat@right
            elif key=='cavity':
                ac,ad,bc,bd,cd,wc,wd=a
                if wc is not None:cd=wc[:,None]*cd
                if wd is not None:cd=cd*wd
                ans=np.empty_like(ac)
                for site in range(n):
                    ans[site]=np.einsum('ij,ij->i',(bc*ac[site])@cd,bd*ad[site])
            else:raise ValueError(key)
            values[i]=ans
            if i in self.outputs:
                if compensated:
                    for k in np.flatnonzero(self.outputs[i]):terms[k].append(self.outputs[i][k]*ans)
                else:log+=self.outputs[i]*ans
            for d in set(deps):
                if d and self.last_use[d]==i:del values[d]
            if i not in self.last_use:del values[i]
        if compensated:
            log=np.array([complex(math.fsum(x.real for x in t),math.fsum(x.imag for x in t))
                          for t in terms])
        if device and deferred:
            import torch
            weights,scalars=zip(*deferred)
            log+=torch.stack(scalars).cpu().numpy()@np.stack(weights)
        return log

program=lru_cache(None)(Program)


def amplitudes(S,v,J,cap,device=None,compensated=False):
    v=v+J.sum(0)
    B=-.5*J.T@J;op=2*S@B
    powers=[S];log=np.zeros(cap+1,complex)
    for k in range(1,cap+1):
        mat=powers[-1]
        log[k]=np.einsum('ij,ji',B,mat)/k+.5*v@mat@v
        if k<cap:powers.append(op@mat)
    if len(J):log+=program(cap).evaluate([J@p@J.T for p in powers],[J@p@v for p in powers],device,compensated)
    return np.cumsum(series_exp(log))
