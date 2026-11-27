"""Finite pair expansion of a complete displaced measurement amplitude."""
from collections import defaultdict
from functools import lru_cache
import itertools
import math
import numpy as np
from iteration118_clusters import sparse_wick
from iteration118_auxiliary import series_exp


@lru_cache(None)
def graphs(cap):
    # A connected Wick diagram is determined by its loops, unpaired legs
    # and edge multiplicities. Logarithms collect each connected diagram once.
    result=defaultdict(float)
    for h in range(1,2*cap//3+1):
        edges=list(itertools.combinations(range(h),2))
        for orders in itertools.combinations_with_replacement(range(3,2*cap+1),h):
            if sum(orders)>2*cap:continue
            denom=math.prod(math.factorial(orders.count(p)) for p in set(orders))
            vertex=math.prod((-1)**(p-1)/p for p in orders)/denom
            for cross,loops,free,degree,multiplicity in sparse_wick(orders,cap):
                reach={0}
                for _ in range(h-1):
                    for (i,j),m in zip(edges,cross):
                        if m and (i in reach or j in reach):reach.update((i,j))
                if len(reach)!=h:continue
                # Relabel a small diagram canonically for the homomorphism sum.
                edge=dict(zip(edges,cross))
                def key(perm):
                    return (tuple((loops[i],free[i]) for i in perm),
                            tuple(edge[tuple(sorted((perm[i],perm[j])))] for i,j in edges))
                # Invariant vertex classes avoid enumerating permutations
                # that cannot identify two labeled versions of the same graph.
                signature=[(loops[i],free[i],tuple(sorted(edge[tuple(sorted((i,j)))] for j in range(h) if i!=j)))
                           for i in range(h)]
                ordered=sorted(range(h),key=signature.__getitem__)
                groups=[tuple(g) for _,g in itertools.groupby(ordered,key=signature.__getitem__)]
                permutations=itertools.product(*(itertools.permutations(g) for g in groups))
                label=min(key(tuple(itertools.chain.from_iterable(p))) for p in permutations)
                result[(degree,label)]+=vertex*multiplicity
    return list(result.items())


def product(a,b,room):
    return [sum(a[i]*b[k-i] for i in range(k+1)) for k in range(room+1)]


class Evaluation:
    def __init__(self,cov,mean,cap):
        self.cov=cov;self.mean=mean;self.cap=cap;self.cache={}
        self.n=len(mean[0]);self.edge_marginal=None

    def powers(self,kind,n,room):
        key=(kind,n,room)
        if key not in self.cache:
            src=self.cov if kind=="edge" else self.mean if kind=="mean" else [x.diagonal() for x in self.cov]
            if n==1:
                self.cache[key]=src[:room+1]
                return self.cache[key]
            base=np.ones_like(src[0])
            val=[base]+[0.] * room
            for _ in range(n):val=product(val,src,room)
            self.cache[key]=val
        return self.cache[key]

    def tetra(self,indices,args):
        edges={tuple(ord(c)-97 for c in name):a for name,a in zip(indices,args) if len(name)==2}
        if len(edges)!=6 or not all(i<4 and j<4 for i,j in edges):return None
        # A homogeneous tetrahedron with at most one changed edge or two
        # decorated vertices needs only a two-index marginal of the base graph.
        base=self.powers("edge",1,0)[0]
        changed=[key for key,a in edges.items() if a is not base and not np.shares_memory(a,base)]
        if len(changed)>1:return None
        nodes={ord(name)-97:a for name,a in zip(indices,args) if len(name)==1}
        if len(nodes)>2:return None
        pair=changed[0] if changed else next((key for key in edges if set(nodes)<=set(key)),None)
        if pair is None or not set(nodes)<=set(pair):return None
        if self.edge_marginal is None:
            out=np.empty_like(base)
            for i,weights in enumerate(base):
                # H[i,j] = (C[j]*C[i]) @ C @ (C[j]*C[i]).T
                prod=base*weights
                out[i]=np.einsum("ij,ij->i",prod@base,prod)
            self.edge_marginal=out
        val=self.edge_marginal*edges[pair]
        if pair[0] in nodes:val=val*nodes[pair[0]][:,None]
        if pair[1] in nodes:val=val*nodes[pair[1]][None,:]
        return val.sum()

    def evaluate(self,label,room):
        nodes,cross=label;h=len(nodes);factors=[];indices=[]
        for i,(loops,free) in enumerate(nodes):
            if loops or free:
                key=("node",loops,free,room)
                if key not in self.cache:self.cache[key]=product(self.powers("var",loops,room),
                                                                self.powers("mean",free,room),room)
                factors.append(self.cache[key]);indices.append(chr(97+i))
        for (i,j),m in zip(itertools.combinations(range(h),2),cross):
            if m:
                factors.append(self.powers("edge",m,room));indices.append(chr(97+i)+chr(97+j))
        expression=",".join(indices)+"->"
        vals=np.zeros(room+1,complex)
        for degrees in compositions(len(factors),room):
            args=[f[k] for f,k in zip(factors,degrees)]
            if any(np.ndim(v)==0 for v in args):continue
            value=self.tetra(indices,args) if h==4 else None
            if value is None:value=network_sum(indices,args,h,self.n)
            vals[sum(degrees)]+=value
        return vals


def network_sum(indices,arrays,h,n):
    nodes={i:None for i in range(h)};edges={}
    for name,a in zip(indices,arrays):
        ix=tuple(ord(k)-97 for k in name)
        if len(ix)==1:nodes[ix[0]]=a
        else:edges[ix]=a
    return eliminate(nodes,edges,n)


def eliminate(nodes,edges,n):
    def link(i,j):
        if i<j:return edges[i,j]
        return edges[j,i].T
    while len(nodes)>1:
        adjacent={i:[] for i in nodes}
        for a,b in edges:adjacent[a].append(b);adjacent[b].append(a)
        i=min(nodes,key=lambda a:len(adjacent[a]))
        nbr=adjacent[i];weight=nodes.pop(i)
        matrices=[link(i,j) for j in nbr]
        for j in nbr:del edges[min(i,j),max(i,j)]
        if len(nbr)==1:
            j=nbr[0];value=matrices[0].sum(0) if weight is None else weight@matrices[0]
            nodes[j]=value if nodes[j] is None else value*nodes[j]
        elif len(nbr)==2:
            j,k=nbr
            left=matrices[0] if weight is None else weight[:,None]*matrices[0]
            value=left.T@matrices[1]
            if j>k:j,k=k,j;value=value.T
            edges[j,k]=value if (j,k) not in edges else value*edges[j,k]
        else:
            tot=0j
            for x in range(n):
                updated=nodes.copy()
                for j,mat in zip(nbr,matrices):
                    updated[j]=mat[x] if nodes[j] is None else mat[x]*nodes[j]
                value=eliminate(updated,edges.copy(),n)
                tot+=value if weight is None else value*weight[x]
            return tot
    value=next(iter(nodes.values()))
    return n if value is None else value.sum()


@lru_cache(None)
def compositions(n,room):
    if not n:return [()]
    return [(j,*rest) for j in range(room+1) for rest in compositions(n-1,room-j)]


def amplitudes(S,v,J,cap):
    # Physical rows J are divided by their nonzero coherent monomers.
    # exp(v.g) prod(1+J_i.g); vacuum weight is handled outside this routine.
    v=v+J.sum(0);B=-.5*J.T@J;op=2*S@B
    powers=[S];log=np.zeros(cap+1,complex)
    for k in range(1,cap+1):
        mat=powers[-1]
        log[k]=np.einsum("ij,ji",B,mat)/k+.5*v@mat@v
        if k<cap:powers.append(op@mat)
    if len(J):
        ev=Evaluation([J@p@J.T for p in powers],[J@p@v for p in powers],cap)
        for (degree,label),coef in graphs(cap):
            log[degree:]+=coef*ev.evaluate(label,cap-degree)
    return np.cumsum(series_exp(log))
