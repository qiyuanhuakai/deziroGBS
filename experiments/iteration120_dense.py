"""Batched conditioning for the small non-tree tensor networks at high order."""
from functools import lru_cache
from itertools import combinations
import numpy as np
import torch


def neighbors(n,edges):
    adj={i:[] for i in n}
    for a,b in edges:adj[a].append(b);adj[b].append(a)
    return adj


@lru_cache(None)
def plan(count,links):
    # Condition as few sites as needed, then eliminate only leaves and paths.
    sites=tuple(range(count))
    for size in range(count):
        for fixed in combinations(sites,size):
            free=set(sites)-set(fixed)
            edges={e for e in links if all(i in free for i in e)}
            order=[]
            while len(free)>1:
                adj=neighbors(free,edges)
                site=min(free,key=lambda i:(len(adj[i]),i))
                rest=adj[site]
                if len(rest)>2:break
                order.append((site,tuple(rest)));free.remove(site)
                edges={e for e in edges if site not in e}
                if len(rest)==2:edges.add(tuple(sorted(rest)))
            if len(free)==1:return fixed,order,next(iter(free))
    raise ValueError('no conditioning set')


class Dense:
    def __init__(self,device):
        self.device=device
        self.cache={}

    def array(self,x):
        if x is None:return None
        key=id(x)
        if key not in self.cache:self.cache[key]=(x,torch.as_tensor(np.ascontiguousarray(x),device=self.device))
        return self.cache[key][1]

    @staticmethod
    def times(a,b):
        return b if a is None else a if b is None else a*b

    def evaluate(self,links,count,arguments,n):
        fixed,order,final=plan(count,links)
        v=[self.array(a) for a in arguments[:count]]
        edges={e:self.array(a) for e,a in zip(links,arguments[count:])}
        free=set(range(count))-set(fixed)
        all_ids=torch.arange(n**len(fixed),device=self.device)
        total=0.
        # Allow large batches on the GPU and bound peak scratch space.
        step=max(1,min(len(all_ids),2_000_000//(n*n)))
        for ids in all_ids.split(step):
            ix={i:(ids//(n**j))%n for j,i in enumerate(fixed)}
            node={i:(v[i][None,:] if v[i] is not None else None) for i in free}
            local={};scale=None
            for i in fixed:
                if v[i] is not None:scale=self.times(scale,v[i][ix[i]])
            for (i,j),mat in edges.items():
                if i in ix and j in ix:scale=self.times(scale,mat[ix[i],ix[j]])
                elif i in ix:node[j]=self.times(node[j],mat[ix[i]])
                elif j in ix:node[i]=self.times(node[i],mat.T[ix[j]])
                else:local[i,j]=mat[None,:,:]
            for i,rest in order:
                w=node.pop(i);mats=[]
                for j in rest:
                    a=local.pop(tuple(sorted((i,j))))
                    mats.append(a if i<j else a.transpose(-2,-1))
                if len(rest)==1:
                    j=rest[0]
                    val=mats[0].sum(-2) if w is None else (w.unsqueeze(1)@mats[0]).squeeze(1)
                    node[j]=self.times(node[j],val)
                elif len(rest)==2:
                    j,k=rest;left,right=mats
                    val=left.transpose(-2,-1)@(right if w is None else right*w.unsqueeze(-1))
                    if j>k:j,k=k,j;val=val.transpose(-2,-1)
                    local[j,k]=self.times(local.get((j,k)),val)
                else:
                    scale=self.times(scale,n if w is None else w.sum(-1))
            val=n if node[final] is None else node[final].sum(-1)
            val=self.times(scale,val)
            total=total+(val.expand(len(ids)).sum() if isinstance(val,torch.Tensor) else val*len(ids))
        return total
