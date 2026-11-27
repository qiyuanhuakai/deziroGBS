"""Incremental bare Gaussian moments for three and four marked detector sites.

The covariance derivatives required after quadratic tilting are Gaussian heat
derivatives of the same polynomial in its mean. At seven pairs only four marks
survive; four-mark polynomials have at most two uncontracted legs.
"""
from functools import lru_cache
import itertools
import math
import numpy as np
from iteration118_clusters import F,sparse_wick,mul


def accumulate(table,key,value):
    if key in table:table[key]+=value
    else:table[key]=np.array(value,complex,copy=True)


def expand_tables(table,n):
    for key,v in list(table.items()):
        if v.ndim:
            dest=np.zeros((n,)*v.ndim,complex)
            dest[(slice(0,n-1),)*v.ndim]=v
            table[key]=dest


def put(table,n,p,old_exp,current_exp,term):
    # term is indexed by all previous sites still carrying a free leg.
    exp=tuple(e for e in old_exp if e)+((current_exp,) if current_exp else ())
    key=(p,exp)
    if key not in table:table[key]=np.zeros((n,)*len(exp),complex)
    if current_exp:table[key][(slice(0,n-1),)*(len(exp)-1)+(n-1,)]+=term
    elif exp:table[key][(slice(0,n-1),)*len(exp)]+=term
    else:table[key]+=term


def power_axis(v,k,axis,rank):
    if not k:return 1.
    shape=[1]*rank;shape[axis]=len(v)
    return (v**k).reshape(shape)


def project(term,exponents):
    if np.ndim(term)==0:return term
    axes=tuple(i for i,k in enumerate(exponents) if k==0)
    return term.sum(axis=axes) if axes else term


@lru_cache(None)
def orders3(K):
    rows=[]
    for orders in itertools.product(range(3,2*K-5),repeat=3):
        if sum(orders)>2*K:continue
        for edges,loops,singles,_,factor in sparse_wick(orders,K):
            rows.append((orders,edges,loops,singles,factor))
    return rows


class BareClusters:
    def __init__(self,cap):
        if cap not in (6,7):raise ValueError("six or seven pairs required")
        self.cap=cap;self.three={};self.four={}

    def append(self,J,S,R):
        n=len(J)
        if n<3:return
        k=J@S@J.T;new=n-1;var=k.diagonal()
        expand_tables(self.four,n)
        if new>=3:
            link=k[:new,new];selfvar=var[new]
            for (p,exp),v in self.three.items():
                for h in range(3,2*self.cap-p+1):
                    for used in itertools.product(*[range(e+1) for e in exp]):
                        available=h-sum(used)
                        if available<0:continue
                        free=tuple(e-b for e,b in zip(exp,used))
                        limit=2*self.cap-p-h-sum(free)
                        if limit<0:continue
                        choices=list(range(available%2,min(limit,available)+1,2))
                        if not choices:continue
                        mult=v
                        for axis,u in enumerate(used):
                            if u:mult=mult*power_axis(link,u,axis,len(exp))
                        reduced=project(mult,free)*math.prod(math.comb(e,b) for e,b in zip(exp,used))
                        for cur in choices:
                            loop=(available-cur)//2
                            scale=F[h]*R[new,h]*selfvar**loop/(F[cur]*F[loop]*2**loop)
                            put(self.four,n,p+h,free,cur,scale*reduced)
        expand_tables(self.three,n)
        # i<j<new. Reuse powers and the factors depending on just one old site.
        oldvar=var[:new];oldlink=k[:new,new]
        power=lambda v:[v**i for i in range(2*self.cap+1)]
        cross=power(k[:new,:new]);diag=power(oldvar);links=power(oldlink)
        mask=np.triu(np.ones((new,new)),1)
        node={};columns={}
        for orders,edges,loops,singles,factor in orders3(self.cap):
            p,q,h=orders;c,g,f=edges;a,b,e=loops
            ik=(p,a,g);jk=(q,b,f)
            if ik not in node:node[ik]=R[:new,p]*diag[a]*links[g]
            if jk not in node:node[jk]=R[:new,q]*diag[b]*links[f]
            key=(ik,jk,c)
            if key not in columns:columns[key]=node[ik][:,None]*node[jk][None,:]*cross[c]*mask
            term=columns[key]*(factor*R[new,h]*var[new]**e)
            put(self.three,n,sum(orders),singles[:2],singles[2],project(term,singles[:2]))

    def effective(self,J,powers):
        cap=self.cap;cache={}
        if not self.three:return cache
        cov=[J@p@J.T for p in powers[1:]]
        # Rebuild only the dependence on the current quadratic tilt. The bare
        # tensors above are independent of this tilt and accumulate incrementally.
        for table in (self.three,self.four):
            for (p,exp),coef in table.items():
                order=(p+sum(exp))//2;extra=cap-order
                if not exp:
                    accumulate(cache,(order,()),coef)
                    continue
                rank=len(exp)
                for edges,loops,singles,_,factor in sparse_wick(exp,sum(exp)):
                    poly=[coef*factor]+[0.]*extra
                    for a,number in enumerate(loops):
                        if not number:continue
                        vv=[power_axis(x.diagonal(),1,a,rank) for x in cov[:extra+1]]
                        for _ in range(number):poly=mul(poly,vv,extra)
                    for (a,b),number in zip(itertools.combinations(range(rank),2),edges):
                        if not number:continue
                        shape=[1]*rank;shape[a]=shape[b]=len(J)
                        vv=[x.reshape(shape) for x in cov[:extra+1]]
                        for _ in range(number):poly=mul(poly,vv,extra)
                    free=tuple(e for e in singles if e)
                    for j,mat in enumerate(poly):
                        if np.ndim(mat)==0 and mat==0:continue
                        accumulate(cache,(order-sum(free)+j,free),project(mat,singles))
        return cache


def contract_cache(cache,all_powers,K,correction):
    for (s,exponents),coeff in cache.items():
        if not exponents:
            correction[s]+=coeff
        elif len(exponents)==1:
            d=exponents[0]
            for a in range(d,K-s+1):correction[s+a]+=coeff@all_powers[d][a]
        elif len(exponents)==2:
            d,e=exponents
            for a in range(d,K-s-e+1):
                proj=coeff.T@all_powers[d][a]
                for b in range(e,K-s-a+1):correction[s+a+b]+=(proj*all_powers[e][b]).sum(0)
        else:
            d,e,f=exponents
            n=len(coeff)
            for c in range(f,K-s-d-e+1):
                z=all_powers[f][c]
                tmp=(coeff.reshape(n*n,n)@z).reshape(n,n,-1)
                for a in range(d,K-s-c-e+1):
                    v=np.einsum("ijz,iz->jz",tmp,all_powers[d][a])
                    for b in range(e,K-s-a-c+1):correction[s+a+b+c]+=(v*all_powers[e][b]).sum(0)
