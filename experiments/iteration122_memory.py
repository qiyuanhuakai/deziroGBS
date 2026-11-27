"""Reference finite-Fock memory sampler at arbitrary pair cutoff (small source ranks)."""
from itertools import accumulate
from math import factorial,lgamma
from time import perf_counter
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from iteration122_loss import OUT,cov,physical_total,json_write


def compositions(k,r):
    if r==1:
        yield (k,)
    else:
        for j in range(k+1):
            for v in compositions(k-j,r-1):
                yield (j,*v)


class Layout:
    def __init__(self,r,degree):
        self.states=[list(compositions(n,r)) for n in range(degree+1)]
        self.index=[{v:i for i,v in enumerate(layer)} for layer in self.states]
        self.offset=list(accumulate([0]+[len(layer) for layer in self.states]))


def passive(u,left,right):
    blocks=[np.ones((1,1),complex)]
    for n in range(1,len(left.states)):
        old=np.array(right.states[n-1])
        block=np.zeros((len(right.states[n]),len(left.states[n])),complex)
        dest=[];weights=[]
        for h in range(u.shape[0]):
            lifted=old.copy();lifted[:,h]+=1
            dest.append([right.index[n][tuple(v)] for v in lifted])
            weights.append(np.sqrt(lifted[:,h]))
        for i,occ in enumerate(left.states[n]):
            axis=next(k for k,v in enumerate(occ) if v)
            parent=list(occ);parent[axis]-=1
            coeff=blocks[-1][:,left.index[n-1][tuple(parent)]]/np.sqrt(occ[axis])
            for j,amp in enumerate(u[:,axis]):
                block[dest[j],i]+=amp*weights[j]*coeff
        blocks.append(block)
    return blocks


def local_pnr(gram,alpha,rng):
    L=len(gram)-1
    u=abs(alpha)**2
    trace=gram.trace().real
    roots=np.sqrt(np.arange(L+1))
    if u<4*(L+1):
        row=np.empty(L+1,complex);row[0]=np.exp(-u/2)
        for j in range(1,L+1):row[j]=-alpha.conjugate()*row[j-1]/roots[j]
        level=rng.random()*trace;n=0
        while True:
            mass=np.vdot(row,gram@row).real
            if level<mass:return n,row/np.sqrt(mass)
            level-=mass;n+=1;old=row.copy()
            row[0]=old[0]*alpha/np.sqrt(n)
            for j in range(1,L+1):
                row[j]=(np.sqrt(n)*old[j-1]-alpha.conjugate()*row[j-1])/roots[j]
            if not row.any():raise ArithmeticError('number recurrence exhausted')
    # This practical envelope adapts a single ring to the conditional diagonal.
    # The proof in the paper also gives a polynomial-in-degree mixture of rings.
    a=np.sqrt(u);j=np.arange(L+1);lf=np.array([lgamma(k+1) for k in j])
    radii=np.array([.125,.5,1.25,np.exp(lgamma(L+1)/L)]) if L else np.array([.125])
    pmf=np.exp(-radii[:,None]+j*np.log(radii[:,None])-lf)
    bounds=np.minimum((np.maximum(gram.diagonal().real,0)/pmf).sum(1),trace/pmf.min(1));idx=np.argmin(bounds)
    x=radii[idx];arcs=max(8,int(np.ceil(4*np.sqrt(x))))
    d=(x+2*a*np.sqrt(x)*np.cos(np.pi*np.arange(arcs+1)/arcs))/u
    means=u*(1+d);slope=np.log1p(d)
    rem=slope-d;small=abs(d)<.001
    rem[small]=sum((-1)**(k+1)*d[small]**k/k for k in range(2,9))
    envelope=2*bounds[idx]*np.exp(np.max(means[:-1]*np.diff(slope)**2)/8)*(1+2e-12)
    phase=(alpha.conjugate()/a)**j
    while True:
        pos=int(rng.integers(2*arcs));n=int(rng.poisson(means[pos//2+pos%2]))
        p=np.ones(L+1);z=(n-u)/a
        if L:p[1]=z
        for k in range(1,L):
            p[k+1]=((z-k/a)*p[k]-roots[k]*p[k-1])/roots[k+1]
        row=phase*p;mass=np.vdot(row,gram@row).real
        ratio=np.exp((n-u)*slope+u*rem)
        rel=(.5*(ratio[0]+ratio[-1])+ratio[1:-1].sum())/arcs
        prob=mass/(envelope*rel)
        if not -1e-10<=prob<=1+1e-10:raise ArithmeticError('invalid Poisson envelope')
        if rng.random()<prob:return n,row/np.sqrt(mass)


def local_click(gram,alpha,rng):
    L=len(gram)-1
    v=np.empty(L+1,complex);v[0]=np.exp(-abs(alpha)**2/2)
    for k in range(1,L+1):v[k]=-alpha.conjugate()*v[k-1]/np.sqrt(k)
    den=1+np.sqrt(max(0.,1-np.vdot(v,v).real))
    rows=np.vstack([v,np.eye(L+1)-np.outer(v.conj(),v)/den])
    masses=np.einsum('ij,ij->i',rows.conj(),rows@gram.T).real
    if masses.min() < -2e-10:raise ArithmeticError('negative click mass')
    c=np.cumsum(masses.clip(0));i=np.searchsorted(c,rng.random()*c[-1],side='right')
    return int(i>0),rows[i]/np.sqrt(masses[i])


class Memory:
    def __init__(self,t,s,eta,cap):
        r=len(s);self.degree=2*cap
        left=self.left=Layout(r,self.degree)
        right=Layout(r+1,self.degree)
        a=1-eta+eta*np.exp(-2*s)
        pair=(a-1)/(a+1)
        self.field=(t/np.sqrt(eta)*(1j*np.sinh(s)*np.sqrt(eta*(1-eta)/a))).T
        f=np.zeros(left.offset[-1],complex)
        for n,layer in enumerate(left.states):
            for i,v in enumerate(layer):
                if not any(k%2 for k in v):
                    f[left.offset[n]+i]=np.prod([p**(k//2)*np.sqrt(float(factorial(k)))/(2**(k//2)*factorial(k//2)) for p,k in zip(pair,v)])
        self.initial=f/np.linalg.norm(f)
        q=t/np.sqrt(eta)
        eig,u=eigh(np.eye(r)-q.conj().T@q)
        if eig[0] < -2e-12:raise ValueError('transfer exceeds envelope')
        R=(u*np.sqrt(eig.clip(0)))@u.conj().T
        self.maps=[]
        for i in reversed(range(len(q))):
            V,R=np.linalg.qr(np.vstack([q[i],R]))
            if i==0:
                np.testing.assert_allclose(R.conj().T@R,np.eye(r),atol=2e-12)
                V=V@R
            self.maps.append(passive(V,left,right))
        self.maps.reverse()

    def draw(self,n,route,rng):
        state=np.tile(self.initial,(n,1))
        alpha=rng.standard_normal((n,len(self.field)))@self.field
        off=self.left.offset;L=self.degree
        result=np.empty((n,len(self.maps)),np.uint64)
        for site,blocks in enumerate(self.maps):
            Y=np.zeros((n,L+1,len(self.initial)),complex)
            for k,B in enumerate(blocks):
                values=state[:,off[k]:off[k+1]]@B.T
                pos=0
                for j in range(k+1):
                    count=off[k-j+1]-off[k-j]
                    Y[:,j,off[k-j]:off[k-j+1]]=values[:,pos:pos+count]
                    pos+=count
            gram=Y.conj()@Y.transpose(0,2,1)
            for i in range(n):
                value,row=(local_click if route else local_pnr)(gram[i],alpha[i,site],rng)
                result[i,site]=value;state[i]=row@Y[i]
            if max(abs(np.sum(abs(state)**2,axis=1)-1))>2e-9:raise ArithmeticError('conditional state lost normalization')
        return result


def run():
    rank,m,eta,cap=2,6,.51,15
    rng=np.random.default_rng([122,800])
    u=np.linalg.qr(rng.standard_normal((m,rank))+1j*rng.standard_normal((m,rank)))[0]
    v=np.linalg.qr(rng.standard_normal((rank,rank))+1j*rng.standard_normal((rank,rank)))[0]
    t=(u*np.sqrt([.26,.48]))@v
    ss=np.array([[2.5,2.15],[1.5,1.1]])
    start=perf_counter();engines=[Memory(t,s,eta,cap) for s in ss]
    setup=perf_counter()-start
    shots=1024;arrays=dict(T=t,squeezing=ss);width=1<<m
    empty=np.ones(width)
    for word in range(1,width):
        idx=np.flatnonzero((word>>np.arange(m))&1)
        for s in ss:
            x=cov(t[idx],s)
            empty[word]*=np.exp(-.5*np.linalg.slogdet(np.eye(len(x))+x)[1])
    optical=empty[::-1].copy()
    for j in range(m):
        a=optical.reshape(-1,2,1<<j);a[:,1]-=a[:,0]
    stats=[];arrays['optical_clicks']=optical
    for route in (0,1):
        start=perf_counter();rngs=[np.random.default_rng([122,810,route,j]) for j in range(2)]
        rows=[]
        for _ in range(shots//8):
            a,b=[eng.draw(8,route,generator) for eng,generator in zip(engines,rngs)]
            rows.append((a|b) if route else (a+b))
        x=np.vstack(rows);duration=perf_counter()-start
        name='click' if route else 'pnr';arrays[name]=x
        counts=np.bincount((x>0)@(1<<np.arange(m)),minlength=width)
        stats.append(dict(route=name,seconds_per_row=duration/shots,rows=shots,complete_click_empirical_tv=float(abs(counts/shots-optical).sum()/2)))
    np.savez_compressed(OUT/'memory.npz',**arrays)
    info=dict(ports=m,rank=rank,eta=eta,cap=cap,setup_seconds=setup,readouts=stats)
    json_write(OUT/'memory.json',info)
    print(info,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):run()
