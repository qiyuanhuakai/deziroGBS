"""Sequential replacement of virtual heterodynes for a finite-pair Gaussian ket."""
import itertools
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import i0e
from threadpoolctl import threadpool_limits

from iteration118_clusters import two_sites,Triple
from iteration118_high_clusters import BareClusters,accumulate,contract_cache
from iteration115_displacement_v1 import conditional
from iteration114_pairs_v1 import sector
from iteration115_visible_v1 import DATA,hardware

OUT=DATA/"iteration118_auxiliary"
FAC=np.array([math.factorial(i) for i in range(17)],float)


def series_exp(log):
    ans=np.ones_like(log)
    for j in range(1,len(log)):
        ans[j]=sum(k*log[k]*ans[j-k] for k in range(1,j+1))/j
    return ans


def series_mul(a,b,cap):
    return np.array([sum(a[i]*b[n-i] for i in range(n+1)) for n in range(cap+1)])


def prepare(t,s,K):
    q,v,_,response,_=conditional(t,s)
    r=len(s);eye=np.eye(r);j=np.vstack((eye,1j*eye))/np.sqrt(2)
    S=j.T@(np.eye(2*r)-np.linalg.inv((v+np.eye(2*r))/2))@j
    retained,_=sector(.5*np.log(np.linalg.eigvalsh(v)[r:]),2*K)
    fields=q@(response[:r]+1j*response[r:])/2
    return q,S,np.linalg.cholesky(v+np.eye(2*r)),fields,retained


def virtual(q,S,chol,K,rng):
    m,r=q.shape;attempts=0
    while True:
        attempts+=1
        z=chol@rng.standard_normal(2*r)
        a=(z[:r]+1j*z[r:])*.5*np.exp(1j*rng.uniform(0,2*np.pi))
        T=a.conj()@S@a.conj()/2
        poly=sum(T**j/FAC[j] for j in range(K+1))
        u=2*abs(T)
        prob=abs(poly)**2*np.exp(-u)/((K+1)*i0e(u))
        assert 0<=prob<=1+1e-12
        if rng.random()<prob:break
    vac=(rng.standard_normal(m)+1j*rng.standard_normal(m))/np.sqrt(2)
    return vac+q@(a-q.conj().T@vac),attempts


def refine(ket,alpha,direct,rng):
    # ket encodes the physical local Fock amplitudes before displacement.
    length=len(ket);f=np.ones(length,complex)
    f[0]=np.exp(-abs(alpha)**2/2)
    for k in range(1,length):f[k]=-alpha.conjugate()*f[k-1]/np.sqrt(k)
    vacuum=f.copy()
    total=np.vdot(ket,ket).real;target=rng.random()*total
    prob=abs(f@ket)**2
    if target<prob:return 0,f
    target-=prob
    if direct:
        den=1+np.sqrt(max(0,1-np.vdot(vacuum,vacuum).real))
        for k in range(length):
            f=-vacuum[k].conj()*vacuum/den
            f[k]+=1
            prob=abs(f@ket)**2
            if target<prob:return 1,f
            target-=prob
    else:
        for n in range(1,65536):
            prev=f.copy();f[0]=prev[0]*alpha/np.sqrt(n)
            for k in range(1,length):
                f[k]=(np.sqrt(n)*prev[k-1]-alpha.conjugate()*f[k-1])/np.sqrt(k)
            prob=abs(f@ket)**2
            if target<prob:return n,f
            target-=prob
            if n>48+4*abs(alpha)**2 and prob<1e-30:break
    raise ArithmeticError(("measurement resolution exhausted",target,total))


def pair_terms(cap,J,S1,S2,residual):
    r=len(S1)
    const=np.zeros(cap+1,complex);linear=np.zeros(r,complex);quad=np.zeros((r,r),complex)
    if len(J)<2:return const,linear,quad
    x=J@S1;cov=x@J.T;var=cov.diagonal()
    second=J@S2@J.T;v2=second.diagonal()
    for p in range(3,2*cap-2):
        for k in range(3,2*cap-p+1):
            coeff=np.triu(residual[:,p,None]*residual[None,:,k],1)
            for c in range(min(p,k)+1):
                for a in range((p-c)//2+1):
                    for b in range((k-c)//2+1):
                        d=p-c-2*a;e=k-c-2*b
                        order=a+b+c+d+e
                        if order>cap:continue
                        factor=FAC[p]*FAC[k]/(FAC[d]*FAC[e]*FAC[c]*2**(a+b)*FAC[a]*FAC[b])
                        term=factor*coeff*(var[:,None]**a)*(var[None,:]**b)*cov**c
                        if d+e==0:
                            const[order]+=term.sum()
                            if order<cap:
                                assert order+1==cap
                                # Differentiate products explicitly, including zeros of a covariance.
                                extra=np.zeros_like(term)
                                if a:extra+=a*var[:,None]**(a-1)*v2[:,None]*var[None,:]**b*cov**c
                                if b:extra+=b*var[:,None]**a*var[None,:]**(b-1)*v2[None,:]*cov**c
                                if c:extra+=c*var[:,None]**a*var[None,:]**b*cov**(c-1)*second
                                const[order+1]+=(factor*coeff*extra).sum()
                        elif d+e==1:
                            assert order==cap
                            linear+=(term.sum(1) if d else term.sum(0))@x
                        else:
                            assert order==cap and d+e==2
                            if d==1:quad+=x.T@term@x
                            else:quad+=x.T@((term.sum(1) if d else term.sum(0))[:,None]*x)
    return const,linear,quad


class History:
    def __init__(self,S,K):
        if K not in range(2,8):raise ValueError("supported pair caps are 2 through 7")
        self.S=S;self.K=K;self.r=len(S);self.D=2*K
        self.B=np.zeros_like(S);self.rows=[];self.res=[];self.triple=Triple();self.bare=BareClusters(K) if K>=6 else None
        # Sample the local polynomial on a circle; FFT recovers all coefficients.
        self.radius=2.5 if K>=6 else 1.6
        self.grid=self.radius*np.exp(2j*np.pi*np.arange(self.D+1)/(self.D+1))
        self.rebuild()

    def rebuild(self):
        K=self.K
        J=np.array(self.rows,dtype=complex).reshape(-1,self.r)
        self.R=np.array(self.res,dtype=complex).reshape(-1,self.D+1)
        op=2*self.S@self.B
        powers=[self.S];logs=[0j]
        for n in range(1,K+1):
            logs.append(np.einsum("ij,ji",self.B,powers[-1])/n)
            if n<K:powers.append(op@powers[-1])
        self.powers=np.array(powers)
        self.trace=np.array(logs)
        self.map=J@self.powers
        var=np.einsum("nhi,hi->nh",self.map,J)
        if K<=4:
            self.pair,self.linear,self.quad=pair_terms(K,J,powers[0],powers[1],self.R)
        else:
            self.cache=two_sites(K,J,powers,self.R)
            self.pair=np.zeros(K+1,complex);self.linear=np.zeros(self.r,complex);self.quad=np.zeros_like(self.S)
            if K==5:
                self.triple.append(J,self.S,self.R)
                self.pair[K]=self.triple.constant
                if self.triple.linear is not None:self.linear=self.triple.linear
            else:
                self.bare.append(J,self.S,self.R)
                self.many=self.bare.effective(J,powers)
                for (d,e,degree),v in self.cache.items():
                    accumulate(self.many,(degree,tuple(n for n in (d,e) if n)),v)
        # Single-site correction as powers of its tilted Gaussian mean.
        # coeff[p,h] multiplies M_h(t)**p; precontract all powers of the tilted variance.
        self.single=np.zeros((K+1,K+1,len(J)),complex)
        V=np.vstack((np.zeros(len(J)),var))
        prev=np.zeros_like(V);prev[0]=1
        for a in range(K+1):
            for p in range(K-a+1):
                n=2*a+p
                if 3<=n<=self.D:
                    self.single[p]+=FAC[n]/(2**a*FAC[a]*FAC[p])*prev*self.R[:,n]
            if a<K:prev=series_mul(prev,V,K)

    def update(self,row,f):
        # Remove an irrelevant common scalar. a rarely small constant can magnify roundoff.
        if abs(f[0])<1e-15:raise ArithmeticError("zero constant in the local refinement")
        f=f/f[0]/np.sqrt(FAC[:self.D+1])
        a=f[1];b=f[2]-a*a/2
        coeff=np.zeros(self.D+1,complex);coeff[1]=-a;coeff[2]=-b
        residual=series_mul(f,series_exp(coeff),self.D)
        self.rows.append(row);self.res.append(residual)
        self.B+=b*np.outer(row,row)
        self.rebuild()
        return a

    def ket(self,v,row,first_pair=0):
        K=self.K;z=self.grid
        pv=self.powers@v;pw=self.powers@row
        log=np.zeros((K+1,len(z)),complex)
        log[1:]=self.trace[1:,None]+.5*((pv@v)[:,None]+2*(pv@row)[:,None]*z+(pw@row)[:,None]*z*z)
        base=series_exp(log)
        correction=np.repeat(self.pair[:,None],len(z),axis=1)
        correction[0]+=1
        if self.rows:
            V=v[None,:]+z[:,None]*row
            correction[K]+=V@self.linear+np.einsum("zi,ij,zj->z",V,self.quad,V,optimize=True)
            mu=np.zeros((K+1,len(self.rows),len(z)),complex)
            mu[1:]=(self.map@v)[:,:,None]+(self.map@row)[:,:,None]*z
            power=np.zeros_like(mu);power[0]=1
            all_powers=[]
            for p in range(K+1):
                all_powers.append(power)
                for n in range(p,K+1):
                    for j in range(n-p+1):
                        correction[n]+=self.single[p,j]@power[n-j]
                if p<K:power=series_mul(power,mu,K)
            if K>=6:contract_cache(self.many,all_powers,K,correction)
            if K==5:
                for (d,e,s),coeff in self.cache.items():
                    if not d and not e:correction[s]+=coeff;continue
                    if not d or not e:
                        for a in range(d+e,K-s+1):correction[s+a]+=coeff@all_powers[d+e][a]
                    else:
                        for a in range(d,K-s-e+1):
                            proj=coeff.T@all_powers[d][a]
                            for b in range(e,K-s-a+1):correction[s+a+b]+=(proj*all_powers[e][b]).sum(0)
        result=sum(base[n]*correction[j] for n in range(K+1) for j in range(max(0,first_pair-n),K+1-n))
        coef=np.fft.fft(result)/len(z)
        return coef*np.sqrt(FAC[:self.D+1])/self.radius**np.arange(self.D+1)


def draw(model,K,direct,seed,diagnostics=None):
    q,S,chol,field,_=model
    rv,ra,rm=(np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))
    beta,tries=virtual(q,S,chol,K,rv);beta=beta.conj()
    alpha=field@ra.standard_normal(field.shape[1])
    v=beta@q;h=History(S,K);bits=[];previous=None;worst=0.;worst_relative=0.
    for i,row in enumerate(q):
        v-=beta[i]*row
        ket=h.ket(v,row)
        if diagnostics is not None and previous is not None:
            value=ket@(beta[i]**np.arange(len(ket))/np.sqrt(FAC[:len(ket)]))
            scale=np.linalg.norm(ket)*np.exp(abs(beta[i])**2/2)
            worst=max(worst,abs(value-previous)/max(abs(previous),scale,1e-280))
            worst_relative=max(worst_relative,abs(value-previous)/max(abs(previous),1e-280))
        outcome,f=refine(ket,alpha[i],direct,rm)
        if diagnostics is not None:previous=(f@ket)/f[0]
        if outcome:v+=h.update(row,f)*row
        else:v-=alpha[i].conjugate()*row
        bits.append(outcome)
    if diagnostics is not None:diagnostics.update(prefix_residual=float(worst),relative_to_amplitude=float(worst_relative))
    return np.array(bits,np.uint16),tries
