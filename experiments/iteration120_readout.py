"""Apply the two fitted readout channels to PNR or direct-click records."""
import json
import math
from functools import lru_cache
import numpy as np
from threadpoolctl import threadpool_limits
from iteration114_rank_v1 import cov
from iteration119_hardware import hardware
from iteration120_calibration import OUT


class GaussianPrefix:
    """Diagonal photon-number probabilities from Gaussian Bargmann derivatives."""
    def __init__(self,x):
        n=len(x)//2;eye=np.eye(n);q=np.eye(2*n)+x
        d=.5*np.block([[eye,eye],[-1j*eye,1j*eye]])
        self.matrix=2*d.T@(np.eye(2*n)-np.linalg.inv(q))@d
        self.vacuum=np.exp(-.5*np.linalg.slogdet(q)[1])

    @lru_cache(maxsize=65536)
    def wick(self,indices):
        if not indices:return 1.
        i,rest=indices[0],indices[1:]
        out=0j
        for j in set(rest):
            k=rest.index(j)
            out+=rest.count(j)*self.matrix[i,j]*self.wick(rest[:k]+rest[k+1:])
        return out

    @lru_cache(maxsize=8192)
    def probability(self,counts):
        n=len(counts);indices=tuple(i for i,k in enumerate(counts*2) for _ in range(k))
        p=self.vacuum*self.wick(indices)/math.prod(math.factorial(k) for k in counts)
        if abs(np.imag(p))>1e-10 or np.real(p)<-1e-14:raise ArithmeticError(p)
        return max(0.,float(np.real(p)))


class BankMarginals:
    def __init__(self,t,ss):
        self.t=t;self.ss=ss
        self.local={};self.cdf={}

    def spectral(self,bank,j,rng):
        row=();previous=1.
        for width in range(1,9):
            key=bank,j,width
            if key not in self.local:
                sites=slice(8*bank,8*bank+width)
                self.local[key]=GaussianPrefix(cov(self.t[sites],self.ss[j]))
            kernel=self.local[key]
            target=rng.random()*previous
            for k in range(256):
                word=row+(k,);mass=kernel.probability(word)
                if target<mass:
                    row=word;previous=mass;break
                target-=mass
            else:raise ArithmeticError('PNR conditional mass exhausted')
        return np.asarray(row,dtype=np.uint32)

    def nonempty(self,bank,direct,rng):
        if direct:
            if bank not in self.cdf:
                # In these small local coordinates the diagonal axes use offset 8.
                X=[cov(self.t[8*bank:8*bank+8],s) for s in self.ss]
                law=local_clicks(X)
                assert law.min()>=0
                self.cdf[bank]=np.cumsum(law[1:])/law[1:].sum()
            word=1+np.searchsorted(self.cdf[bank],rng.random())
            if word>255:raise ArithmeticError('click conditional mass exhausted')
            return (word>>np.arange(8))&1
        while True:
            row=self.spectral(bank,0,rng)+self.spectral(bank,1,rng)
            if row.any():return row


def local_clicks(X):
    vacancy=[]
    for mask in range(256):
        ix=np.flatnonzero((mask>>np.arange(8))&1);axes=np.r_[ix,8+ix]
        vacancy.append(np.prod([np.exp(-.5*np.linalg.slogdet(np.eye(len(axes))+s[np.ix_(axes,axes)])[1]) for s in X]))
    law=np.array(vacancy[::-1])
    for i in range(8):
        view=law.reshape(-1,2,1<<i);view[:,1]-=view[:,0]
    return law


def checks():
    err=0.;n=.36;q=GaussianPrefix(n*np.eye(4))
    for a in range(5):
        for b in range(4):err=max(err,abs(q.probability((a,b))-n**(a+b)/(1+n)**(a+b+2)))
    s=.61;matrix=np.diag([np.exp(-2*s),np.exp(2*s)])
    q=GaussianPrefix((matrix-np.eye(2))/2)
    for k in range(13):
        # Odd total photon numbers vanish exactly for the zero-displacement squeezed vacuum.
        exact=0 if k%2 else math.comb(k,k//2)*(np.tanh(s)/2)**k/np.cosh(s)
        err=max(err,abs(q.probability((k,))-exact))
    # Correlated two-mode squeezed vacuum. Each number in one arm fixes the other.
    s=.48;c=np.cosh(2*s);d=np.sinh(2*s)
    vv=np.array([[c,d,0,0],[d,c,0,0],[0,0,c,-d],[0,0,-d,c]])
    q=GaussianPrefix((vv-np.eye(4))/2)
    for i in range(7):
        for j in range(7):
            expected=np.tanh(s)**(2*i)/np.cosh(s)**2 if i==j else 0
            err=max(err,abs(q.probability((i,j))-expected))
    assert err<1e-14,err
    return err


def run(n):
    err=checks()
    with np.load(OUT/'response_predictions.npz') as pred:
        empty=pred['empty_survival'];rate=pred['poisson_mean_per_port'];nominal=pred['published_multiplicity']
    t,ss=hardware();bank_draw=BankMarginals(t,ss)
    # Independently check local FP64 click probabilities and vacuum/one-photon probabilities.
    for i in range(144):
        X=[cov(t[8*i:8*i+8],s) for s in ss];law=local_clicks(X)
        assert min(law)>-1e-13 and abs(law.sum()-1)<1e-13
        assert abs(law[0]-nominal[i,0])<1e-12
        vac=[GaussianPrefix(x).probability((0,)*8) for x in X]
        assert abs(np.prod(vac)-law[0])<1e-12
    arrays={};stats=[]
    for direct,name in enumerate(('pnr','click')):
        with np.load(OUT/f'full_{direct}_{n}.npz') as f:
            records=f[name];ids=f['accepted_indices']
        for mechanism in ('poisson','refill'):
            output=records.astype(np.uint32,copy=True).reshape(-1,144,8)
            changes=0
            for row,index in zip(output,ids):
                rng=np.random.default_rng([120,137,direct,int(mechanism=='refill'),int(index)])
                if mechanism=='poisson':
                    if direct:extra=rng.random((144,8))<(-np.expm1(-rate))[:,None];row|=extra
                    else:extra=rng.poisson(rate[:,None],(144,8));row+=extra.astype(np.uint32)
                    changes+=np.count_nonzero(extra.any(1))
                else:
                    add=(~row.any(1))&(rng.random(144)>=empty)
                    for i in np.flatnonzero(add):row[i]=bank_draw.nonempty(int(i),direct,rng)
                    changes+=np.count_nonzero(add)
            result=output.reshape(-1,1152)
            if direct:assert result.max()==1
            assert np.all(result>=records)
            arrays[f'{mechanism}_{name}']=result
            stats.append(dict(readout=name,mechanism=mechanism,rows=len(records),nonempty_added_banks_or_background_banks=int(changes)))
    np.savez_compressed(OUT/'response_samples.npz',**arrays)
    (OUT/'response_samples.json').write_text(json.dumps(dict(toy_max_error=err,streams=stats),indent=2)+'\n')
    print(stats,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):run(256)
