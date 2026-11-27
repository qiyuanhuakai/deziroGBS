"""State-distance upper bound after resolving conjugate quadratures of a purification."""
import argparse
import json
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from iteration115_displacement_v1 import conditional
from iteration118_auxiliary import OUT,hardware,prepare


def setup(t,s,cap):
    _,v,_,R,_=conditional(t,s)
    _,S,_,_,retained=prepare(t,s,cap)
    r=len(s);eye=np.eye(r);om=np.block([[0*eye,eye],[-eye,0*eye]])
    T=-R.T@om;U=T@R/2
    noise=eye+U@U.T
    V0=noise+T@T.T
    Vp=noise+T@v@T.T
    cp=np.linalg.cholesky(Vp)
    vi=np.linalg.inv(V0)
    # Sample p=cp z under the exact optical purification.
    logp0=.5*(np.linalg.slogdet(Vp)[1]-np.linalg.slogdet(V0)[1])
    adjust=eye-cp.T@vi@cp
    C=T@np.block([[eye,eye],[1j*eye,-1j*eye]])
    H=np.block([[0*eye,eye],[eye,0*eye]])-C.T@vi@C
    l=C.T@vi@cp
    Ax=np.zeros_like(H);Ay=Ax.copy();Ax[:r,:r]=S.conj();Ay[r:,r:]=S
    opx=Ax@H;opy=Ay@H
    # Coefficients of R(x,y)=(I-AH)^(-1) A and log det.
    mats={};tr=np.zeros((cap+1,cap+1),complex)
    quad=np.zeros((cap+1,cap+1,r,r),complex)
    for degree in range(1,2*cap+1):
        prev=mats;mats={}
        for i in range(max(0,degree-cap),min(cap,degree)+1):
            j=degree-i
            if degree==1:coef=Ax if i else Ay
            else:
                coef=0
                if i:coef=coef+opx@prev[i-1,j]
                if j:coef=coef+opy@prev[i,j-1]
            mats[i,j]=coef
            tr[i,j]=np.einsum("ij,ji",coef,H)/(2*degree)
            quad[i,j]=.5*l.T@coef@l
    Tbra=np.linalg.solve(eye-S.conj()@H[:r,:r],S.conj())
    h=H[r:,r:]+H[r:,:r]@Tbra@H[:r,r:]
    leff=l[r:]+H[r:,:r]@Tbra@l[:r]
    op=S@h
    cross_tr=np.zeros(cap+1,complex);cross_q=np.zeros((cap+1,r,r),complex)
    cross_tr[0]=-.5*np.log1p(-np.linalg.eigvals(S.conj()@H[:r,:r])).sum()
    cross_q[0]=.5*l[:r].T@Tbra@l[:r]
    coef=S.copy()
    for i in range(1,cap+1):
        cross_tr[i]=np.einsum("ij,ji",coef,h)/(2*i)
        cross_q[i]=.5*leff.T@coef@leff
        coef=op@coef
    ztrue=np.exp(-.5*np.log1p(-np.linalg.eigvalsh(S@S.conj().T)).sum())
    zcut=ztrue*retained
    return logp0,adjust,tr,quad,cross_tr,cross_q,zcut,ztrue


def evaluate(model,z):
    base,adjust,tr,q,c,mat,zcut,ztrue=model
    K=len(c)-1;b=len(z)
    outer=(z[:,:,None]*z[:,None,:]).reshape(b,-1)
    gaussian=base+.5*outer@adjust.ravel()
    logs=(tr.reshape(-1,1)+q.reshape((K+1)**2,-1)@outer.T).reshape(K+1,K+1,b)
    val=np.zeros_like(logs);val[0,0]=1.
    for n in range(1,2*K+1):
        for i in range(max(0,n-K),min(n,K)+1):
            j=n-i
            val[i,j]=sum((a+b)*logs[a,b]*val[i-a,j-b] for a in range(i+1) for b in range(j+1) if a+b)/n
    qt=val.sum((0,1))*np.exp(gaussian)/zcut
    logs=c[:,None]+mat.reshape(K+1,-1)@outer.T
    val=np.zeros_like(logs);val[0]=1.
    for n in range(1,K+1):val[n]=sum(j*logs[j]*val[n-j] for j in range(1,n+1))/n
    cross=np.exp(gaussian+logs[0])*val.sum(0)/np.sqrt(zcut*ztrue)
    assert abs(qt.imag).max()<2e-9 and qt.real.min()>0
    return qt.real,cross


def run(cap,weak,rows):
    t,ss=hardware();models=[setup(t,s,k) for s,k in zip(ss,(cap,weak))]
    rng=np.random.default_rng(np.random.SeedSequence([118,720,cap,weak]))
    out=[]
    for start in range(0,rows,256):
        n=min(256,rows-start);a=np.ones(n);b=np.ones(n,complex)
        for m in models:
            q,c=evaluate(m,rng.normal(size=(n,len(ss[0]))))
            a*=q;b*=c
        sq=abs(b)**2
        assert np.all(sq<=a+2e-9)
        rad=np.maximum(0,(1+a)**2-4*sq)
        benefit=2*sq/(1+a+np.sqrt(rad))
        assert benefit.min()>=0 and benefit.max()<=1+2e-10
        out.append(np.column_stack((1-benefit,a,b.real,b.imag)))
    values=np.concatenate(out)
    v=values[:,0]
    # Empirical Bernstein bound for bounded iid samples, Maurer & Pontil (2009), Thm 4.
    upper=v.mean()+np.sqrt(2*v.var(ddof=1)*np.log(200)/rows)+7*np.log(200)/(3*(rows-1))
    info=dict(pair_caps=[cap,weak],rows=rows,estimate=float(v.mean()),upper99=float(upper),
              standard_error=float(v.std(ddof=1)/np.sqrt(rows)),mean_likelihood_ratio=float(values[:,1].mean()),
              expected_cross=float(np.prod([np.sqrt(m[-2]/m[-1]) for m in models])),
              observed_cross=float(values[:,2].mean()),mean_cross_imag=float(values[:,3].mean()))
    print(info,flush=True)
    (OUT/f"environment_{cap}_{weak}.json").write_text(json.dumps(info,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("cap",type=int);a.add_argument("weak",type=int);a.add_argument("--rows",type=int,default=32768)
    opts=a.parse_args()
    with threadpool_limits(limits=3):run(opts.cap,opts.weak,opts.rows)
