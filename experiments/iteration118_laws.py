"""Independent Gaussian determinant series for arbitrary pair cutoffs."""
import argparse
import json
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from iteration115_projection_review_v3 import factor
from iteration117_quartic_laws import marginal,physical_models,contraction
from iteration118_auxiliary import OUT,prepare,hardware


def factor_polynomial(S,t,metric,cap):
    r=len(S);v=.5*np.vstack((t,-1j*t));a=v.conj();av=a.T@metric;bv=v.T@metric
    A=-2*S.conj()@(av@a);B=S.conj()@(np.eye(r)-2*av@v)
    C=S@(np.eye(r)-2*bv@a);D=-2*S@(bv@v)
    # Trace powers of diag(x,y)H, followed by formal exponentiation in both variables.
    U=np.column_stack((A,B));V=np.column_stack((C,D))
    prev={(0,0):np.eye(2*r,dtype=complex)}
    log=np.zeros((cap+1,cap+1),complex)
    for n in range(1,2*cap+1):
        cur={}
        for i in range(max(0,n-cap),min(n,cap)+1):
            j=n-i
            top=U@prev[i-1,j] if i else np.zeros_like(U)
            bottom=V@prev[i,j-1] if j else np.zeros_like(V)
            cur[i,j]=np.vstack((top,bottom))
            log[i,j]=(np.trace(top[:,:r])+np.trace(bottom[:,r:]))/(2*n)
        prev=cur
    val=np.zeros_like(log);val[0,0]=1.
    for n in range(1,2*cap+1):
        for i in range(max(0,n-cap),min(n,cap)+1):
            j=n-i
            val[i,j]=sum((a+b)*log[a,b]*val[i-a,j-b] for a in range(i+1) for b in range(j+1)
                         if a+b)/(i+j)
    return val


def run(cap,weak):
    t,ss=hardware();mods=physical_models();sizes=[cap,weak]
    z=np.exp(-2j*np.pi*np.arange(257)/512)
    curve=np.ones(len(z),complex);g=1.
    for model,k in zip(mods,sizes):
        S,_,transfer,x=marginal(model,None)
        e,u=eigh(x)
        norm=factor_polynomial(S,transfer,np.zeros_like(x),k).sum().real
        for idx,node in enumerate([*z,1.08]):
            w=1-node;den=1+w*e
            metric=(u*(w/den))@u.T;vac=np.exp(-.5*np.log(den).sum())
            f=factor_polynomial(S,transfer,metric,k)
            if idx==5:
                v=contraction(S,model[3],transfer,metric,vac)
                np.testing.assert_allclose(v,vac*f[:3,:3].sum()/model[3],atol=3e-13,rtol=0)
            pgf=vac*f.sum()/norm
            if idx<len(z):curve[idx]*=pgf
            else:g*=pgf.real
    target=np.ones(len(z),complex);gref=1.
    for _,e in factor(t,ss):
        target*=np.exp(-.5*np.log1p((1-z[:,None])*e).sum(1))
        gref*=np.exp(-.5*np.log1p(-.08*e).sum())
    p=np.fft.irfft(curve,n=512);q=np.fft.irfft(target,n=512)
    assert abs(p.sum()-1)<2e-12 and p.min()>-1e-12
    tv=float(abs(p-q).sum()/2);tail=float((g+gref)*1.08**-512)
    info=dict(pair_caps=sizes,total_number_tv_lower=tv,total_number_tv_upper=tv+tail,
              modulo_alias_bound=tail,mean_photons=float(p@np.arange(512)),target_mean=float(q@np.arange(512)))
    name=f"law_{cap}_{weak}"
    np.savez_compressed(OUT/(name+".npz"),model=p,optical=q)
    (OUT/(name+".json")).write_text(json.dumps(info,indent=2)+"\n",encoding="utf8")
    print(info,flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("cap",type=int);a.add_argument("weak",type=int);opts=a.parse_args()
    with threadpool_limits(limits=2):run(opts.cap,opts.weak)
