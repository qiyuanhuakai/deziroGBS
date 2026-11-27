"""Independent Fock-space and numerical checks of auxiliary readout."""
import itertools
import json
from time import perf_counter
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
from iteration118_auxiliary import History,FAC,OUT,refine


def local_test(K,m=4):
    r=3;D=2*K
    rng=np.random.default_rng(118020)
    q=np.linalg.qr(rng.normal(size=(m,r))+1j*rng.normal(size=(m,r)))[0]
    u=np.linalg.qr(rng.normal(size=(r,r))+1j*rng.normal(size=(r,r)))[0]
    S=(u*np.array([.53,.31,.18]))@u.T
    pair=q@S@q.T
    # Ordinary Bargmann monomials. Repeated multiplication by the quadratic gives
    # every exact Fock coefficient through this cutoff without formal Gaussian contractions.
    zero=(0,)*m;pol={zero:1.};power={zero:1.}
    for degree in range(1,K+1):
        nextp={}
        for occ,a in power.items():
            for i in range(m):
                for j in range(i+1):
                    k=list(occ);k[i]+=1;k[j]+=1;k=tuple(k)
                    nextp[k]=nextp.get(k,0.)+a*(.5 if i==j else 1.)*pair[i,j]/degree
        power=nextp
        pol.update(power)
    beta=rng.normal(size=m)+1j*rng.normal(size=m)
    alpha=np.array([.51-.3j,.7+.1j,-.24+.2j,.63j,.78-.24j,.42+.57j])[:m]
    err=0;checked=0
    for path in ((0,1,2),(1,2,1),(2,2,2),(3,1,0)):
        hist=History(S,K);v=beta@q;past=[]
        for i,row in enumerate(q):
            v-=beta[i]*row
            local=hist.ket(v,row)
            truth=np.zeros(D+1,complex)
            for occ,a in pol.items():
                term=a*np.sqrt(FAC[occ[i]])
                for j,f in enumerate(past):term*=f[occ[j]]*np.sqrt(FAC[occ[j]])
                for j in range(i+1,m):term*=beta[j]**occ[j]
                truth[occ[i]]+=term
            # Previous normalized local bras omit the common product of their constants.
            scale=np.prod([f[0] for f in past])
            err=max(err,float(abs(local*scale-truth).max()));checked+=1
            if i==m-1:break
            k=path[i%len(path)]
            a=alpha[i];vac=np.array([(-a.conjugate())**n/np.sqrt(FAC[n]) for n in range(D+1)])*np.exp(-abs(a)**2/2)
            if k==0:
                f=vac;v-=a.conjugate()*row
            elif k==2:
                f=-np.conj(vac[1])*vac/(1+np.sqrt(max(0,1-np.vdot(vac,vac).real)));f[1]+=1
                v+=hist.update(row,f)*row
            else:
                create=np.diag(np.sqrt(np.arange(1,24)),-1)
                f=expm(a*create-a.conj()*create.T)[k,:D+1]
                v+=hist.update(row,f)*row
            past.append(f)
    assert err<3e-11,(K,err)
    return dict(modes=m,pair_cap=K,local_kets=checked,max_amplitude_error=err)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        rows=[local_test(k) for k in (2,3,4,5,6,7)] + [local_test(k,5) for k in (6,7)]
        OUT.mkdir(exist_ok=True)
        (OUT/"local_test.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf8")
        print(rows,flush=True)
