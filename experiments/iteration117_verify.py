"""Independent Fock/oscillator quadrature check and finite-sample diagnostics."""
from itertools import combinations_with_replacement
import json
import math
import numpy as np
from scipy.linalg import expm
from scipy.stats import chisquare
from threadpoolctl import threadpool_limits

from iteration117_quartic import OUT
from iteration117_quartic_laws import void
from iteration117_resolved_pnr import core_kernel,probabilities
from iteration117_transport import moments


def oscillator_test():
    size=18;create=np.diag(np.sqrt(np.arange(1,size)),-1)
    S=np.array([[.35j,.23-.07j],[.23-.07j,.26+.11j]])
    q=np.eye(2,dtype=complex)
    def apply(a,i,x):
        return a@x if i==0 else x@a.T
    def pair(x):
        return sum(.5*S[i,j]*apply(create,i,apply(create,j,x)) for i in range(2) for j in range(2))
    x=np.zeros((size,size),complex);x[0,0]=1
    two=pair(x);ket=x+two+.5*pair(two);norm=np.vdot(ket,ket).real;ket/=np.sqrt(norm)
    _,n,_,m=moments(S)
    aa=[apply(create.T,i,ket) for i in range(2)]
    np.testing.assert_allclose(n,[[np.vdot(aa[j],aa[i]) for j in range(2)] for i in range(2)],atol=2e-14)
    np.testing.assert_allclose(m,[[np.vdot(ket,apply(create.T,i,aa[j])) for j in range(2)] for i in range(2)],atol=2e-14)
    field=np.array([.17+.14j,-.08+.16j])
    a=np.r_[field.real,field.imag]
    model=q,S,2*np.outer(a,a),norm
    ref=np.zeros((size,size))
    z,w=np.polynomial.hermite.hermgauss(32)
    for val,weight in zip(z*np.sqrt(2),w/np.sqrt(np.pi)):
        displaced=ket
        for i,d in enumerate(val*field):displaced=apply(expm(d*create-d.conjugate()*create.T),i,displaced)
        ref+=weight*abs(displaced)**2
    err=[]
    kernel=core_kernel(model,[0,1])
    for count in range(7):
        pats=list(combinations_with_replacement(range(2),count))
        arr=np.array(pats,int).reshape(len(pats),count)
        occ=np.array([np.bincount(a,minlength=2) for a in arr])
        den=np.array([math.prod(math.factorial(k) for k in u) for u in occ])
        err.extend(abs(probabilities(kernel,arr,den)-ref[tuple(occ.T)]))
    err.append(abs(void(model,[0])-ref[0,:].sum()))
    assert max(err)<2e-12
    return float(max(err))


def check_run(prefix):
    source=json.loads((OUT/(prefix+"benchmark.json")).read_text())
    records=np.load(OUT/(prefix+"records.npz"))
    ref=np.load(OUT/(prefix+"panel_34_35.npz"))["click"]
    pop=np.array([i.bit_count() for i in range(256)]);word=np.arange(65536)
    category=9*pop[word%256]+pop[word//256]
    p=np.bincount(category,weights=ref,minlength=81)
    info=[]
    lowprob=p*4096<12
    for k,row in enumerate(source["rows"]):
        h=records[f"panel_{k}"];n=int(h.sum())
        assert n==row["shots"]==4096
        cfg=np.r_[p[~lowprob],p[lowprob].sum()]*n
        observed=np.r_[h[~lowprob],h[lowprob].sum()]
        g=float(chisquare(observed,cfg).pvalue)
        out=dict(direct=bool(k),bank_pair_multinomial_pearson_p=g,accepted_rows=n)
        if k==0 and not prefix:
            law=np.load(OUT/"laws.npz")["number"]
            counts=records["photons_0"] if "photons_0" in records else records["photons"]
            obs=np.bincount(np.arange(len(counts))%len(law),weights=counts,minlength=len(law))/n
            out["total_KS"]=float(abs((obs-law).cumsum()).max())
            out["iid_nominal_DKW99"]=float(np.sqrt(np.log(200)/(2*n)))
        info.append(out)
    return info


if __name__=="__main__":
    with threadpool_limits(limits=2):
        result=dict(independent_oscillator_max_error=oscillator_test(),unmodified=check_run(""),transported=check_run("transport_"))
        (OUT/"validation.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
        print(result,flush=True)
