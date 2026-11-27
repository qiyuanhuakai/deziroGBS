"""Recontract ill-conditioned observed histories with one exceptional bra kept explicit."""
import json
import numpy as np
from threadpoolctl import threadpool_limits

from iteration118_auxiliary import FAC,OUT,History,hardware,prepare,refine,virtual


class Quantile:
    def __init__(self,value):
        self.value=value

    def random(self):
        return self.value


def replay(route,spectrum,index):
    t,ss=hardware();cap=5 if spectrum else 7
    q,S,chol,field,_=prepare(t,ss[spectrum],cap)
    seed=[118,610,7,1,route,spectrum,index]
    rv,ra,rm=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    beta=virtual(q,S,chol,cap,rv)[0].conj();alpha=field@ra.standard_normal(field.shape[1])
    v=beta@q;hist=History(S,cap);previous=None;trace=[];strength=np.linalg.norm(S,2)
    for i,row in enumerate(q):
        v-=beta[i]*row
        ket=hist.ket(v,row);u=rm.random();count,f=refine(ket,alpha[i],route,Quantile(u))
        err=0.
        if previous is not None:
            val=ket@(beta[i]**np.arange(len(ket))/np.sqrt(FAC[:len(ket)]))
            err=abs(val-previous)/max(abs(previous),np.linalg.norm(ket)*np.exp(abs(beta[i])**2/2))
        previous=(f@ket)/f[0]
        leverage=0.
        if count:
            a=f[1]/f[0];b=f[2]/f[0]/np.sqrt(2)-a*a/2
            width=np.linalg.norm(row)*np.sqrt(strength)
            leverage=abs(a)*width+abs(b)*width**2
            v+=hist.update(row,f)*row
        else:v-=alpha[i].conjugate()*row
        trace.append((ket,count,f,err,leverage,u))
    worst=int(np.argmax([x[3] for x in trace]))
    pivot=int(np.argmax([x[4] for x in trace[:worst]]))
    # Select the highest-risk prefixes and points throughout the remaining history.
    sites=set(np.argsort([x[3] for x in trace])[-12:])|set(np.linspace(pivot+1,len(q)-1,24,dtype=int))
    sites={i for i in sites if i>pivot}
    hist=History(S,cap);v=beta@q
    max_distance=0.;different=0;agreement_on_worst=None
    d=2*cap;circle=2.25*np.exp(2j*np.pi*np.arange(d+1)/(d+1))
    for i,(row,(stored,count,f,err,leverage,u)) in enumerate(zip(q,trace)):
        v-=beta[i]*row
        if i in sites:
            values=np.stack([hist.ket(v+node*q[pivot],row) for node in circle])
            coeff=np.fft.fft(values,axis=0)/(d+1)
            explicit=(trace[pivot][2]*np.sqrt(FAC[:d+1])/2.25**np.arange(d+1))@coeff
            a=stored/np.linalg.norm(stored);b=explicit/np.linalg.norm(explicit)
            b*=np.exp(-1j*np.angle(np.vdot(a,b)))
            distance=np.linalg.norm(a-b)
            max_distance=max(max_distance,distance)
            new,bra=refine(explicit,alpha[i],route,Quantile(u))
            same=new==count and np.array_equal(bra,f)
            different+=not same
            if i==worst:agreement_on_worst=float(distance)
        if i==pivot:continue
        if count:v+=hist.update(row,f)*row
        else:v-=alpha[i].conjugate()*row
    return dict(direct=bool(route),spectrum=spectrum,index=index,deferred_port=pivot,
                worst_prefix=worst,original_residual=float(trace[worst][3]),prefixes_checked=len(sites),
                max_unit_ket_distance=float(max_distance),ket_distance_at_worst=agreement_on_worst,
                changed_rank_one_outcomes=int(different))


if __name__=="__main__":
    with threadpool_limits(limits=1):
        rows=[replay(1,0,1),replay(0,1,7)]
        print(rows,flush=True)
        (OUT/"replay.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf8")
