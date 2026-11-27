"""Independent rejection from a seven/five-pair ket to a nine/six-pair ket."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits

from iteration118_auxiliary import FAC,History,hardware,prepare,refine,virtual
from iteration119_benchmark import OUT,stable
from iteration119_matching import graphs


def clicked(ket,alpha,detect,rng):
    d=len(ket)-1;fac=np.sqrt(FAC[:d+1]);a=abs(alpha)
    def bra(w):
        return w**np.arange(d+1)/fac
    vac=bra(-alpha.conjugate())*np.exp(-a*a/2)
    if detect.random()*np.vdot(ket,ket).real < abs(vac@ket)**2:
        return 0,vac,-alpha.conjugate()
    c=np.zeros(d+1)
    for k,x in enumerate(abs(ket)):
        for j in range(k+1):c[j]+=x*a*math.comb(k,j)*a**(k-j)/fac[k]
        for j in range(k):c[j]+=x*k*math.comb(k-1,j)*a**(k-1-j)/fac[k]
    weights=c*fac;normal=weights.sum()
    while True:
        ell=rng.choice(d+1,p=weights/normal)
        r2=rng.gamma(ell+1);z=np.sqrt(r2)*np.exp(2j*np.pi*rng.random());t=rng.random()
        w=-(1-t)*alpha.conjugate()+np.sqrt(t)*z.conjugate()
        basis=bra(w);f=alpha*basis
        f[1:]+=np.sqrt(np.arange(1,d+1))*basis[:-1]
        ratio=np.exp(-(1-t)*a*a)*abs(f@ket)**2/(normal*np.polynomial.polynomial.polyval(r2,c/fac))
        assert 0<=ratio<=1+1e-9,ratio
        if rng.random()<ratio:return 1,f,w


def initialize():
    global models,BLAS
    BLAS=threadpool_limits(limits=1)
    t,ss=hardware();models=[]
    for s,lo,hi in zip(ss,(7,5),(9,6)):
        models.append((prepare(t,s,lo),lo,hi,prepare(t,s,hi)[-1]))


def draw_one(args):
    route,index=args;start=perf_counter();archives={}
    for j,(model,lo,hi,R) in enumerate(models):
        q,S,chol,fields,_=model;key=f"{route}_{index}_{j}"
        rv,ra,rm,rx=[np.random.default_rng(s) for s in np.random.SeedSequence([119,71,route,j,index]).spawn(4)]
        beta,tries=virtual(q,S,chol,lo,rv);beta=beta.conjugate();z=ra.normal(size=fields.shape[1])
        alpha=fields@z;v=beta@q;hist=History(S,lo);rec=[];field=np.zeros(len(S),complex)
        for i,row in enumerate(q):
            v-=beta[i]*row;ket=hist.ket(v,row)
            if route:out,f,w=clicked(ket,alpha[i],rm,rx)
            else:out,f=refine(ket,alpha[i],False,rm);w=-alpha[i].conjugate()
            field+=w*row
            if out:v+=hist.update(row,f)*row
            else:v+=w*row
            rec.append(out)
        archives["row"+key]=np.array(rec,np.uint16);archives["v"+key]=field;archives["z"+key]=z
    archives[f"time{route}_{index}"]=perf_counter()-start
    return archives


def weight(args):
    route,index,path=args;start=perf_counter();ratio=1.;deferred=[]
    with np.load(path) as data:
        for j,(model,lo,hi,R) in enumerate(models):
            q,S,_,fields,retained=model;key=f"{route}_{index}_{j}"
            alpha=fields@data["z"+key];row=data["row"+key]
            sites=np.repeat(np.arange(len(q)),row)
            amps,bad=stable(S,data["v"+key],q[sites]/alpha[sites,None],hi,1.8)
            ratio*=abs(amps[hi]/amps[lo])**2*(retained/R);deferred.append(bad)
    u=np.random.default_rng([119,72,route,index]).random()
    return dict(direct=bool(route),index=index,w=float(ratio),u=float(u),accepted=bool(2*u<ratio),
                seconds=perf_counter()-start,deferred=deferred)


def run(shots,workers,stage):
    OUT.mkdir(exist_ok=True)
    tasks=[(route,i) for i in range(shots) for route in (0,1)]
    path=OUT/f"high_proposals_{shots}.npz"
    if stage=="draw":
        if path.exists():raise FileExistsError(path)
        with ProcessPoolExecutor(workers,initializer=initialize) as pool:
            save={}
            for chunk in pool.map(draw_one,tasks):save.update(chunk)
        np.savez_compressed(path,**save);print("generated",len(tasks),flush=True)
        return
    with ProcessPoolExecutor(workers,initializer=initialize) as pool:
        scores=list(pool.map(weight,[(r,i,path) for r,i in tasks]))
    initialize();R0=np.prod([m[0][-1] for m in models]);R1=np.prod([m[-1] for m in models])
    # The positive part of |T><T|-M|B><B| has one positive eigenvalue.
    M=2.;d=1-R0/R1
    cap_loss=2*M*d/(np.sqrt((M-1)**2+4*M*d)+M-1)
    info=dict(proposal_pair_caps=[7,5],target_pair_caps=[9,6],ceiling=M,proposal_retention=R0,
              target_retention=R1,clipping_mass_upper=cap_loss,core_trace_upper=np.sqrt(1-R1),
              full_record_tv_upper=cap_loss+np.sqrt(1-R1),scores=scores)
    with np.load(path) as data:
        rows={}
        for route,key in enumerate(("pnr","click")):
            accepted=[s["index"] for s in scores if s["direct"]==bool(route) and s["accepted"]]
            if not accepted:continue
            items=[]
            for i in accepted:
                a,b=(data[f"row{route}_{i}_{j}"] for j in (0,1))
                items.append(a|b if route else a+b)
            rows[key]=np.stack(items)
    np.savez_compressed(OUT/f"subpercent_{shots}.npz",**rows)
    (OUT/f"subpercent_{shots}.json").write_text(json.dumps(info,indent=2)+"\n",encoding="utf8")
    print("accepted counts",{key:len(x) for key,x in rows.items()},"bound",info["full_record_tv_upper"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("draw","score"));p.add_argument("--shots",type=int,default=8)
    p.add_argument("--workers",type=int,default=4);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.shots,a.workers,a.stage)
