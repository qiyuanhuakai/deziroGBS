"""Exact independent sampling by rejection from orthogonal pair-number bands."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
import json
import os
import tempfile
from time import perf_counter

import numpy as np
from scipy.special import i0e
from threadpoolctl import threadpool_limits

from iteration117_quartic import prepare as quartic
from iteration118_auxiliary import FAC,History,hardware,prepare,refine
from iteration119_benchmark import OUT,native,stable
from iteration119_subpercent import clicked


def conditioned_virtual(q,S,chol,first,last,rng):
    m,r=q.shape
    while True:
        x=chol@rng.standard_normal(2*r)
        a=(x[:r]+1j*x[r:])*.5*np.exp(2j*np.pi*rng.random())
        t=a.conj()@S@a.conj()/2
        p=sum(t**j/FAC[j] for j in range(first,last+1))
        v=2*abs(t)
        accept=abs(p)**2*np.exp(-v)/((last-first+1)*i0e(v))
        if rng.random()<accept:break
    noise=(rng.standard_normal(m)+1j*rng.standard_normal(m))/np.sqrt(2)
    return (noise+q@(a-q.conj().T@noise)).conj()


def initialize(optics=None):
    global models,limit
    limit=threadpool_limits(limits=1)
    t,ss=hardware() if optics is None else optics;models=[]
    for j,s in enumerate(ss):
        stops=[2,5,7] if j==0 else [2,5]
        model=prepare(t,s,stops[-1])
        R=np.array([prepare(t,s,k)[-1] for k in stops])
        mass=np.diff(np.r_[0,R])/R[-1]
        # Runtime tuning affects only the relative proposal costs; the Cauchy
        # normalization enforces the envelope for every choice of positive costs.
        costs=np.array([.14,1.8,130.] if j==0 else [.025,.7])
        weights=np.sqrt(mass/costs)*np.sqrt(mass*costs).sum()
        assert abs(np.sum(mass/weights)-1)<2e-14
        models.append((model,stops,mass,weights))


def component(route,j,index,band):
    (q,S,chol,fields,_),stops,_,_=models[j]
    first=stops[band-1]+1;last=stops[band]
    rv,ra,rm,rx=[np.random.default_rng(s) for s in np.random.SeedSequence([119,81,route,j,index]).spawn(4)]
    beta=conditioned_virtual(q,S,chol,first,last,rv);z=ra.standard_normal(fields.shape[1]);alpha=fields@z
    v=beta@q;hist=History(S,last);row=np.zeros(len(q),np.uint16);field=np.zeros(len(S),complex)
    for i,d in enumerate(q):
        v-=beta[i]*d;ket=hist.ket(v,d,first)
        if route:out,f,w=clicked(ket,alpha[i],rm,rx)
        else:out,f=refine(ket,alpha[i],False,rm);w=-alpha[i].conjugate()
        field+=w*d;row[i]=out
        if out:v+=hist.update(d,f)*d
        else:v+=w*d
    return row,field,z


def task(args):
    route,j,index,band,base=args
    data,start=base,perf_counter()
    if band:data=component(route,j,index,band)
    row,v,z=data
    (q,S,_,fields,_),stops,mass,weights=models[j]
    alpha=fields@z;sites=np.repeat(np.arange(len(q)),row)
    amps,bad=stable(S,v,q[sites]/alpha[sites,None],stops[-1],1.8)
    parts=np.diff(np.r_[0,amps[stops]])
    ratio=abs(amps[-1])**2/np.sum(weights*abs(parts)**2/mass)
    if not 0<=ratio<=1+3e-12:raise ArithmeticError(ratio)
    u=np.random.default_rng([119,82,route,j,index]).random()
    accepted=u<ratio
    return (row,v,z) if accepted else None,(index,band,float(ratio),u,perf_counter()-start,bad)


def generate_batch(lib,setup,route,j,start,batch):
    (args,norm),transform,_=setup
    z=np.array([np.random.default_rng([119,83,route,j,i]).standard_normal(len(transform))
                for i in range(start,start+batch)])
    alpha=np.ascontiguousarray(z@transform)
    records=np.empty(alpha.shape,np.uint8);linear=np.empty_like(alpha);log=np.empty(batch)
    rc=lib.proposal119(batch,alpha.shape[1],len(transform),route,8,119003,start,j,
                    *[a.ctypes.data for a in args],norm,alpha.ctypes.data,records.ctypes.data,linear.ctypes.data,log.ctypes.data)
    assert rc==0,rc
    v=linear@args[0]
    weights=models[j][-1]
    for off in range(batch):
        index=start+off
        band=np.random.default_rng([119,84,route,j,index]).choice(len(weights),p=weights/weights.sum())
        yield route,j,index,band,None if band else (records[off],v[off],z[off])


def run(shots,workers,batch):
    OUT.mkdir(exist_ok=True);initialize()
    t,ss=hardware();setups=[quartic(t,s) for s in ss]
    saved={};report=[]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp,ProcessPoolExecutor(workers,initializer=initialize) as pool:
        lib=native(tmp)
        for route in (0,1):
            timer=perf_counter();combined=[];diagnostics={}
            for j,setup in enumerate(setups):
                start=0;selected=[];log=[]
                while len(selected)<shots:
                    tasks=generate_batch(lib,setup,route,j,start,batch)
                    for data,info in pool.map(task,tasks):
                        if data is not None:selected.append(data)
                        log.append(info)
                    start+=batch
                    print("accepted",route,j,len(selected),"of",start,flush=True)
                key=f"{route}_{j}"
                records,fields,z=map(np.stack,zip(*selected[:shots]))
                combined.append(records);saved["row"+key]=records;saved["v"+key]=fields;saved["z"+key]=z
                saved["log"+key]=np.array(log)
                diagnostics[str(j)]=dict(proposals=start,successes=len(selected),coefficients=models[j][-1].tolist(),
                                         band_counts=np.bincount(np.array(log)[:,1].astype(int),minlength=len(models[j][1])).tolist())
            saved["click" if route else "pnr"]=combined[0]|combined[1] if route else np.add(combined[0],combined[1],dtype=np.uint32)
            report.append(dict(direct=bool(route),shots=shots,wall_s=perf_counter()-timer,spectra=diagnostics))
            print(report[-1],flush=True)
        if os.name=="nt":
            import _ctypes
            _ctypes.FreeLibrary(lib._handle)
    np.savez_compressed(OUT/f"exact_bands_{shots}.npz",**saved)
    (OUT/f"exact_bands_{shots}.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--shots",type=int,default=32)
    p.add_argument("--workers",type=int,default=6);p.add_argument("--batch",type=int,default=24);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.shots,a.workers,a.batch)
