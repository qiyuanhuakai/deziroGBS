"""Full-width displaced quartic Gaussian core with PNR and threshold instruments."""
import argparse
import ctypes as ct
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

from iteration114_pairs_v1 import sector
from iteration115_displacement_v1 import conditional
from iteration115_visible_v1 import DATA, ROOT, hardware

OUT=DATA/"iteration117_quartic"
P=ct.c_void_p
I=ct.c_int


def loadlib():
    OUT.mkdir(exist_ok=True)
    dll=OUT/("quartic.dll" if os.name=="nt" else "quartic.so")
    src=ROOT/"native/displaced_quartic117.cpp"
    subprocess.run(["g++","-std=c++17","-O3","-fopenmp","-fPIC","-shared",str(src),"-o",str(dll)],check=True)
    if os.name=="nt":loadlib.runtime=os.add_dll_directory(str(Path(shutil.which("g++")).parent))
    lib=ct.CDLL(str(dll.resolve()))
    lib.quartic117.argtypes=[I]*5+[ct.c_uint64,ct.c_uint32,ct.c_uint]+[P]*7+[ct.c_double]+[P]*5
    return lib


def prepare(t,s,transport=False):
    q,v,_,response,purity=conditional(t,s)
    r=q.shape[1];eye=np.eye(r)
    k=np.eye(2*r)-np.linalg.inv((v+np.eye(2*r))/2)
    j=np.vstack((eye,1j*eye))/np.sqrt(2)
    S=j.T@k@j
    gain=.5*np.log(np.linalg.eigvalsh(v)[r:])
    retained,levels=sector(gain,4)
    info={}
    if transport:
        from iteration117_transport import corrected
        response,info=corrected(S,response,retained)
    scale=(q@(response[:r]+1j*response[r:]))/2
    return precompute(q,S),scale.T.copy(),dict(retained=retained,sector_mass=levels.tolist(),
            purity=purity,transport=info,mean_displacement=float((abs(scale)**2).sum()))


def precompute(q,S):
    m,r=q.shape
    np.testing.assert_allclose(q.conj().T@q,np.eye(r),atol=2e-13)
    np.testing.assert_allclose(S,S.T,atol=2e-13)
    gram=np.zeros((r,r),complex)
    g,w,cubic=np.empty((3,m,r,r),complex)
    norms=np.empty((m,2))
    def metrics():
        weighted=gram@S@gram.T
        pair=.5*np.vdot(S,weighted).real
        corr=weighted@S.conj().T
        fourth=.5*pair**2+.25*np.sum(corr*corr.T).real
        return weighted,pair*gram+corr@gram,(pair,fourth)
    for i in range(m-1,-1,-1):
        g[i]=gram
        w[i],cubic[i],norms[i]=metrics()
        gram+=np.outer(q[i].conj(),q[i])
    _,_,full=metrics()
    norm=1/np.sqrt(1+sum(full))
    return [np.ascontiguousarray(x) for x in (q,S,q@S.T,g,w,cubic,norms)],float(norm)


def draw(lib,model,alpha,click,spectrum,offset,threads,replay=None):
    args,norm=model;m,r=args[0].shape
    alpha=np.ascontiguousarray(alpha);n=len(alpha)
    out=np.zeros((n,m),np.uint8);diag=np.zeros(2)
    path=np.zeros(n) if replay is not None else None
    rc=lib.quartic117(n,m,r,click,threads,117000001,offset,spectrum,*[a.ctypes.data for a in args],
                     norm,alpha.ctypes.data,out.ctypes.data,diag.ctypes.data,
                     None if replay is None else replay.ctypes.data,None if path is None else path.ctypes.data)
    assert rc==0,(rc,diag)
    return out,diag,path


def toy(lib):
    rng=np.random.default_rng(117003)
    m,r=3,2
    q=np.linalg.qr(rng.normal(size=(m,r))+1j*rng.normal(size=(m,r)))[0]
    u=np.linalg.qr(rng.normal(size=(r,r))+1j*rng.normal(size=(r,r)))[0]
    S=(u*np.array([.62,.42]))@u.T
    model=precompute(q,S)
    dim=16
    create=np.diag(np.sqrt(np.arange(1,dim)),-1)
    def act(x,op,axis):
        return np.moveaxis(np.tensordot(op,x,axes=(1,axis)),0,axis)
    pair=q@S@q.T
    vacuum=np.zeros((dim,)*m,complex);vacuum[(0,)*m]=1.
    def plus_pair(x):
        return sum(.5*pair[i,j]*act(act(x,create,j),create,i) for i in range(m) for j in range(m))
    two=plus_pair(vacuum)
    ket=vacuum+two+.5*plus_pair(two)
    np.testing.assert_allclose(1/np.linalg.norm(ket),model[1],atol=1e-14)
    ket*=model[1]
    alpha=np.array([.31+.28j,-.43+.12j,.17-.36j])
    for i,a in enumerate(alpha):ket=act(ket,expm(a*create-a.conjugate()*create.T),i)
    probs=abs(ket)**2
    words=np.array(list(itertools.product(range(7),repeat=m)),np.uint8)
    _,diag,values=draw(lib,model,np.tile(alpha,(len(words),1)),0,0,0,4,words)
    error=float(abs(values-probs[tuple(words.T)]).max())
    ref=np.zeros(8)
    for idx in np.ndindex(probs.shape):ref[sum((n>0)<<j for j,n in enumerate(idx))]+=probs[idx]
    steps=np.array(list(itertools.product(range(6),repeat=m)),np.uint8)
    _,_,refinements=draw(lib,model,np.tile(alpha,(len(steps),1)),1,0,0,4,steps)
    actual=np.bincount((steps>0)@(1<<np.arange(m)),weights=refinements,minlength=8)
    err_click=float(abs(ref-actual).max())
    assert max(error,err_click)<2e-11,(error,err_click,diag)
    return dict(pnr_patterns=len(words),direct_instrument_paths=len(steps),max_pnr=error,max_click=err_click)


def run(shots,batch,power="high",transport=False,repeat=False):
    from iteration117_power_inputs import load_power
    prefix="" if power=="high" else power+"_"
    source_id={"high":0,"median":1,"low":2}[power]+3*transport
    if transport:prefix="transport_"+prefix
    if repeat:prefix="repeat_"+prefix
    lib=loadlib();checks=toy(lib);print("toy",checks,flush=True)
    t,ss=hardware() if power=="high" else load_power(power)
    start=perf_counter();models=[prepare(t,s,transport) for s in ss];setup=perf_counter()-start
    bound=np.sqrt(1-np.prod([stats["retained"] for _,_,stats in models]))
    if transport:bound=np.sqrt(1-np.prod([stats["transport"]["root_fidelity_lower"] for _,_,stats in models])**2)
    print("setup",setup,"full_bound",bound,flush=True)
    rows=[];save={}
    for direct in (0,1):
        times=[];worst=np.zeros(2);hist=np.zeros(1153,int);nums=np.zeros(1153,int)
        pair=np.zeros(81,int)
        for off in range(-batch,shots,batch):
            n=batch if off<0 else min(batch,shots-off)
            start=perf_counter();rec=[]
            for j,(spec,transform,_) in enumerate(models):
                domain=[117,510,direct,j,off+batch] if source_id==0 else [117,510,source_id,direct,j,off+batch]
                rng=np.random.default_rng(np.random.SeedSequence(domain))
                alpha=rng.normal(size=(n,len(transform)))@transform
                out,diag,_=draw(lib,spec,alpha,direct,2*source_id+j,2**31 if off<0 else off,8)
                worst=np.maximum(worst,diag);rec.append(out)
            out=rec[0]|rec[1] if direct else rec[0].astype(np.uint16)+rec[1]
            elapsed=perf_counter()-start
            if off<0:continue
            times.append(elapsed)
            if off==0:save[f"sample_{direct}"]=out[:16].copy()
            banks=(out>0).reshape(n,144,8).sum(2)
            hist+=np.bincount(banks.sum(1),minlength=len(hist))
            pair+=np.bincount(9*banks[:,34]+banks[:,35],minlength=81)
            if not direct:nums+=np.bincount(out.sum(1).astype(int),minlength=len(nums))
        row=dict(direct=bool(direct),shots=shots,batch=batch,times_seconds=times,
                 mean_us=1e6*sum(times)/shots,normalization_residual=worst.tolist(),
                 mean_clicks=float(hist@np.arange(len(hist))/shots))
        assert worst.max()<1e-6,worst
        save[f"clicks_{direct}"]=hist;save[f"panel_{direct}"]=pair
        if not direct:
            save["photons"]=nums
            row["mean_photons"]=float(nums@np.arange(len(nums))/shots)
        rows.append(row);print(direct,shots,row["mean_us"],row["mean_clicks"],flush=True)
    info=dict(toy=checks,full_record_TV_upper=float(bound),components=[stats for _,_,stats in models],
              setup_seconds=setup,rows=rows)
    np.savez_compressed(OUT/(prefix+"records.npz"),**save)
    (OUT/(prefix+"benchmark.json")).write_text(json.dumps(info,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--shots",type=int,default=128);p.add_argument("--batch",type=int,default=8)
    p.add_argument("--repeat",action="store_true");p.add_argument("--transport",action="store_true");p.add_argument("--power",choices=("high","low","median"),default="high");p.add_argument("--toy",action="store_true");opt=p.parse_args()
    with threadpool_limits(limits=3):
        if opt.toy:print(toy(loadlib()),flush=True)
        else:run(opt.shots,opt.batch,opt.power,opt.transport,opt.repeat)
