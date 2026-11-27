"""Generate independent quartic proposals and evaluate both whole-record likelihoods."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from threadpoolctl import threadpool_limits

from iteration117_quartic import prepare as quartic
from iteration118_auxiliary import prepare,hardware
from iteration115_visible_v1 import DATA,ROOT
from iteration119_matching import amplitudes

OUT=DATA/"iteration119_weights"


def native(tmp):
    dll=Path(tmp)/("proposal.dll" if os.name=="nt" else "proposal.so")
    subprocess.run(["g++","-std=c++17","-O3","-fopenmp","-fPIC","-shared",
                    str(ROOT/"native/record_proposal119.cpp"),"-o",str(dll)],check=True)
    if os.name=="nt":native.runtime=os.add_dll_directory(str(Path(shutil.which("g++")).parent))
    lib=ct.CDLL(str(dll));I=ct.c_int;P=ct.c_void_p
    lib.proposal119.argtypes=[I]*5+[ct.c_uint64,ct.c_uint32,ct.c_uint]+[P]*7+[ct.c_double]+[P]*4
    return lib


def generate(shots,start):
    t,ss=hardware();setups=[quartic(t,s) for s in ss]
    root=OUT/f"proposal_{start}_{shots}.npz";saved={}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        lib=native(tmp)
        for direct in (0,1):
            timer=perf_counter()
            for j,((args,norm),transform,_) in enumerate(setups):
                z=np.array([np.random.default_rng([119,51,direct,j,i]).standard_normal(len(transform))
                            for i in range(start,start+shots)])
                alpha=np.ascontiguousarray(z@transform)
                data=np.empty(alpha.shape,np.uint8);linear=np.empty_like(alpha);ratio=np.empty(shots)
                rc=lib.proposal119(shots,alpha.shape[1],len(transform),direct,8,119001,start,j,
                    *[a.ctypes.data for a in args],norm,alpha.ctypes.data,data.ctypes.data,linear.ctypes.data,ratio.ctypes.data)
                assert rc==0,rc
                tag=f"{direct}_{j}"
                saved["row"+tag]=data;saved["z"+tag]=z;saved["v"+tag]=linear@args[0];saved["log"+tag]=ratio
            saved[f"elapsed{direct}"]=perf_counter()-timer
            print("proposal",direct,shots,saved[f"elapsed{direct}"],flush=True)
        if os.name=="nt":
            import _ctypes
            _ctypes.FreeLibrary(lib._handle)
    np.savez_compressed(root,**saved)


def stable(S,v,J,cap,threshold):
    # Each labeled monomer is affine. Replace a risky monomer 1+J.g by
    # (d+J.g)/d and average the numerator on a circle d=1+s*phase.
    # Its constant coefficient is the desired product, including repeated sites.
    width=np.linalg.norm(J,axis=1)*np.sqrt(np.linalg.norm(S,2))
    bad=np.flatnonzero(width>threshold)
    if not len(bad):return amplitudes(S,v,J,cap),0
    scale=1+width[bad]/threshold
    total=0j
    for phase in np.exp(2j*np.pi*(np.arange(len(bad)+1)+.25)/(len(bad)+1)):
        den=1+scale*phase
        rows=J.copy();rows[bad]/=den[:,None]
        total+=den.prod()*amplitudes(S,v,rows,cap)
    return total/(len(bad)+1),len(bad)


def initialize(path,k):
    global BLAS,bank,models,caps
    BLAS=threadpool_limits(limits=1)
    with np.load(path) as f:bank={key:f[key] for key in f}
    t,ss=hardware();models=[];caps=[k,min(5,k)]
    for s,cap in zip(ss,caps):
        q,S,_,fields,R=prepare(t,s,cap)
        p0=np.sqrt(np.linalg.det(np.eye(len(S))-S@S.conj().T).real)
        norm2=prepare(t,s,2)[-1]/p0
        models.append((q,S,fields.T,norm2,R/p0))


def score(item):
    route,i=item;zratio=1+0j;err=0.;bad=0;start=perf_counter()
    for j,(q,S,transform,norm2,norm) in enumerate(models):
        key=f"{route}_{j}"
        row=bank["row"+key][i];alpha=bank["z"+key][i]@transform
        sites=np.repeat(np.arange(len(q)),row)
        J=q[sites]/alpha[sites,None]
        amp,n=stable(S,bank["v"+key][i],J,caps[j],2.)
        bad+=n
        err=max(err,abs(np.log(abs(amp[2])**2/norm2)-bank["log"+key][i]))
        zratio*=amp[-1]/amp[2]*np.sqrt(norm2/norm)
    return zratio,err,bad,perf_counter()-start


def run(shots,k,start,workers):
    OUT.mkdir(exist_ok=True)
    path=OUT/f"proposal_{start}_{shots}.npz"
    if not path.exists():generate(shots,start)
    with ProcessPoolExecutor(workers,initializer=initialize,initargs=(path,k)) as pool:
        list(pool.map(score,[(0,0)]*workers))
        arr={};summary=[]
        for direct in (0,1):
            timer=perf_counter();values=list(pool.map(score,[(direct,i) for i in range(shots)],chunksize=1))
            dt=perf_counter()-timer;z=np.array([v[0] for v in values]);w=abs(z)**2
            arr[f"z{direct}"]=z;arr[f"err{direct}"]=np.array([v[1] for v in values])
            arr[f"time{direct}"]=np.array([v[3] for v in values])
            summary.append(dict(direct=bool(direct),shots=shots,weight_mean=float(w.mean()),
                weight_max=float(w.max()),max_quartic_log_error=float(arr[f"err{direct}"].max()),
                max_deferred=max(v[2] for v in values),score_s_per_row=dt/shots,
                clipping_loss_estimates={str(m):float(1-np.minimum(w,m).mean()) for m in [1,2,4,8,16]}))
            print(summary[-1],flush=True)
    name=f"score_{start}_{shots}_{k}"
    np.savez_compressed(OUT/(name+".npz"),**arr)
    (OUT/(name+".json")).write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--shots",type=int,default=64);p.add_argument("--cap",type=int,default=7)
    p.add_argument("--start",type=int,default=0);p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    with threadpool_limits(limits=1):run(a.shots,a.cap,a.start,a.workers)
