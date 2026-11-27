"""Quartic sampler after a uniform-loss reference and a passive contraction.

Input arrays: complex transfer T (rows = observed ports, columns = independent
squeezers), and squeezing (one nonnegative real row per independent spectrum).
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits

from iteration114_rank_v1 import physical_total,cov
from iteration117_quartic import precompute
from iteration122_powers import ROOT
from iteration122_family_certificate import pair_retained,upper_decimal
from iteration122_powers import OUT,json_write

FOLDER=OUT/'unequal_loss'


def build(tmp):
    import ctypes as ct
    import os,shutil,subprocess
    library=Path(tmp)/('bright.dll' if os.name=='nt' else 'bright.so')
    subprocess.run(['g++','-O3','-std=c++17','-fPIC','-shared','-fopenmp',str(ROOT/'native/bright_quartic122.cpp'),'-o',str(library)],check=True)
    if os.name=='nt':build.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
    lib=ct.CDLL(str(library.resolve()));I=ct.c_int;P=ct.c_void_p
    lib.quartic122.argtypes=[I]*5+[ct.c_uint64,ct.c_uint32,ct.c_uint]+[P]*7+[ct.c_double]+[P]*4
    lib.local122.argtypes=[I,P,P,ct.c_uint64,P,P]
    return lib


def draw(lib,spec,alpha,route,spectrum,offset,seed):
    args,norm=spec
    n,m=alpha.shape;r=args[0].shape[1]
    alpha=np.ascontiguousarray(alpha);out=np.empty((n,m),np.uint64)
    diag=np.zeros(2);counters=np.zeros(2,np.uint64)
    rc=lib.quartic122(n,m,r,route,8,122000000+seed,offset,spectrum,*[a.ctypes.data for a in args],
                      norm,alpha.ctypes.data,out.ctypes.data,diag.ctypes.data,counters.ctypes.data)
    if rc:raise ArithmeticError(('local readout failed',rc))
    return out,diag,counters


def make_example(bright=False):
    rng=np.random.default_rng([122,100])
    m,r=(32,8) if bright else (1152,50)
    def columns(a,b):
        z=rng.standard_normal((a,b))+1j*rng.standard_normal((a,b))
        return np.linalg.qr(z,mode='reduced')[0]
    t=(columns(m,r)*np.sqrt(.07*np.linspace(.29,.94,r)))@columns(r,r)
    s=np.stack([np.linspace(2.65,3.50,r),np.linspace(1.55,2.45,r)])
    folder=OUT/'bright' if bright else FOLDER
    if bright:
        # Keep eight dark ports informative while most ports carry large counts.
        t[:8]*=1e-3
        s+=6.5
    folder.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(folder/'input.npz',T=t,squeezing=s)


def complete(t,eta):
    r=t.shape[1]
    val,u=np.linalg.eigh(np.eye(r)-t.conj().T@t/eta)
    if val.min() < -1e-12:
        raise ValueError('Reference transmission is smaller than this transfer requires')
    q=np.vstack([t/np.sqrt(eta),(u*np.sqrt(val.clip(0)))@u.conj().T])
    if np.max(abs(q.conj().T@q-np.eye(r)))>1e-11:
        raise ArithmeticError('Isometric completion failed')
    return q


def decompose(q,s,eta):
    squeezed=1-eta+eta*np.exp(-2*s)
    S=np.diag((squeezed-1)/(squeezed+1)).astype(complex)
    noise=1j*np.sqrt(eta*(1-eta)/squeezed)*np.sinh(s)
    return precompute(q,S),(q*noise).T.copy()


def references(t,ss,ports):
    n=4096
    pgf=np.ones(n//2+1,complex)
    v=1.
    for s in ss:
        pgf*=np.fft.rfft(physical_total(t,s,n))
        x=cov(t[ports],s)
        v*=np.exp(-.5*np.linalg.slogdet(np.eye(len(x))+x)[1])
    return np.fft.irfft(pgf,n=n),float(v)


def run(lib,path,dest,eta,shots,seed):
    if not 0<eta<1 or not 1<=shots<=2**24 or not 0<=seed<=2**24:
        raise ValueError('Invalid transmission, sample count or seed domain')
    if dest.suffix != ".npz":
        raise ValueError("Output must have an .npz extension")
    for target in (dest, dest.with_suffix(".json")):
        if target.exists():
            raise FileExistsError(target)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with np.load(path) as f:
        t=f['T'];ss=f['squeezing']
    m,r=t.shape
    if ss.ndim!=2 or not 1<=len(ss)<=65536 or ss.shape[1]!=r or min(m,r)<1 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(ss)) or np.any(ss<0):
        raise ValueError('Invalid optical input')
    # Duplicate a zero-gain reference mode when an odd rank requires a rational
    # worst-case vacuum coefficient. It changes only this conservative bound.
    rank_bound=r+(r%2)
    R=pair_retained((eta/(2-eta))**2,rank_bound,2)
    bound=upper_decimal(1-R**len(ss))
    setup=perf_counter()
    q=complete(t,float(eta))
    models=[decompose(q,s,float(eta)) for s in ss]
    setup=perf_counter()-setup
    ports=np.arange(min(8,m))
    law,void=references(t,ss,ports)
    arrays=dict(ports=ports,optical_mod4096=law)
    stats=[]
    for route in (0,1):
        records=[];times=[];worst=np.zeros(2);total_counts=np.zeros(2,np.uint64)
        for start in range(0,shots,32):
            n=min(32,shots-start)
            begin=perf_counter()
            record=np.zeros((n,m),np.uint64)
            for j,(spec,field) in enumerate(models):
                rng=np.random.default_rng([122,1001,seed,route,j,start])
                alpha=rng.standard_normal((n,r))@field
                # A readout-specific namespace also separates refinement streams.
                out,err,counts=draw(lib,spec,alpha,route,j,start,seed)
                total_counts+=counts
                if route:
                    record|=out[:,:m]
                else:
                    record+=out[:,:m]
                worst=np.maximum(worst,err)
            records.append(record)
            times.append(perf_counter()-begin)
        rec=np.vstack(records)
        name='click' if route else 'pnr'
        arrays[name]=rec.astype(np.uint8) if route else rec
        obs=float((rec[:,ports]==0).all(1).mean())
        row=dict(route=name,rows=shots,microseconds_per_delivered=1e6*sum(times)/shots,seconds=times,
                 maximum_numerical_residual=worst.tolist(),observed_vacuum=obs,optical_vacuum=void,maximum_port=int(rec.max()),
                 bright_proposals=int(total_counts[0]),bright_site_measurements=int(total_counts[1]))
        if not route:
            ns=rec.sum(1,dtype=np.int64)
            counts=np.bincount(ns%4096,minlength=4096)
            row.update(observed_mean=float(ns.mean()),mod4096_ks=float(abs(np.cumsum(counts/shots-law)).max()))
        stats.append(row)
    info=dict(readouts=stats,eta=str(eta),visible_ports=m,hidden_ports=r,sources_per_spectrum=r,
              spectra=len(ss),reference_setup_seconds=setup,quartic_full_tv_upper=bound,
              input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              exact_model_mean=float(sum(np.sum(abs(t)**2@np.sinh(s)**2) for s in ss)),
              run_seed=seed)
    np.savez_compressed(dest,**arrays)
    json_write(dest.with_suffix('.json'),info)
    print(m,r,info['exact_model_mean'],bound,[(x['route'],x['microseconds_per_delivered']) for x in stats],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['example','sample','bright-example'])
    ap.add_argument('--input',type=Path,default=FOLDER/'input.npz')
    ap.add_argument('--output',type=Path,default=FOLDER/'records.npz')
    ap.add_argument('--eta',type=Fraction,default=Fraction(7,100))
    ap.add_argument('--shots',type=int,default=4096)
    ap.add_argument('--seed',type=int,default=122)
    args=ap.parse_args()
    with threadpool_limits(2):
        if args.stage!='sample':make_example(args.stage=='bright-example')
        else:
            import tempfile
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                lib=build(tmp)
                try:run(lib,args.input,args.output,args.eta,args.shots,args.seed)
                finally:
                    import os
                    if os.name=='nt':
                        import _ctypes
                        _ctypes.FreeLibrary(lib._handle)
