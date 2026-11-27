"""General mixed, multispectral collective cores and an independent full Fock law."""
import argparse
import ctypes as ct
import math
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import numpy as np
from scipy.signal import convolve
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,save,manifest
from iteration106_field_v1 import DATA,actual_input
from iteration106_sample_v1 import benchmark,panel_hist
from iteration107_cliques_v1 import optical_x
from iteration107_fast_v1 import oracle,gaussian_click,ONE
from iteration107_projection_v1 import coefficients
from iteration109_sample_v1 import tv
from iteration110_collective_v1 import Collective,sparse,prepare


def prepare_mixed(lib,spectra,width,cut):
    width=np.ascontiguousarray(width,np.int32);d=len(width)
    if not 1<=d<=8 or np.any(width<1) or not 0<=cut<=512 or math.comb(cut+d,d)>8000000:raise ValueError('Invalid collective layout')
    if np.shape(spectra)[1:]!=(2*d,2*d) or not len(spectra) or (1+2*len(spectra))**d>500000:raise ValueError('Unsupported determinant polynomial')
    omega=np.kron(np.array([[0,1],[-1,0]]),np.eye(d));poly=np.ones((1,)*d);logvac=0;e=[]
    for x in spectra:
        if not np.isfinite(x).all() or np.max(abs(x-x.T))>1e-10 or np.linalg.eigvalsh(np.eye(2*d)+2*x+1j*omega).min()<-1e-9:raise ValueError('Unphysical Gaussian covariance')
        v,det,_=coefficients(np.eye(2*d)+x);logvac-=det/2;e.extend(np.linalg.eigvalsh(x))
        poly=convolve(poly,v,method='direct')
    assert abs(poly.flat[0]-1)<1e-12
    poly.flat[0]=0;index=np.argwhere(poly!=0);values=np.ascontiguousarray(poly[tuple(index.T)])
    index=np.ascontiguousarray(index,np.int32);size=math.comb(cut+d,d)
    states=np.empty((size,d),np.uint16);p=np.empty((3,size))
    fn=lib.mixed110;ptr=ct.c_void_p;fn.argtypes=[ct.c_int]*4+[ptr,ptr,ct.c_double,ptr,ct.c_int]+[ptr]*4
    assert fn(d,cut,size,len(values),index.ctypes.data,values.ctypes.data,float(np.exp(logvac)),width.ctypes.data,12,
              states.ctypes.data,*[v.ctypes.data for v in p])==0
    e=np.array(e);maxe=e.max()
    def gt(z):return -np.log1p((1-z)*e).sum()/2-(cut+1)*np.log(z)
    tail=0. if maxe<=0 else float(np.exp(min(0.,minimize_scalar(gt,bounds=(1,1+.999999/maxe),method='bounded').fun)))
    neg=float(-np.minimum(p[0],0).sum());assert neg<1e-10 and max(abs(p[1:].sum(1)-p[0].sum()))<1e-10
    plans=[];stats=[]
    for v in (p[2],p[1]):
        plan,stat=sparse(v,states,2e-13);plans.append(plan);stats.append(stat)
    return plans,dict(tail=tail,negative_mass=neg,terms=len(values),tables=stats),states,p


def run(out):
    if out.exists() and any(p.name!='mixed.dll' for p in out.iterdir()):raise FileExistsError(out)
    out.mkdir(exist_ok=True,parents=True);compiler=shutil.which('g++')
    subprocess.run([compiler,'-O3','-ffp-contract=off','-std=c++20','-march=x86-64-v3','-fopenmp','-shared',str(ROOT/'native/gaussian_core110_v1.cpp'),'-o',str(out/'mixed.dll')],check=True)
    rt=os.add_dll_directory(str(Path(compiler).parent)) if os.name=='nt' else None
    lib=ct.CDLL(str(out/'mixed.dll'));core=DATA/'iteration110_collective_v1';base=np.load(core/'optical.npz')
    pure=ct.CDLL(str(core/'laws.dll'));width=[4]*4;cut=68
    st=perf_counter();plans,meta,states,p=prepare_mixed(lib,base['core_x'][None],width,cut);setup=perf_counter()-st
    # Resolve every Fock tuple against the pure-state route with independent losses.
    u,r,eta=[base[k] for k in ('unitary','squeezing','efficiency')];original,_=prepare(pure,u,r,eta,width,cut)
    gaps=[]
    for j,law in enumerate((p[2],p[1])):
        s,c=original[j];keys=np.array([1,128,16384,2097152]);order=np.argsort(states@keys);ids=np.searchsorted((states@keys)[order],s@keys)
        q=np.zeros(len(states));q[order[ids]]=np.diff(c,prepend=np.uint64(0)).astype(float)/2**52
        gaps.append(tv(q,law));assert gaps[-1]<1e-9
    t,sq=actual_input();ports=np.array([0,272,280,856]);x=optical_x(t[ports],sq)
    mixed,met,ss,pp=prepare_mixed(lib,x,[1]*4,32)
    ref=gaussian_click(oracle(True),x,ONE);sc,cd=mixed[0];w=np.diff(cd,prepend=np.uint64(0)).astype(float)/2**52
    direct=np.bincount(sc@(1<<np.arange(4)),weights=w,minlength=16)
    threshold=np.bincount((ss>0)@(1<<np.arange(4)),weights=pp[1],minlength=16)
    assert max(tv(ref,direct),tv(ref,threshold))<1e-10
    rows=[];hist=[]
    for cells in (1,288,32768):
        engine=Collective(mixed,[1]*4,cells,core/'gpu.dll');batch=min(2048,4194304//(cells*4))
        timing=[benchmark(lambda n,s,o:engine.draw(n,pn,s,o),batch,110430000+cells*2+pn,9) for pn in (0,1)]
        bound=cells*(met['tail']+max(i['dropped']+i['rounding'] for i in met['tables']))
        rows.append(dict(cells=cells,ports=cells*4,mean_photons=float(cells*np.trace(x,axis1=1,axis2=2).sum()/2),
                         us=[v['us'] for v in timing],batch=batch,ideal_budget_tv_upper=bound))
        if cells==1:
            for pn in (0,1):
                hh=np.zeros(16,int)
                for off in range(0,65536,2048):hh+=panel_hist(engine.draw(2048,pn,110440000+pn,off),np.arange(4))
                hist.append(hh)
    np.savez_compressed(out/'mixed.npz',states=ss,optical_four_pnr=pp[1],click=direct,threshold=threshold,reference=ref,histograms=hist)
    save(out,'summary.json',dict(cross_checks=gaps,core_states=len(states),independent_setup_s=setup,independent=meta,physical_ports=ports.tolist(),
        mixed=met,source_spectra=len(x),direct_vs_reference=tv(ref,direct),threshold_vs_reference=tv(ref,threshold),scaling=rows));manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(ap.parse_args().output.resolve())
