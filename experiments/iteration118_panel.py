"""Full stress-panel click probabilities and joint bank PNR, from two Cauchy contours."""
import argparse
import ctypes as ct
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from iteration115_visible_v1 import DATA, ROOT, hardware
from iteration117_quartic_laws import marginal, physical_models
from iteration118_auxiliary import OUT, prepare
from iteration118_laws import factor_polynomial


def library(directory):
    dll=Path(directory)/("panel.dll" if os.name=="nt" else "panel.so")
    subprocess.run(["g++","-std=c++17","-O3","-fopenmp","-shared","-fPIC",
                    str(ROOT/"native/panel_void118.cpp"),"-o",str(dll)],check=True)
    if os.name=="nt":library.runtime=os.add_dll_directory(str(Path(shutil.which("g++")).parent))
    lib=ct.CDLL(str(dll));lib.panel_void118.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_void_p]
    lib.panel_void118.restype=None
    return lib


def count_grid(cov,z):
    # Eliminate bank A once per first count variable. Eigenvalues of the
    # resulting bank B block then give every node in the second variable.
    a,b,c=cov[:16,:16],cov[:16,16:],cov[16:,16:]
    w=1-z;eye=np.eye(16)
    left=eye+w[:,None,None]*a
    eff=c-w[:,None,None]*(b.T@np.linalg.solve(left,np.broadcast_to(b,left.shape)))
    e=np.linalg.eigvals(eff)
    return np.exp(-.5*(np.log1p(w[:,None]*np.linalg.eigvals(a)).sum(1)[:,None]+
                       np.log1p(w[None,:,None]*e[:,None,:]).sum(2)))


def one_spectrum(model,k,retained,lib,ports,n,rho,grid):
    S,_,t,noise=marginal(model,ports);r=len(S)
    v=.5*np.vstack((t,-1j*t));V=np.column_stack((v.conj(),v))
    eyes=np.eye(r);eig=np.linalg.eigvalsh(S@S.conj().T)
    norm=retained*np.exp(-.5*np.log1p(-eig).sum())
    roots=np.exp(2j*np.pi*np.arange(n)/n);nodes=rho*roots
    weights=sum(nodes**(-j) for j in range(k+1))/n
    zz=np.exp(2j*np.pi*np.arange(grid)/grid)
    order=np.arange(32).reshape(2,16).T.ravel()
    den=np.eye(32)+noise
    checks={1,257,12659,65535}
    exact={}
    for mask in checks:
        ids=np.flatnonzero(mask&(1<<np.arange(16)));axes=np.r_[ids,16+ids]
        mat=den[np.ix_(axes,axes)]
        exact[mask]=(np.exp(-.5*np.linalg.slogdet(mat)[1])*
            factor_polynomial(S,t[ids],np.linalg.inv(mat),k).sum()/norm).real
    vv=np.zeros(1<<16,complex);pair=np.zeros((grid,grid),complex)
    # xy is constant along each antidiagonal. Invert the 50-by-50 block once.
    for d,phase in enumerate(roots):
        xy=rho**2*phase;inv=np.linalg.inv(eyes-xy*S.conj()@S)
        l=inv@S.conj();rr=(np.linalg.inv(eyes-xy*S@S.conj()))@S
        mix=xy*l@S
        A,B=V[:,:r],V[:,r:]
        base=noise+2*(A@mix@B.T+B@mix.T@A.T)
        da=2*A@l@A.T;db=2*B@rr@B.T
        j=(d-np.arange(n))%n
        matrices=base+nodes[:,None,None]*da+nodes[j,None,None]*db
        matrices=np.ascontiguousarray(matrices[:,order][:,:,order])
        w=weights*weights[j]*np.exp(-.5*np.log1p(-xy*eig).sum())/norm
        void=np.empty((n,1<<16),complex)
        D=np.ascontiguousarray(matrices+np.eye(32))
        lib.panel_void118(D.ctypes.data,n,16,3,void.ctypes.data)
        vv+=w@void
        for coef,cov in zip(w,matrices):pair+=coef*count_grid(cov,zz)
    err=max(abs(vv[m]-exact[m]) for m in checks)
    assert err<1e-10 and abs(vv.imag).max()<1e-11,(err,abs(vv.imag).max())
    # Fourier summation folds sectors j+l*n into j with amplitude rho**(l*n).
    # The folded extra ket is orthogonal to the retained one; disjoint high
    # sectors and Cauchy--Schwarz bound its norm even without a tail estimate.
    overlap_mass=rho**(2*n)/(1-rho**(2*n))
    radii=np.linspace(1.01,.999/eig.max(),150)
    source_tail=min(np.exp(.5*np.sum(np.log1p(-eig)-np.log1p(-x*eig))-n*np.log(x)) for x in radii)
    tail=overlap_mass*min(1-retained,source_tail)
    # Unnormalized excess mass, relative to the exact normalization.
    folded_bound=np.sqrt(tail/retained)+tail/(2*retained)
    radius=1.25;w=1-radius;e,u=np.linalg.eigh(noise)
    assert min(1+w*e)>0
    expcount=(factor_polynomial(S,t,(u*(w/(1+w*e)))@u.T,k).sum()*
              np.exp(-.5*np.log1p(w*e).sum())/norm).real
    return vv.real,pair,folded_bound,float(expcount),float(err)


def run(n,rho,k,weak,banks):
    optical=hardware();ports=(8*np.array(banks)[:,None]+np.arange(8)).ravel()
    grid=128
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        lib=library(tmp);results=[]
        for s,model,cap in zip(optical[1],physical_models(optical),(k,weak)):
            retained=prepare(optical[0],s,cap)[-1]
            results.append(one_spectrum(model,cap,retained,lib,ports,n,rho,grid))
        # Windows cannot unlink a loaded library until the process has exited.
        if os.name=="nt":
            import _ctypes
            _ctypes.FreeLibrary(lib._handle)
    raw=(results[0][0]*results[1][0])[::-1].copy().astype(np.longdouble)
    for h in range(16):
        pair=raw.reshape(-1,2,1<<h);pair[:,1]-=pair[:,0]
    clicks=np.array(raw,float)
    counts=np.fft.fft2(results[0][1]*results[1][1])/grid**2
    assert abs(counts.imag).max()<1e-12 and abs(counts.sum()-1)<3e-10
    assert abs(clicks.sum()-1)<3e-10 and np.minimum(clicks,0).sum()>-1e-8
    truth=np.load(DATA/"iteration115_review_v3/projection_panels.npz")["truth_"+"_".join(map(str,banks))]
    if banks==[34,35]:reference=np.load(DATA/"iteration115_panel_accuracy_v2/pnr_laws.npz")["optical_bank"]
    else:
        from iteration106_field_v1 import actual_input
        from iteration107_cliques_v1 import optical_x
        from iteration115_panel_accuracy_v2 import bank_counts
        raw,gains=actual_input()
        reference=bank_counts(optical_x(raw[ports],gains),[*range(8),*range(16,24)])
    alias_model=np.prod([x[3] for x in results])*1.25**(-grid)
    trunc=sum(x[2] for x in results)+2*results[0][2]*results[1][2]
    pnr=float(abs(reference-counts.real).sum()/2)
    click=float(abs(clicks-truth).sum()/2)
    info=dict(banks=banks,pair_caps=[k,weak],cauchy_nodes=n,radius=rho,grid=grid,
              full_panel_click_tv=click,bank_pnr_modulo_tv=pnr,folded_state_allowance=float(trunc),
              bank_count_model_alias=float(alias_model),direct_subset_checks=[x[4] for x in results],
              negative_click_mass=float(-np.minimum(clicks,0).sum()))
    print(info,flush=True)
    name=f"panel_{k}_{weak}_{n}"+("" if banks==[34,35] else "_"+"_".join(map(str,banks)))
    np.savez_compressed(OUT/(name+".npz"),click=clicks,bank_pnr=counts.real)
    (OUT/(name+".json")).write_text(json.dumps(info,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--nodes",type=int,default=28)
    p.add_argument("--radius",type=float,default=.62);p.add_argument("--caps",type=int,nargs=2,default=[7,5]);p.add_argument("--banks",type=int,nargs=2,default=[34,35]);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.nodes,a.radius,*a.caps,a.banks)
