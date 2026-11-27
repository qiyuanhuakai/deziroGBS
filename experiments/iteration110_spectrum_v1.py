"""Match the total-number generating function while keeping every spatial single-mode state fixed."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from iteration106_field_v1 import DATA,actual_input,optical
from iteration109_model_v1 import build,panel
from iteration109_global_v1 import total


def roots(x,power):
    eig,vec=np.linalg.eigh(x)
    positive=eig>np.maximum(1e-16,1e-13*eig[...,-1:])
    w=np.where(positive,np.maximum(eig,1e-250)**power,0)
    return (vec*w[...,None,:])@vec.swapaxes(-1,-2)


class Spectrum:
    def __init__(self,model):
        self.parts=[]
        for f,local,rot in zip(model['factors'],model['local'],model['rot']):
            k,m,_=f.shape
            ff=f.reshape(k,-1)
            u,s,v=np.linalg.svd(ff,full_matrices=False)
            keep=s>max(1e-14,s.max()*1e-12)
            s=s[keep];base=(s[:,None]*v[keep]).reshape(-1,m,2)
            blocks=np.einsum('kgi,kgj->gij',base,base)
            # Smooth changes of relative latent variance by spectral rank.
            rank=np.arange(len(s));basis=np.stack([np.exp(-rank/scale) for scale in (1.5,6.,20.)],axis=1)
            core=(rot*local[:,None,:])@rot.transpose(0,2,1)
            self.parts.append((base,basis,roots(blocks,.5),core))
        self.template=model

    def factors(self,coeff):
        result=[]
        for base,basis,restore,_ in self.parts:
            raw=base*np.exp(basis@coeff)[:,None,None]
            current=np.einsum('kgi,kgj->gij',raw,raw)
            transform=roots(current,-.5)@restore
            result.append(np.einsum('kgi,gij->kgj',raw,transform))
        return result

    def values(self,coeff):
        eig=[]
        for f,(_,_,_,core) in zip(self.factors(coeff),self.parts):
            k,m,_=f.shape;x=2*f.reshape(k,-1).T@f.reshape(k,-1)
            tiled=x.reshape(m,2,m,2);idx=np.arange(m);tiled[idx,:,idx,:]+=core
            eig.extend(np.linalg.eigvalsh(x))
        return np.array(eig)

    def materialize(self,coeff):
        return dict(self.template,factors=self.factors(coeff))


def fit(model,target):
    s=Spectrum(model);grid=np.array([.008,.018,.032,.05,.075,.10,.14,.19])
    z=1-np.exp(1j*grid)
    def spectrum(e):
        values=-.5*np.log1p(z[:,None]*e).sum(1)
        return np.r_[values.real,values.imag]
    weights=np.tile(np.exp(spectrum(target)[:len(grid)]),2)
    def error(c):return (spectrum(s.values(c))-spectrum(target))*weights
    sol=least_squares(error,np.zeros(3),bounds=(-1.5,1.5),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=100)
    return s,sol


def run(out):
    out.mkdir(parents=True,exist_ok=False);t,r=actual_input();target=optical(t,r)
    m=build(t,r,8,1);start=perf_counter();spec,sol=fit(m,target);elapsed=perf_counter()-start
    before=spec.values(np.zeros(3));after=spec.values(sol.x)
    p,pt=total(target);q,qt=total(after);b,bt=total(before)
    best=spec.materialize(sol.x)
    drift=max(abs(np.einsum('kgi,kgj->gij',f,f)-np.einsum('kgi,kgj->gij',g,g)).max() for f,g in zip(m['factors'],best['factors']))
    assert drift<1e-11
    np.savez_compressed(out/'laws.npz',optical=p,before=b,after=q,eig_optical=target,eig_before=before,eig_after=after,coeff=sol.x)
    data=dict(coeff=sol.x.tolist(),fit_s=elapsed,steps=sol.nfev,cost=sol.cost,converged=bool(sol.success),covariance_drift=drift,
              baseline_tv=float(abs(p-b).sum()/2),calibrated_tv=float(abs(p-q).sum()/2),
              calibrated_ks=float(abs((p-q).cumsum()).max()),alias=[pt,bt,qt],ranks=[len(x) for x in best['factors']])
    print(data,flush=True);save(out,'summary.json',data);manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(ap.parse_args().output.resolve())
