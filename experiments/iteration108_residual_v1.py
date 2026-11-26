"""Quantify a feasible pure squeezed residual and classical quadrature noise on the actual input."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import manifest,save
from iteration106_field_v1 import actual_input
from iteration108_laws_v1 import modes


def split(v):
    d=len(v)//2;eye=np.eye(d);omega=np.block([[0*eye,eye],[-eye,0*eye]])
    e,b=np.linalg.eigh(v);root=(b*np.sqrt(e))@b.T
    a=root@omega@root
    w,u=np.linalg.eigh(-a@a);assert min(w)>1-1e-9
    pure=root@(u*(1/np.sqrt(w)))@u.T@root
    p,b=np.linalg.eigh(pure)
    def at(power):return (b*np.exp(np.log(p)*power))@b.T
    low=0.;high=1.
    for _ in range(52):
        mid=(low+high)/2
        if np.linalg.eigvalsh(v-at(mid)).min()>=0:high=mid
        else:low=mid
    power=min(1.,high+1e-11);pure=at(power);noise=v-pure
    symplectic=float(np.max(abs(pure@omega@pure-omega)))
    margin=float(np.linalg.eigvalsh(noise).min());assert margin>-1e-9 and symplectic<1e-9
    squeezing=.5*np.sort(np.log(p)*power)[d:]
    lam=np.tanh(squeezing)**2;vac=np.sqrt(np.prod(1-lam))
    pmf=[vac]
    for n in range(1,13):pmf.append(sum(np.sum(lam**j)*pmf[n-j] for j in range(1,n+1))/(2*n))
    return pure,noise,dict(power=power,symplectic_error=symplectic,min_noise_eigenvalue=margin,
                           output_photons=float((np.trace(v)-2*d)/4),residual_photons=float(np.sum(np.sinh(squeezing)**2)),
                           displacement_photons=float(np.trace(noise)/4),pair_tail_4=float(1-sum(pmf[:5])),
                           pair_tail_6=float(1-sum(pmf[:7])),vacuum=float(vac))


def run(out):
    out.mkdir(exist_ok=False,parents=True);t,r=actual_input()
    basis,x,_=modes(t,r);rows=[];arrays={}
    for j,mat in enumerate(x):
        v=np.eye(len(mat))+2*mat;p,y,met=split(v);rows.append(met)
        assert abs(p+y-v).max()<1e-12
        arrays[f'pure{j}']=p;arrays[f'noise{j}']=y
    np.savez_compressed(out/'decomposition.npz',**arrays)
    save(out,'summary.json',dict(spectra=rows,scope='Covariance decomposition on an orthonormal optical basis. Sampling the displaced pure state still requires coherent interference between its displacement and residual photons.'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(p.parse_args().output.resolve())
