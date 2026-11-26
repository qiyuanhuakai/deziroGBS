"""Optical total-photon PMF and the first two conditional moments of the number of detector clicks."""
import argparse
import ctypes as ct
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,ORDER,Native,realify,I,D,bind,save,manifest
from iteration103_sample_v1 import BASE,DATA

OPT=DATA/'iteration104_optical_v1'


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    src=ROOT/'native/click_moments104_v1.cpp'
    dll=out/'moments.dll'
    subprocess.run([shutil.which('g++'),'-O3','-std=c++20','-fopenmp','-shared','-ffp-contract=off',str(src),'-o',str(dll)],check=True)
    runtime=Native(BASE/'spatial.dll')
    moments=bind(ct.CDLL(str(dll)),'moments104',[I,D,D,D],None)
    z=np.load(ORDER)
    v=z['v'][np.argsort(z['order'])]
    m=len(v)
    a=realify(v)[np.c_[np.arange(m),m+np.arange(m)].ravel()]
    assert np.max(abs(a.T@a-np.eye(a.shape[1])))<1e-13
    parts=[np.linalg.eigh(x) for x in z['x']]
    modes=[np.ascontiguousarray(a@u) for _,u in parts]
    nfft=1024
    freq=np.zeros((3,nfft//2+1),complex)
    skip=np.zeros(3)
    local_error=0.
    used=0
    tic=perf_counter()
    for j in range(freq.shape[1]):
        w=np.exp(-2j*np.pi*j/nfft) if j not in (0,nfft//2) else complex(1 if j==0 else -1)
        fac=1.
        eta=1.
        vals=[]
        for e,_ in parts:
            d=1+(1-w)*e
            fac*=np.exp(-.5*np.log(d).sum())
            vals.append(w*e/d)
            eta*=min(1.,float(((1+e)/d).real.min()))
        freq[0,j]=fac
        # Accretivity supplies a lower singular-value bound for every principal
        # submatrix. This bounds the actual discarded Fourier coefficients.
        bound=abs(fac)*np.array([1.,m*(1+1/eta),m*(m-1)*(1+2/eta+1/eta**2)])
        if bound[-1]<1e-12:
            skip+=2*bound/nfft
            continue
        used+=1
        mats=[]
        for t,b in zip(vals,modes):
            mat=(b*t.real)@b.T+1j*((b*t.imag)@b.T)
            mat.flat[::len(mat)+1]+=1.
            mats.append(mat)
        s=np.zeros(4)
        moments(m,mats[0].view(float),mats[1].view(float),s)
        freq[1:,j]=fac*s.view(complex)
        if j in (0,8,32):
            # Direct Gaussian determinants for six observed ports, retaining
            # the complete optical state in the total-number generating factor.
            sub=[np.ascontiguousarray(mat[:12,:12]) for mat in mats]
            moments(6,sub[0].view(float),sub[1].view(float),s)
            def vacuum(ports):
                ids=[k for g in ports for k in (2*g,2*g+1)]
                return np.prod([np.linalg.det(mat[np.ix_(ids,ids)])**-.5 for mat in sub])
            single=[vacuum([i]) for i in range(6)]
            exact=[sum(1-x for x in single),2*sum(1-single[i]-single[k]+vacuum([i,k]) for i in range(6) for k in range(i))]
            local_error=max(local_error,float(np.max(abs(s.view(complex)-exact))))
    co=np.fft.irfft(freq,axis=1)
    exact=np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf']
    assert abs(exact[:nfft]-co[0]).max()<1e-13 and local_error<1e-11
    # For N>=1024 the probability is Chernoff bounded using a real radius.
    r=1.12
    tail=float(np.exp(sum(-.5*np.log1p((1-r)*e).sum() for e,_ in parts)-nfft*np.log(r)))
    mu=float(freq[1,0].real)
    variance=float(freq[2,0].real+mu-mu**2)
    assert abs(co[1].sum()-mu)<1e-10
    stable=exact[:nfft]>1e-10
    n=np.arange(nfft)[stable]
    mean=co[1,stable]/exact[n]
    var=(co[2,stable]+co[1,stable])/exact[n]-mean**2
    assert (mean>=0).all() and (mean<=n+1e-5).all() and var.min()>-1e-4
    np.savez_compressed(out/'moments.npz',prob=exact[:nfft],click_mass=co[1],factorial_mass=co[2],frequency=freq,stable=stable)
    save(out,'summary.json',dict(modes=m,spectra=len(parts),fft=nfft,frequencies_evaluated=used,seconds=perf_counter()-tic,mean_clicks=mu,click_variance=variance,photon_tail_bound=tail,discarded_coefficient_bounds=skip.tolist(),six_port_direct_max_error=local_error,photon_mass_difference=float(abs(exact[:nfft]-co[0]).max()),stable_number_range=[int(n.min()),int(n.max())],stable_probability=float(exact[n].sum())))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    with threadpool_limits(limits=6,user_api='blas'):main(p.parse_args().output.resolve())
