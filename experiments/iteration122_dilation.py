"""Apply a calibrated clamped sampler to a different dominated transfer."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

from iteration122_powers import OUT,optics,json_write
from iteration122_family_certificate import positive_hermitian

FOLDER=OUT/'dominated'


def example():
    t,ss=optics('high')
    val,u=np.linalg.eigh(t.conj().T@t)
    root=(u*np.sqrt(val))@u.conj().T
    rng=np.random.default_rng([122,250])
    def iso(m,n):
        return np.linalg.qr(rng.standard_normal((m,n))+1j*rng.standard_normal((m,n)),mode='reduced')[0]
    new=(iso(96,50)*np.sqrt(np.linspace(.32,.87,50)))@iso(50,50)@root
    FOLDER.mkdir(exist_ok=True)
    np.savez_compressed(FOLDER/'target.npz',T=new)


def complete(target,reference):
    delta=reference.conj().T@reference-target.conj().T@target
    val,u=np.linalg.eigh(delta)
    if val[0]<-1e-12:raise ValueError('Target is outside the reference Gram order')
    enlarged=np.vstack([target,(u*np.sqrt(val.clip(0)))@u.conj().T])
    err=np.max(abs(enlarged.conj().T@enlarged-reference.conj().T@reference))
    if err>1e-12:raise ArithmeticError(err)
    return enlarged,float(err)


def check():
    from flint import acb,acb_mat,ctx
    from iteration120_interval import optical_data,exact
    ctx.prec=180;ctx.threads=2
    _,_,_,g,*_=optical_data(OUT/'high/input.npz')
    with np.load(FOLDER/'target.npz') as f:t=f['T']
    m,r=t.shape
    a=acb_mat(m,r,[acb(exact(z.real),exact(z.imag)) for z in t.ravel()])
    pivot=positive_hermitian(g-a.conjugate().transpose()*a)
    old,ss=optics('high')
    _,err=complete(t,old)
    info=dict(observed_ports=m,hidden_ports=r,smallest_positive_schur_pivot_lower=pivot,
              maximum_completed_gram_residual=err,reference_power='high',
              visible_mean=float(sum((abs(t)**2@np.sinh(s)**2).sum() for s in ss)),
              full_tv_upper='0.004815')
    json_write(FOLDER/'gram_certificate.json',info)
    print(info,flush=True)


def sample(stage,route,n):
    import iteration120_generate as gen
    with np.load(FOLDER/'target.npz') as f:target=f['T']
    ref,ss=optics('high')
    full,_=complete(target,ref)
    gen.OUT=FOLDER
    if stage=='base':
        gen.generate(route,n,10,(full,ss),(12+route)*2**22)
    else:
        gen.score(route,n,2,(full,ss),1229)
        # The full refined record, including auxiliary loss outputs, is scored
        # before this marginal is exposed to the caller.
        name='click' if route else 'pnr'
        with np.load(FOLDER/f'full_{route}_{n}.npz') as f:
            np.savez_compressed(FOLDER/(name+'.npz'),records=f[name][:,:len(target)])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['example','check','base','score'])
    p.add_argument('route',type=int,choices=[0,1],nargs='?')
    p.add_argument('--candidates',type=int,default=128)
    args=p.parse_args()
    with threadpool_limits(1):
        if args.stage=='example':example()
        elif args.stage=='check':check()
        else:sample(args.stage,args.route,args.candidates)
