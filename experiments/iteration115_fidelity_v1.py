"""Gaussian fidelity in the occupied optical subspace and whole-record bounds for the tilt mixture."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigvals
from threadpoolctl import threadpool_limits
from iteration115_banks_v1 import bank_model
from iteration115_visible_v1 import DATA,hardware,cov
from iteration111_memory_v1 import realify


def root_fidelity(v,w,tol=1e-7):
    # V,W here use vacuum I/2. The positive spectrum of 2i V_aux Omega
    # gives the symplectic factors in Banchi, Braunstein and Pirandola.
    d=len(v)//2;eye=np.eye(d)
    om=np.block([[0*eye,eye],[-eye,0*eye]])
    aux=om.T@np.linalg.solve(v+w,.25*om+w@om@v)
    z=eigvals(2j*aux@om)
    pos=np.sort(z.real)[d:]
    assert abs(z.imag).max()<1e-8 and pos.min()>1-1e-8
    # Pure spectator modes contribute zero. Dropping near-unit factors and
    # reducing the others reports the conservative numerical evaluation.
    log=np.arccosh(np.maximum(1.,pos-tol)).sum()-.5*np.linalg.slogdet(v+w)[1]
    return float(np.exp(.5*log))


def run(out):
    out.mkdir(parents=True,exist_ok=True);t,ss=hardware();m,r=t.shape
    bank=np.linalg.svd(t.reshape(144,8,r),full_matrices=False)[0][:,:,0]
    bright=np.zeros((m,144),complex)
    bright[np.arange(m),np.arange(m)//8]=bank.ravel()
    extra=t-bright@(bright.conj().T@t)
    dark=np.linalg.qr(extra,mode='reduced')[0]
    basis=np.column_stack((bright,dark))
    assert abs(basis.conj().T@basis-np.eye(basis.shape[1])).max()<1e-10
    assert abs(t-basis@(basis.conj().T@t)).max()<1e-10
    base,xb,_=bank_model(t,ss,8);rb=realify(basis)
    info=json.loads((DATA/'iteration115_tilt_v1/summary.json').read_text())
    rows=[];darkmass=[]
    for j,s in enumerate(ss):
        dim=144+r;indices=np.r_[0:144,dim:dim+144];di=np.r_[144:dim,dim+144:2*dim]
        xx=cov(basis.conj().T@t,s);mix=xx[np.ix_(indices,di)]
        den=np.eye(len(di))+xx[np.ix_(di,di)]
        prob=np.exp(-.5*np.linalg.slogdet(den)[1]);darkmass.append(prob)
        truth=.5*np.eye(288)+xx[np.ix_(indices,indices)]-mix@np.linalg.solve(den,mix.T)
        x=(rb.T@xb[j]@rb)[np.ix_(indices,indices)]
        e,u=np.linalg.eigh(x)
        fidelity=[];difference=[]
        for logz in [0.,*info['log_tilts']]:
            z=np.exp(logz);target=.5*np.eye(288)+(u*(z*e/(1+(1-z)*e)))@u.T
            f=root_fidelity(truth,target)
            f2=root_fidelity(target,truth,0.)
            difference.append(abs(f2-f))
            fidelity.append(float(np.sqrt(prob)*f))
        rows.append(dict(spectrum=j,dark_vacuum=prob,root_fidelity=fidelity,
                         max_reverse_unadjusted_gap=max(difference)))
    roots=np.prod([v['root_fidelity'] for v in rows],axis=0)
    mixture=float(roots[1:]@info['weights'])
    result=dict(components=rows,modes_in_union=basis.shape[1],vacuum_outside_bright=float(np.prod(darkmass)),
                base_trace_upper=float(np.sqrt(1-roots[0]**2)),mixture_root_fidelity_lower=mixture,
                mixture_trace_upper=float(np.sqrt(1-mixture**2)),symplectic_eigenvalue_margin=1e-7)
    # Thermal analytical formula and pure-state vacuum overlap independently fix the convention.
    for a,b in [(.2,.8),(0.,1.3)]:
        expected=1/(np.sqrt((1+a)*(1+b))-np.sqrt(a*b))
        assert abs(root_fidelity((.5+a)*np.eye(2),(.5+b)*np.eye(2),0)-expected)<2e-7
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=4):run(p.parse_args().output)
