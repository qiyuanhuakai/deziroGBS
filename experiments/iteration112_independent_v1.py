"""Independent finite-register identities and tests of the marked-quasiparticle number sampler."""
import argparse
import ctypes as ct
import itertools
import json
import math
import os
import shutil
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration111_memory_v1 import prepare,branches,dyads,click_table,complete_clicks
from iteration112_records_v1 import DATA,I,U,CU,P,compositions,fock
from iteration112_dilution_v1 import optics,click_effects,pnr_effects,overlap_factor
from iteration112_coarse_v1 import fourier


def anova():
    op=np.load(DATA/'iteration110_collective_v1/optical.npz')
    ref=np.load(DATA/'iteration110_bus_v1/laws.npz')
    s,ax,observed=prepare(op,ref['bus'],1.)
    z,c,g,_=branches(s,ax,3);w=(np.outer(c,c)*g).ravel()
    x,bright,_,_=optics(1)
    table=click_table(x,dyads(z,observed,4))
    law=complete_clicks(table,w)
    q=click_table(x+(bright-x)/4,np.zeros((1,1,8),complex))[:,0,0].real
    base=np.kron(np.kron(q,q),np.kron(q,q))
    chi=float(np.dot((law-base)/base,law-base))
    d=table.reshape(16,-1)-q[:,None]
    kernel=(d.T/q)@d.conj()
    orders=[float(math.comb(4,k)*np.real(w@(kernel**k)@w)) for k in range(1,5)]
    assert abs(sum(orders)-chi)<1e-12
    return dict(direct_chi_square=chi,orthogonal_orders=orders,absolute_difference=abs(sum(orders)-chi))


def derivative(x,basis,pnr):
    r=basis.shape[1]
    radix=6
    if pnr:q,e=pnr_effects(x,basis,radix,np.arange(radix**4))
    else:q,e=click_effects(x,basis)
    step=.0015
    errors=[]
    for i,j in ((0,0),(0,r-1),(r-1,0)):
        values=[]
        for a,b in itertools.product((-step,step),repeat=2):
            d=(a*basis[:,i]+b*basis[:,j].conj())
            law=(fourier(x,d[None],radix)[:,0] if pnr else click_table(x,d[None,None])[:,0,0])
            values.append(law*np.exp(a*b*(i==j)))
        diff=(values[3]-values[2]-values[1]+values[0])/(4*step**2)
        errors.append(float(np.max(abs(diff-e[:,i,j]))))
    assert max(errors)<4e-6
    return q,e,max(errors)


def reference(s,n,e):
    r=len(s);states=list(compositions(n,r))
    amps=fock(s,n)[-1][-len(states):]
    data=dict(zip(states,amps))
    psi=np.empty((r,)*n,complex)
    for idx in np.ndindex(*psi.shape):
        v=tuple(np.bincount(idx,minlength=r))
        mult=math.factorial(n)/math.prod(math.factorial(j) for j in v)
        psi[idx]=data[v]/math.sqrt(mult)
    rho=psi.reshape(r*r,-1)@psi.reshape(r*r,-1).conj().T
    p=np.einsum('iajb,yij,zab->yz',rho.reshape(r,r,r,r),e,e,optimize=True).real
    assert abs(p.sum()-1)<1e-10 and p.min()>-1e-12
    return p


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    rt=os.add_dll_directory(str(Path(shutil.which('g++')).parent)) if os.name=='nt' else None
    lib=ct.CDLL(str(DATA/'iteration112_records_v1/sparse.dll'))
    fn=lib.sparse_emit112;fn.argtypes=[I]*6+[U,CU]+[P]*12
    x,_,s,basis=optics(4);s=s[-3:];basis=basis[:,-3:]
    rows=[];arrays={}
    for pn in (0,1):
        q,e,err=derivative(x,basis,pn)
        if pn:
            # Aggregate by physical total photon counts to a sixteen-outcome POVM.
            sums=np.sum([np.arange(6**4)//6**i%6 for i in range(4)],axis=0)
            codes=np.minimum(15,sums)
            es=np.zeros((16,3,3),complex);np.add.at(es,codes,e);e=es
            q=np.bincount(codes,weights=q)
        eig,vec=np.linalg.eigh(e)
        factors=np.ascontiguousarray(vec*np.sqrt(np.maximum(eig,0))[:,None,:])
        prefix=np.ascontiguousarray(e.cumsum(0));words=np.arange(len(q),dtype=np.int32)
        lam,_=overlap_factor(q,e)
        for n in (2,4):
            cdf=np.zeros(n+1);cdf[-1]=1
            raw=reference(s,n,e)
            shots=131072;loc=np.empty((shots,n),np.int32);pack=np.empty_like(loc);num=np.empty(shots,np.int32)
            args=[cdf,*fock(s,n),prefix,factors,words,loc,pack,num]
            rc=fn(shots,32,3,n,len(q),8,112900000+100*pn+n,0,*[a.ctypes.data for a in args]);assert rc==0
            assert (num==n).all() and (np.diff(np.sort(loc,axis=1),axis=1)>0).all()
            hist=np.bincount(pack[:,0]*len(q)+pack[:,1],minlength=len(q)**2).reshape(raw.shape)
            diff=hist/shots-raw
            max_z=float(np.max(abs(diff)/np.sqrt(np.maximum(raw*(1-raw),1/shots)/shots)))
            assert max_z<6.5
            tv=float(abs(diff).sum()/2)
            row=dict(pnr=bool(pn),n=n,shots=shots,empirical_pair_tv=tv,max_standardized_error=max_z,finite_difference_error=err)
            if n==2:
                # Enumerate all 4-cell outcomes for two randomly located entangled photons.
                a=np.indices((16,)*4).reshape(4,-1)
                ratio=raw/np.outer(q,q)
                likelihood=sum(ratio[a[i],a[j]] for i,j in itertools.combinations(range(4),2))/6
                chi=float(np.sum(np.prod(q[a],axis=0)*(likelihood-1)**2))
                cap=sum(math.comb(n,k)**2*(lam-1)**k/math.comb(4,k) for k in (1,2))
                assert chi<=cap+1e-10
                row.update(marked_chi_square=chi,overlap_upper=cap)
            rows.append(row);arrays[f'{pn}_{n}_target']=raw;arrays[f'{pn}_{n}_hist']=hist
    np.savez_compressed(out/'checks.npz',**arrays)
    (out/'checks.json').write_text(json.dumps(dict(anova=anova(),marked=rows),indent=2)+'\n',encoding='utf8')
    print(rows,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2):run(ap.parse_args().output.resolve())
