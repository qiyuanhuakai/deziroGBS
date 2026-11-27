"""Finite-width operational bound for arbitrary phases on a uniformly shared bright cell."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration111_memory_v1 import realify
from iteration111_sampling_v1 import pnr_table,DATA
from iteration111_heterogeneous_v1 import shared


def optics(buses):
    op=np.load(DATA/'iteration110_collective_v1/optical.npz')
    u,r,eta=[op[k] for k in ('unitary','squeezing','efficiency')]
    a,b=np.diag(np.sqrt(eta)),np.diag(np.sqrt(1-eta))
    loc=realify(np.block([[a@u,b],[-b@u,a]]))*np.exp(np.r_[-r,np.zeros(4),r,np.zeros(4)])
    vis=loc[np.r_[0:4,8:12]]
    bus=np.load(DATA/'iteration110_bus_v1/laws.npz')['bus']
    rng=np.random.default_rng(112220000)
    v=np.column_stack((bus,rng.normal(size=(4,3))+1j*rng.normal(size=(4,3))))[:,:buses]
    v/=np.linalg.norm(v,axis=0)
    w=np.eye(4,dtype=complex)
    for col,theta in zip(v.T,(1.,-.94,1.18,-.81)):
        w=(np.eye(4)+np.expm1(1j*theta)*np.outer(col,col.conj()))@w
    s,axes,_=shared([loc],[np.eye(4)],w)
    ket=vis@np.vstack((axes/2,axes/(2j)))
    x=op['core_x']
    return x,realify(w)@x@realify(w).T,s,ket


def click_effects(x,ket):
    r=ket.shape[1]
    vacuum=np.ones(16)
    effects=np.tile(np.eye(r,dtype=complex),(16,1,1))
    for mask in range(1,16):
        sites=np.flatnonzero((1<<np.arange(4))&mask)
        ids=np.r_[sites,sites+4]
        q=np.eye(len(ids))+x[np.ix_(ids,ids)]
        vacuum[mask]=1/np.sqrt(np.linalg.det(q))
        k=ket[ids]
        effects[mask]=vacuum[mask]*(np.eye(r)-2*k.T@np.linalg.solve(q,k.conj()))
    def invert(v):
        y=v[::-1].copy()
        for j in range(4):
            t=y.reshape(-1,2,1<<j,*y.shape[1:])
            t[:,1]-=t[:,0]
        return y
    return invert(vacuum),invert(effects)


def pnr_effects(x,ket,radix,keep):
    d,r=ket.shape
    grid=np.indices((radix,)*4).reshape(4,-1).T
    kernel=np.eye(d)-np.linalg.inv(np.eye(d)+x)
    base=np.empty(len(grid),complex)
    i,j=np.triu_indices(r)
    # The complex fugacity kernel need not be Hermitian until after the DFT.
    projected=np.empty((len(grid),len(i)),complex)
    for start in range(0,len(grid),2048):
        sl=slice(start,start+2048)
        z=np.tile(np.exp((-2j*np.pi/radix)*grid[sl]),(1,2))
        ev=np.linalg.eigvals(z[:,:,None]*kernel)
        base[sl]=np.exp(-.5*np.log1p(-ev).sum(1))/np.sqrt(np.linalg.det(np.eye(d)+x))
        t=1-z
        invket=np.linalg.solve(np.eye(d)+t[:,:,None]*x,t[:,:,None]*ket.conj())
        mat=np.eye(r)-2*ket.T@invket
        projected[sl]=base[sl,None]*mat[:,i,j]
    def transform(v,target):
        a=np.fft.ifftn(v.reshape((radix,)*4)).ravel(order='F')[keep].copy()
        a[0]+=target-a.sum()
        return a
    q=transform(base,1).real
    e=np.empty((len(q),r,r),complex)
    for k,(a,b) in enumerate(zip(i,j)):
        val=transform(projected[:,k],float(a==b))
        e[:,a,b]=val
        e[:,b,a]=val.conj()
    return q,e


def overlap_factor(q,e):
    d=e.shape[1]
    assert q.min()>0 and abs(q.sum()-1)<1e-10
    np.testing.assert_allclose(e.sum(0),np.eye(d),atol=2e-10)
    mn=float(np.linalg.eigvalsh(e).min())
    assert mn>-2e-12
    # Indices (a,c),(b,d) implement the tensor product E_y ⊗ E_y.
    flat=e.reshape(len(q),-1)
    operator=((flat.T/q)@flat).reshape(d,d,d,d).transpose(0,2,1,3).reshape(d*d,d*d)
    eig=np.linalg.eigvalsh(operator)
    return float(eig[-1]),mn


def number_law(s,cut):
    t=np.tanh(s)**2
    p=[float(1/np.cosh(s).prod())]
    for n in range(1,cut//2+1):
        p.append(sum(np.sum(t**j)*p[n-j] for j in range(1,n+1))/(2*n))
    return np.array(p)


def certificate(s,cells,lam,cut=48):
    p=number_law(s,cut)
    ns=2*np.arange(len(p))
    clean=np.ones(len(ns))
    for k,n in enumerate(ns):
        clean[k]=np.prod(1-np.arange(n)/cells) if n<=cells else 0.
    mass=p*clean
    # Exclude N>cut before the parity measurement. Projection onto vacuum/single
    # per cell removes coherence with other occupied subsets. Split the two costs:
    # first project away collisions (gentle bound), then discard large clean N (TV).
    tail=max(0.,1-p.sum())
    collision=min(1.,float(np.dot(p,1-clean)+tail))
    tail_clean=min(1.,float(tail/mass.sum()))
    rho=mass/mass.sum()
    candidates=[]
    # Clean sectors have classical N, so the truncation for the overlap bound
    # can be chosen independently and charged linearly.
    for ncap in ns[1:]:
        ns2=ns[ns<=ncap]
        prob=rho[:len(ns2)];keep=prob.sum()
        terms=[]
        for t in range(1,min(int(ncap),cells)+1):
            moment=sum(a*math.comb(int(n),t) for a,n in zip(prob,ns2) if n>=t)/keep
            terms.append(math.exp(2*math.log(moment)+t*math.log(lam-1)-math.log(math.comb(cells,t))) if moment>0 and lam>1 else 0.)
        candidates.append((1-keep+.5*math.sqrt(sum(terms)),int(ncap)))
    dilute,ncap=min(candidates)
    return dict(collision_probability_upper=collision,collision_trace_budget=math.sqrt(collision),
                clean_tail_budget=tail_clean,hidden_excitations_tv_budget=dilute,overlap_cut=ncap,
                optical_measurement_tv_budget=min(1.,math.sqrt(collision)+tail_clean+dilute))


def marginal_overflow(x,radix):
    tail=0.
    for i in range(4):
        e=np.linalg.eigvalsh(x[np.ix_([i,i+4],[i,i+4])])
        a=e/(1+e);p=[1/np.sqrt(np.prod(1+e))]
        for n in range(1,radix):
            p.append(sum(np.sum(a**j)*p[n-j] for j in range(1,n+1))/(2*n))
        tail+=max(0.,1-math.fsum(p))
    return tail


def run(out,radix):
    out.mkdir(parents=True,exist_ok=True)
    x=optics(1)[0]
    zero=np.zeros((1,1,8),complex)
    base=pnr_table(x,zero,radix)[:,0,0].real
    order=np.argsort(-base)
    keep=order[:np.searchsorted(np.cumsum(base[order],dtype=np.longdouble),1-3e-9)+1]
    assert keep[0]==0
    rows=[];arrays=dict(keep=keep)
    for buses in (1,2,4):
        x,bright,s,ket=optics(buses)
        qc,ec=click_effects(x,ket)
        qp,ep=pnr_effects(x,ket,radix,keep)
        for label,q,e in [('click',qc,ec),('pnr',qp,ep)]:
            lam,mn=overlap_factor(q,e)
            arrays[f'{label}_q']=q
            for cells in (4,256,2048,32768,262144):
                extra=0.
                if label=='pnr':
                    local=x+(bright-x)/cells
                    marg=pnr_table(local,zero,radix)[:,0,0].real
                    dropped=max(0.,float(1-np.sum(marg[keep],dtype=np.longdouble)))
                    extra=min(1.,cells*(dropped+marginal_overflow(local,radix)))
                b=certificate(s,cells,lam)
                row=dict(buses=buses,active=len(s),squeezings=s.tolist(),readout=label,ports=4*cells,cells=cells,
                         overlap_operator_norm=lam,min_effect_eigenvalue=mn,register_budget=extra,**b)
                row['full_tv_budget']=min(1.,extra+b['optical_measurement_tv_budget'])
                rows.append(row)
                print(buses,label,cells,round(row['full_tv_budget'],6),flush=True)
        np.savez_compressed(out/'local.npz',**arrays)
        (out/'dilution.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    ap.add_argument('--radix',type=int,default=24)
    args=ap.parse_args()
    with threadpool_limits(limits=3):run(args.output,args.radix)
