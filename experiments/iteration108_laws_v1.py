"""Optical joint laws for distinguished detectors and all remaining outputs.

The first two count axes index resolved physical detectors; the third is the
total in the complement. Direct bits 0..h-1 resolve physical detectors and
each subsequent bit denotes a nonempty range of remaining detectors.
"""
import numpy as np
from scipy.fft import ifftn,next_fast_len
from iteration100_spatial_v1 import realify
from iteration105_jiuzhang_v1 import xpair
from iteration106_field_v1 import factors,brightness
from iteration107_fast_v1 import gaussian_click,oracle,ONE
from iteration107_tables_v1 import kernel
from iteration107_resolved_law_v1 import chernoff


def modes(t,r):
    u,b=np.linalg.qr(t,mode='reduced')
    b=realify(b);k=t.shape[1];quad=[];classic=[]
    for row in r:
        x=np.zeros((2*k,2*k))
        for j,s in enumerate(row):
            ix=np.r_[2*j,2*j+1,k+2*j,k+2*j+1]
            x[np.ix_(ix,ix)]=xpair(s)
        fund=factors(np.eye(k,dtype=complex),row[None,:])[0]
        y=2*fund.T@fund
        quad.append(b@x@b.T);classic.append(b@y@b.T)
    return u,np.stack(quad),np.stack(classic)


def compressed_rows(u,idx):
    return realify(u[idx])


def click_law(u,x,profile,h,bounds,ref):
    # Unnormalized small-block click law after imposing vacuum in an
    # arbitrary union of whole outside groups. Möbius inversion gives the
    # full joint over flags without enumerating any individual outside bit.
    d=x.shape[1];head=compressed_rows(u,np.arange(h))
    grams=[realify(u[a:b].conj().T@u[a:b]) for a,b in zip(bounds[:-1],bounds[1:])]
    nr=len(grams);size=1<<h
    v=np.zeros((1<<nr,size),np.longdouble)
    for mask in range(1<<nr):
        g=sum((item for j,item in enumerate(grams) if mask>>j&1),np.zeros((d,d)))
        # Solve in the input subspace even for a 131,072-row output.
        for w,gain in zip(profile['weights'],profile['values']):
            blocks=[];log=0.
            for xs in x:
                eff=gain*xs;mat=np.eye(d)+eff@g
                sign,det=np.linalg.slogdet(mat)
                assert sign>0
                log-=det/2
                cond=head@np.linalg.solve(mat,eff@head.T)
                blocks.append((cond+cond.T)/2)
            v[mask]+=w*np.exp(log)*gaussian_click(ref,np.stack(blocks),ONE)
    law=v[::-1].copy()
    for j in range(nr):
        for i in range(1<<nr):
            if i>>j&1:law[i]-=law[i^(1<<j)]
    return np.array(law,float).ravel()


def eig_parts(u,x):
    head=compressed_rows(u,np.arange(2))
    out=[]
    for a in x:
        e,v=np.linalg.eigh(a);out.append((e,head@v))
    return out


def count_law(parts,profile,shape):
    # At fixed complement fugacity z, the remaining determinant is a
    # degree-(2,2) polynomial. Its coefficients are the 16 principal minors
    # of the two-detector tilted covariance. A phase-continuous log gives
    # the physical square root on the Fourier torus.
    ca,cb,cc=shape
    a,b,z=[np.exp(-2j*np.pi*np.arange(c)/c) for c in shape]
    ua=z[:,None,None]-a[None,:,None];ub=z[:,None,None]-b[None,None,:]
    result=np.zeros((cc,ca,cb),complex)
    for w,gain in zip(profile['weights'],profile['values']):
        log=np.zeros_like(result)
        for eig,v in parts:
            e=gain*eig
            den=1+(1-z[:,None])*e
            log-=.5*np.log(den).sum(1)[:,None,None]
            mat=np.einsum('ij,zj,kj->zik',v,e/den,v,optimize=True)
            poly=np.ones_like(result)
            for mask in range(1,16):
                ids=[j for j in range(4) if mask>>j&1]
                ia=sum(j%2==0 for j in ids);ib=len(ids)-ia
                coeff=np.linalg.det(mat[:,ids][:,:,ids])
                poly+=coeff[:,None,None]*(ua**ia)*(ub**ib)
            phase=np.angle(poly)
            for ax in (0,1,2):phase=np.unwrap(phase,axis=ax)
            log-=.5*(np.log(abs(poly))+1j*phase)
        result+=w*np.exp(log)
    raw=ifftn(result).transpose(1,2,0)
    assert abs(raw.imag).max()<3e-10 and abs(raw.real.sum()-1)<1e-8
    return raw.real


def cap(eps,e):
    hi=8
    while chernoff(e,hi)>eps:
        hi+=max(4,hi//5)
    return next_fast_len(hi)


def prepare(t,r,h=8,bulk=4,order=None,profile=None):
    t=np.asarray(t,complex);r=np.atleast_2d(r).astype(float)
    if t.ndim!=2 or r.ndim!=2 or r.shape[1]*2!=t.shape[1] or len(t)<2 or min(r.shape)<1 or not np.isfinite(t).all() or not np.isfinite(r).all():
        raise ValueError('Expected a finite paired-source input and at least two ports')
    if np.linalg.eigvalsh(t.conj().T@t).max()>1+1e-9:raise ValueError('Input transfer must be passive')
    m=len(t);order=np.arange(m) if order is None else np.asarray(order)
    if len(order)!=m or not np.array_equal(np.sort(order),np.arange(m)):raise ValueError('Invalid port order')
    h=min(h,m);bulk=min(bulk,m-h)
    if not 2<=h<=16 or not 0<=bulk<=6 or h+bulk>20 or (m>h and not bulk):raise ValueError('Invalid coarse partition')
    bounds=np.linspace(h,m,bulk+1,dtype=np.int32)
    u,target,base=modes(t[order],r)
    full=np.linalg.eigvalsh(target).ravel();other=np.linalg.eigvalsh(base).ravel()
    if profile is None:profile=[v for v in brightness(full,other) if v['feasible']][-1] if max(other)>1e-18 else ONE
    ref=oracle();ac=click_law(u,base,profile,h,bounds,ref);qc=click_law(u,target,ONE,h,bounds,ref)
    # The complement count is bounded by the all-port total. Each of the
    # single-detector bounds includes every spectrum and brightness atom.
    head=compressed_rows(u,np.arange(2));shape=[];maxgain=max(profile['values']);tails=[]
    for q,b in [([np.linalg.eigvalsh(head[[i,i+2]]@s@head[[i,i+2]].T) for s in target],
                  [np.linalg.eigvalsh(head[[i,i+2]]@s@head[[i,i+2]].T) for s in base]) for i in range(2)]+[(full,other)]:
        q=np.ravel(q);b=np.ravel(b);c=max(cap(1e-12/3,q),cap(1e-12/3,maxgain*b))
        shape.append(c);tails.append(max(chernoff(q,c),chernoff(maxgain*b,c)))
    if np.prod(shape)>5000000:raise ValueError('Photon table exceeds configured memory')
    ap=count_law(eig_parts(u,base),profile,shape);qp=count_law(eig_parts(u,target),ONE,shape)
    if m==2:
        ap[:,:,1:]=0;qp[:,:,1:]=0
    qa=qc.reshape(-1,1<<h).sum(0);cp=qp.sum(2)
    four=np.bincount(np.arange(len(qa))%4,weights=qa,minlength=4)
    xx,yy=np.indices(cp.shape)
    same=np.bincount(((xx>0)+2*(yy>0)).ravel(),weights=cp.ravel(),minlength=4)
    dc=abs(four-same).sum()/2
    vacuum=float(np.exp(-.5*np.log1p(full).sum()))
    assert max(dc,abs(qp[0,0,0]-vacuum),abs(qc[0]-vacuum))<5e-9
    cfg=dict(order=np.asarray(order,np.int32),h=np.array(h),bounds=bounds,shape=np.array(shape,np.int32),
             values=np.asarray(profile['values']),weights=np.asarray(profile['weights']))
    stats=dict(shape=shape,tail_bound=sum(tails),threshold_consistency=dc,vacuum=vacuum)
    for name,p,q in [('click',ac,qc),('pnr',ap.ravel(),qp.ravel())]:
        neg=float(max(-q[q<0].sum(),-p[p<0].sum()));assert neg<1e-8,(name,neg)
        mv,cd,tv,err=kernel(p,q);cfg[name+'_move']=mv;cfg[name+'_cdf']=cd
        stats[name]=dict(before=tv,after=err,negative_mass=neg)
    return cfg,stats,dict(click=qc,pnr=qp,base_click=ac,base_pnr=ap)
