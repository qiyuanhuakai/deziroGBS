"""Shared displacement with a physical conditional Gaussian residual in each output block."""
import numpy as np
from iteration106_packed_v1 import compress
from iteration108_laws_v1 import modes
from iteration108_residual_v1 import split


def axes(n):return np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])


def pure_field(v):
    # A pure state's reciprocal eigenvalues share passive quadrature axes.
    # A classical Gaussian on the anti-squeezed axes preserves the normal
    # photon covariance (including individual mode intensities).
    a,f=np.linalg.eigh(v)
    w=np.where(a>1,(a+1/a-2)/2,0.)
    return (f*w)@f.T


def build(t,r,fan=1,alpha=1.):
    t=np.asarray(t,complex);r=np.atleast_2d(r)
    if t.ndim!=2 or r.ndim!=2 or not t.size or not r.size:raise ValueError('Expected nonempty transfer and spectra')
    m=len(t)
    if fan not in (1,2,4,8) or not np.isfinite(t).all() or not np.isfinite(r).all() or t.shape[1]!=2*r.shape[1] or m%fan or not 0<=alpha<=1:raise ValueError('Invalid optical layout')
    if np.linalg.eigvalsh(t.conj().T@t).max()>1+1e-9:raise ValueError('Transfer exceeds passivity')
    if fan>1:
        base,w,_=compress(t,(np.sinh(r)**2).sum(0),1,width=fan)
        power=(abs(w.ravel())**2).reshape(-1,fan);norm=power.sum(1)
        weights=np.divide(power,norm[:,None],out=np.full_like(power,1/fan),where=norm[:,None]>0)
        base=base*np.sqrt(norm[:,None])
    else:base=t;weights=np.ones((m,1))
    u,X,_=modes(base,r);g=len(base)
    row=np.stack((np.c_[u.real,-u.imag],np.c_[u.imag,u.real]),axis=1)
    factors=[];locals=[];rotations=[];gains=[]
    for x in X:
        pure,noise,_=split(np.eye(len(x))+2*x)
        rem=(pure-np.eye(len(x)))/2;sur=pure_field(pure)
        # The residual after subtracting an extra shared classical component
        # must obey V+iΩ >= 0 on every measured conditional mode.
        a=row@rem@row.transpose(0,2,1);b=row@sur@row.transpose(0,2,1)
        def feasible(y):
            v=np.eye(2)+2*(a-y[:,None,None]*b)
            return (np.linalg.det(v)>=1-2e-14)&(v[:,0,0]>0)&(v[:,1,1]>0)
        lo=np.zeros(g);hi=np.ones(g)
        for _ in range(48):
            mid=(hi+lo)/2;ok=feasible(mid);lo=np.where(ok,mid,lo);hi=np.where(ok,hi,mid)
        gain=alpha*lo;gains.append(gain)
        local=a-gain[:,None,None]*b
        val,vectors=np.linalg.eigh(local);rotations.append(vectors)
        locals.append(val)
        def factor(y):
            e,v=np.linalg.eigh(y);keep=e>1e-14;return (v[:,keep]*np.sqrt(e[keep]/2)).T
        source=np.concatenate((np.einsum('kd,gid->kgi',factor(noise/2),row),
                               np.einsum('kd,gid->kgi',factor(sur),row)*np.sqrt(gain)[None,:,None]))
        factors.append(source)
    # Preserve the quadrature axes separately for the analytical control.
    return dict(weights=weights,local=np.array(locals),rot=np.array(rotations),factors=factors,gains=np.array(gains),fan=fan,m=m)


def panel(model,ids,interference=True):
    ids=np.asarray(ids,dtype=int);g=ids//model['fan'];q=ids%model['fan'];n=len(ids)
    scale=np.sqrt(model['weights'][g,q]);blocks=[]
    for f,e,v in zip(model['factors'],model['local'],model['rot']):
        obs=f[:,g,:]*scale[None,:,None];mat=np.concatenate((obs[:,:,0],obs[:,:,1]),axis=1)
        shared=2*mat.T@mat
        core=(v*e[:,None,:])@v.transpose(0,2,1)
        local=np.block([[core[g[:,None],a,b]*(g[:,None]==g[None,:])*np.outer(scale,scale) for b in (0,1)] for a in (0,1)])
        if interference:blocks.append(shared+local)
        else:blocks.extend([shared,local])
    return np.ascontiguousarray(blocks)
