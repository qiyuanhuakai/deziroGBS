"""Rank-reduced update for unequal cells and two overlapping passive supermode phases."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag
from threadpoolctl import threadpool_limits
from iteration111_memory_v1 import branches, click_table, dyads, realify
from iteration107_fast_v1 import gaussian_click, oracle, ONE
from iteration100_baselines_v1 import pair_reference


def shared(local, site_vectors, small):
    # Physical x/p and unobserved loss x/p are interleaved by cell only for this factorization.
    left,right=[],[]
    for s,v in zip(local,site_vectors):
        d=len(s)//4
        obs=np.r_[0:d,2*d:3*d]
        h=np.zeros((len(s),2*v.shape[1]))
        h[obs]=realify(v)
        left.append(np.linalg.solve(s,h))
        right.append(h.T@s)
    l=np.vstack(left)
    r=np.hstack(right)
    h=realify(small)-np.eye(r.shape[0])
    u,rr=np.linalg.qr(np.column_stack([l,r.T]),mode='reduced')
    form=np.block([[h@r@r.T@h.T,h],[h.T,np.zeros_like(h)]])
    update=rr@form@rr.T
    e,b=np.linalg.eigh(np.eye(len(update))+update)
    hot=e>1+1e-11
    ax=u@b[:,hot]
    pieces=np.split(ax,np.cumsum([len(a) for a in local])[:-1])
    axes=np.vstack([-1j*(p[:len(p)//2]+1j*p[len(p)//2:]) for p in pieces])
    assert np.max(abs(axes.conj().T@axes-np.eye(hot.sum())))<1e-10
    return np.log(e[hot])/2,axes,len(update)


def pair_counts(q,ds,radix):
    n=np.exp(-2j*np.pi*np.arange(radix)/radix)
    e,u=np.linalg.eigh(q)
    a=ds@u
    h=1+(1-n[:,None])*e
    base=np.exp(-.5*np.log(h).sum(1))
    # Contract separate components before the FFT; port 0 is the observed marginal here.
    series=base[:,None,None]*np.exp(-np.einsum('za,ija->zij',(1-n[:,None])/h,a*a))
    probs=np.fft.ifft(series,axis=0)
    return np.concatenate([probs[:10],probs[10:].sum(0,keepdims=True)])


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(111600001)
    local=[];xs=[];width=[3,4,2,4]
    for d in width:
        u=np.linalg.qr(rng.standard_normal((d,d))+1j*rng.standard_normal((d,d)))[0]
        eta=rng.uniform(.64,.97,d)
        a,b=np.diag(np.sqrt(eta)),np.diag(np.sqrt(1-eta))
        squeeze=rng.uniform(.24,.78,d)
        s=realify(np.block([[a@u,b],[-b@u,a]]))*np.exp(np.r_[-squeeze,np.zeros(d),squeeze,np.zeros(d)])
        local.append(s)
        visible=s[np.r_[0:d,2*d:3*d]]
        xs.append((visible@visible.T-np.eye(2*d))/2)
    ports=sum(width)
    vectors=rng.standard_normal((ports,2))+1j*rng.standard_normal((ports,2))
    vectors/=np.linalg.norm(vectors,axis=0)
    # A nonzero inner product makes the successive phase operations noncommuting.
    overlap=float(abs(vectors[:,0].conj()@vectors[:,1]))
    direct=block_diag(*xs)
    offsets=np.cumsum([0]+[2*d for d in width[:-1]])
    physical=np.concatenate([np.arange(d)+offset+j*d for j in (0,1) for d,offset in zip(width,offsets)])
    base=direct[np.ix_(physical,physical)]
    rows=[];arrays={}
    quad=oracle(True)
    for buses in (1,2):
        v,_=np.linalg.qr(vectors[:,:buses])
        small=np.eye(buses,dtype=complex)
        for direction,phase in zip(vectors.T[:buses],(1.1,-.82)):
            w=v.conj().T@direction
            small=(np.eye(buses)+np.expm1(1j*phase)*np.outer(w,w.conj()))@small
        sites=np.split(v,np.cumsum(width)[:-1])
        squeeze,axes,dim=shared(local,sites,small)
        f=realify(np.eye(ports)+v@(small-np.eye(buses))@v.conj().T)
        x=f@base@f.T
        truth=gaussian_click(quad,x[None],ONE)
        ignore=gaussian_click(quad,base[None],ONE)
        select=[0,3,ports,ports+3]
        pair=pair_reference((np.eye(2*ports)+x)[np.ix_(select,select)][None],10)
        arrays[f'truth{buses}']=truth;arrays[f'pair{buses}']=pair
        for m in ((3,6) if buses==1 else (2,3,4)):
            z,c,g,bound=branches(squeeze,axes,m)
            weight=np.outer(c,c)*g
            zz=np.split(z,np.cumsum([2*d for d in width])[:-1],axis=1)
            clicks=[];pairs=[]
            for i,(d,s,xx,ket) in enumerate(zip(width,local,xs,zz)):
                ds=dyads(ket,s[np.r_[0:d,2*d:3*d]],1)
                clicks.append(click_table(xx,ds))
                if i<2:
                    selected=[0,d]
                    pairs.append(pair_counts(xx[np.ix_(selected,selected)],ds[...,selected],64))
            # Least significant bits denote the earlier cell.
            p0=(clicks[1][:,None]*clicks[0][None]).reshape(2**sum(width[:2]),-1)
            p1=(clicks[3][:,None]*clicks[2][None]).reshape(2**sum(width[2:]),-1)
            model=((p1*weight.ravel())@p0.T).real.ravel()
            pp=((pairs[1].reshape(11,-1)*weight.ravel())@pairs[0].reshape(11,-1).T).real.ravel()
            tv=float(abs(model-truth).sum()/2)
            assert abs(model.sum()-1)<2e-11 and model.min()>-1e-13 and tv<=bound+1e-11
            assert abs(pp.sum()-1)<2e-11 and pp.min()>-1e-13
            rows.append(dict(buses=buses,reduced_eigensystem=dim,active=len(squeeze),nodes=m,branches=len(c),
                    click_tv=tv,omitted_bus_tv=float(abs(ignore-truth).sum()/2),state_bound=bound,
                    pair_count_model_tv=float(abs(pp-pair).sum()/2),bus_overlap=overlap))
            print(buses,m,tv,flush=True)
            arrays[f'model{buses}_{m}']=model
    np.savez_compressed(out/'laws.npz',**arrays)
    (out/'heterogeneous.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2):run(ap.parse_args().output)
