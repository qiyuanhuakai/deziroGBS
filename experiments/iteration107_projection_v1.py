"""Exact cross-pair response from four latent spatial photon totals and a multidimensional Gaussian PGF."""
import argparse
import itertools
import math
from pathlib import Path
import numpy as np
from scipy.fft import fftn,ifftn,next_fast_len
from scipy.stats import binom
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import occupancy_prior,save,manifest
from iteration100_baselines_v1 import pair_reference
from iteration100_projection_v1 import physical
from iteration105_jiuzhang_v1 import inputs
from iteration106_field_v1 import actual_input,factors
from iteration106_packed_v1 import compress,OUT as PREV
from iteration106_sample_v1 import read,DATA
from iteration107_cliques_v1 import tail_caps
from iteration107_tables_v1 import OUT as TABLES,LIMIT


def coefficients(q):
    d=len(q)//2
    h=np.eye(2*d)-np.linalg.inv(q);poly=np.zeros((3,)*d)
    for mask in range(1<<(2*d)):
        ix=np.array([j for j in range(2*d) if mask>>j&1],int)
        e=tuple(np.bincount(ix%d,minlength=d))
        poly[e]+=(-1.)**len(ix)*(np.linalg.det(h[np.ix_(ix,ix)]) if len(ix) else 1.)
    return poly,float(np.linalg.slogdet(q)[1]),float(np.linalg.norm(h,2))


def joint(x,profile,eps):
    d=x.shape[1]//2
    e=np.array([np.concatenate([np.linalg.eigvalsh(v[np.ix_([i,i+d],[i,i+d])]) for v in x]) for i in range(d)])
    caps=tail_caps(e*max(profile['values']),eps/d)
    # Bound the log determinant phase change between neighboring Fourier
    # nodes. Only two equal variables move along any individual coordinate.
    K=max(np.linalg.eigvalsh(v).max()*max(profile['values']) for v in x)
    caps=[next_fast_len(max(8,int(s),int(np.ceil(4*K+4)))) for s in caps]
    prob=np.zeros(caps)
    for w,g in zip(profile['weights'],profile['values']):
        logf=0.
        for v in x:
            poly,logdet,norm=coefficients(np.eye(2*d)+g*v)
            z=fftn(poly,s=caps,workers=4)
            phase=np.angle(z)
            for axis in reversed(range(d)):phase=np.unwrap(phase,axis=axis)
            logf=logf-.5*(logdet+np.log(abs(z))+1j*phase)
        grid=np.exp(logf)
        assert abs(grid[(0,)*d]-1)<1e-9
        p=ifftn(grid,workers=4)
        assert np.max(abs(p.imag))<1e-12
        prob+=w*p.real
    neg=float(-prob[prob<0].sum())
    assert neg<2e-9 and abs(prob.sum()-1)<1e-9,(neg,prob.sum())
    return prob,caps,neg


def responses(z,g,j,caps,route,port=None):
    a,b=z['pairs'][g];cap1,cap2=caps
    ix=0 if a==j else 1
    def masks(site,c):
        return occupancy_prior(z['weights'][site],c-1)
    def thinning(site,c):
        n=np.arange(c)[:,None];p=z['weights'][site,port]
        # For a physical PNR validate bins 0..11 and 12+ explicitly.
        out=binom.pmf(np.arange(12)[None,:],n,p);return np.c_[out,(1-out.sum(1)).clip(0)]
    first=masks(a,cap1);second=masks(b,cap2)
    def ordered(mat):return mat if ix==0 else mat.T
    selected=[first,second][ix]
    other=[second,first][ix]
    if route=='click':
        move=ordered(z['click_move'][g].reshape(256,256).T.astype(float))
        excess=np.diff(z['click_cdf'][g],prepend=np.uint64(0)).reshape(256,256).T/2**52
        marginal=ordered(excess).sum(1)
        keep=1-move@other.T
        prob=selected[:,None,:]*keep.T[None,:,:]
        prob+=(selected@move@other.T)[:,:,None]*marginal
    else:
        mc=z['pnr_move'][g].reshape(LIMIT,LIMIT).T
        excess=np.diff(z['pnr_cdf'][g],prepend=np.uint64(0)).reshape(LIMIT,LIMIT).T/2**52
        marginal=ordered(excess).sum(1)
        initial=selected if port is None else thinning(j,len(selected))
        moved=masks(j,LIMIT) if port is None else thinning(j,LIMIT)
        replace=marginal@moved
        keep=np.zeros((cap1,cap2))
        keep[:min(cap1,LIMIT),:min(cap2,LIMIT)]=mc[:min(cap1,LIMIT),:min(cap2,LIMIT)]
        rate=ordered(keep)[:,:,None]
        prob=(1-rate)*initial[:,None,:]+rate*replace
    return prob if ix==0 else prob.transpose(1,0,2)


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    t,r=actual_input();config=read(PREV/'summary.json');rr=np.array(config['squeezing']);profile=config['profiles']['1']
    modes,w,_=compress(t,(np.sinh(r)**2).sum(0),1);modes*=np.sqrt((abs(w[:,0])**2).reshape(-1,8).sum(1))[:,None]
    tables=dict(np.load(TABLES/'tables.npz'));pairs=tables['pairs'];loc={int(j):i for i,p in enumerate(pairs) for j in p}
    counts=np.load(TABLES/'counts.npz')['target_counts'];results=[]
    pnr_true=pair_reference(physical(inputs()[0],[289,861]),12)
    for panel,(i,j) in enumerate([(0,1),(36,107),(34,35)]):
        a,b=loc[i],loc[j];ports=np.r_[np.arange(8)+8*i,np.arange(8)+8*j]
        ref=np.load(PREV/f'panel_{[0,1,5][panel]}.npz')
        assert np.array_equal(ports,ref['ports'])
        if a==b:
            assert (pairs[a]==[i,j]).all()
            dist=counts[a].reshape(LIMIT,LIMIT)
            rate=tables['click_move'][a].astype(float);residual=np.diff(tables['click_cdf'][a],prepend=np.uint64(0))/2**52
            realized=ref['1']*(1-rate)+(ref['1']@rate)*residual
            probs=[realized,(occupancy_prior(tables['weights'][j],LIMIT-1).T@dist@occupancy_prior(tables['weights'][i],LIMIT-1)).ravel()]
            meta=dict(input_groups=2,click_reference='Full physical optical word used in the probability table',fft_tail=0.)
        else:
            groups=[*pairs[a],*pairs[b]]
            f=factors(modes[groups],rr);x=np.stack([2*v.T@v for v in f])
            # The axes correspond to physical counts, in exactly the same
            # axis order as the two response tensors.
            raw,caps,neg=joint(x,profile,2e-10)
            mat=raw.reshape(caps[0]*caps[1],-1);probs=[]
            for route in ('click','pnr'):
                left=responses(tables,a,i,caps[:2],route)
                right=responses(tables,b,j,caps[2:],route)
                prob=right.reshape(-1,256).T@(mat.T@left.reshape(-1,256))
                probs.append(prob.ravel())
            pri=occupancy_prior(tables['weights'][i],caps[groups.index(i)]-1);prj=occupancy_prior(tables['weights'][j],caps[groups.index(j)]-1)
            marginal=raw.sum(tuple(k for k in range(4) if groups[k] not in (i,j)))
            check=(prj.T@marginal.T@pri).ravel()
            delta=float(abs(check-ref['1']).sum()/2)
            assert delta<1e-8,delta
            meta=dict(input_groups=4,fft_shape=caps,fft_tail=2e-10,negative_mass=neg,independent_base_tv=delta)
            if panel==1:
                left=responses(tables,a,i,caps[:2],'pnr',1)
                right=responses(tables,b,j,caps[2:],'pnr',5)
                obs=(right.reshape(-1,13).T@(mat.T@left.reshape(-1,13))).ravel()
                meta['two_detector_pnr_tv']=float(abs(obs-pnr_true).sum()/2)
                meta['two_detector_pnr_tail']=float(pnr_true.reshape(13,13)[-1].sum()+pnr_true.reshape(13,13)[:-1,-1].sum())
                np.savez_compressed(out/'pnr.npz',target=pnr_true,model=obs)
        tv=[float(abs(v-ref['target']).sum()/2) for v in probs]
        results.append(dict(groups=[i,j],matched=a==b,prior_tv=float(abs(ref['1']-ref['target']).sum()/2),click_tv=tv[0],pnr_threshold_tv=tv[1],**meta))
        np.savez_compressed(out/f'panel_{panel}.npz',target=ref['target'],click=probs[0],pnr=probs[1])
        print(results[-1],flush=True)
    save(out,'summary.json',dict(rows=results))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(p.parse_args().output.resolve())
