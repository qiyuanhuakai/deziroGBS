"""Complete heterogeneous click laws and marked photon counts for one to four shared phases."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.linalg import block_diag
from threadpoolctl import threadpool_limits
from iteration111_heterogeneous_v1 import shared, realify, pair_counts, pair_reference, gaussian_click, oracle, ONE
from iteration111_memory_v1 import click_table, dyads
from iteration112_sparse_v1 import axis, project


def optics(buses,width=(3,4,2,4),return_transfer=False):
    if not 1<=buses<=6:raise ValueError("Supported phase count is one through six")
    rng=np.random.default_rng(111600001)
    local=[];xs=[]
    for d in width:
        u=np.linalg.qr(rng.standard_normal((d,d))+1j*rng.standard_normal((d,d)))[0]
        eta=rng.uniform(.64,.97,d)
        a,b=np.diag(np.sqrt(eta)),np.diag(np.sqrt(1-eta))
        squeeze=rng.uniform(.24,.78,d)
        s=realify(np.block([[a@u,b],[-b@u,a]]))*np.exp(np.r_[-squeeze,np.zeros(d),squeeze,np.zeros(d)])
        local.append(s)
        vis=s[np.r_[0:d,2*d:3*d]]
        xs.append((vis@vis.T-np.eye(2*d))/2)
    ports=sum(width)
    def directions():return rng.standard_normal((ports,2))+1j*rng.standard_normal((ports,2))
    vectors=directions()
    while vectors.shape[1]<buses:vectors=np.column_stack([vectors,directions()])
    vectors=vectors[:,:buses]
    vectors/=np.linalg.norm(vectors,axis=0)
    v,_=np.linalg.qr(vectors)
    small=np.eye(buses,dtype=complex)
    for direction,phase in zip(vectors.T,(1.1,-.82,.91,-1.24,1.36,-.72)):
        w=v.conj().T@direction
        small=(np.eye(buses)+np.expm1(1j*phase)*np.outer(w,w.conj()))@small
    direct=block_diag(*xs)
    offsets=np.cumsum([0]+[2*d for d in width[:-1]])
    physical=np.concatenate([np.arange(d)+offset+j*d for j in (0,1) for d,offset in zip(width,offsets)])
    base=direct[np.ix_(physical,physical)]
    squeeze,axes,_=shared(local,np.split(v,np.cumsum(width)[:-1]),small)
    coupling=np.eye(ports)+v@(small-np.eye(buses))@v.conj().T
    if return_transfer:
        top=[];env=[];rs=[]
        for d,loc in zip(width,local):
            col=loc[:2*d,:d]+1j*loc[2*d:,:d]
            norms=np.linalg.norm(col,axis=0)
            top.append(col[:d]/norms);env.append(col[d:]/norms);rs.extend(-np.log(norms))
        return np.vstack((coupling@block_diag(*top),block_diag(*env))),np.array(rs)
    f=realify(coupling)
    return local,xs,squeeze,axes,f@base@f.T,base


def local_tables(local,xs,pos,axes):
    z=1j*pos@axes.T
    widths=[len(a)//4 for a in local]
    splits=np.split(z,np.cumsum([2*d for d in widths])[:-1],axis=1)
    for a,x,ket,d in zip(local,xs,splits,widths):
        yield x,dyads(ket,a[np.r_[0:d,2*d:3*d]],1)


def complete(tables,weights):
    a,b,c,d=tables
    left=(b[:,None]*a[None]).reshape(len(a)*len(b),-1)
    right=(d[:,None]*c[None]).reshape(len(d)*len(c),-1)
    return ((right*weights.ravel())@left.T).real.ravel()


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    old=np.load(Path(__file__).resolve().parents[1]/'results/iteration111_heterogeneous_v1/laws.npz')
    rows=[];arrays={}
    for buses in (2,3,4):
        local,xs,s,axes,x,base=optics(buses)
        truth=gaussian_click(oracle(True),x[None],ONE)
        if buses==2:np.testing.assert_allclose(truth,old['truth2'],atol=1e-14)
        ids=[0,3,13,16]
        ref=pair_reference((np.eye(26)+x)[np.ix_(ids,ids)][None],10)
        ignore=gaussian_click(oracle(True),base[None],ONE)
        dictionaries=[axis(t) for t in s]
        arrays[f'truth{buses}']=truth;arrays[f'pair{buses}']=ref
        for budget in (9,25,49,81,129,193):
            t=perf_counter()
            p,c,g,info=project(s,dictionaries,budget)
            tables=[];pairs=[]
            for i,(q,d) in enumerate(local_tables(local,xs,p,axes)):
                tables.append(click_table(q,d))
                if i<2:
                    ii=[0,len(q)//2]
                    pairs.append(pair_counts(q[np.ix_(ii,ii)],d[...,ii],64))
            wei=np.outer(c,c)*g
            model=complete(tables,wei)
            pp=((pairs[1].reshape(11,-1)*wei.ravel())@pairs[0].reshape(11,-1).T).real.ravel()
            tv=float(abs(model-truth).sum()/2)
            assert abs(model.sum()-1)<1e-10 and model.min()>-1e-13 and tv<info['state_bound']+1e-10
            row=dict(buses=buses,squeezing=s.tolist(),**info,click_tv=tv,pair_tv=float(abs(pp-ref).sum()/2),
                     baseline_tv=float(abs(ignore-truth).sum()/2),evaluation_seconds=perf_counter()-t)
            print(buses,budget,tv,info['state_bound'],info['acceptance'],flush=True)
            rows.append(row);arrays[f'law{buses}_{budget}']=model
    np.savez_compressed(out/'laws.npz',**arrays)
    (out/'cases.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=3):run(ap.parse_args().output)
