"""Disjoint pair corrections within the nine coarse count cells, using an integer transport."""
import argparse
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.special import ndtri
from threadpoolctl import threadpool_limits

from iteration100_spatial_v1 import cdf_rows,realized,S,save,manifest
from iteration103_census_v1 import CENSUS,CountLaw,Optical
from iteration103_fit_v1 import DATA

TAIL=DATA/'iteration103_tail_v1'


def transport(q,p):
    # One integer distribution per coarse cell. Keep all common mass; among
    # the remaining moves, first preserve the row count, then the column count.
    shape=q.shape
    ids=np.arange(q.size).reshape(shape)
    bins=np.minimum(np.arange(shape[0]),2)[:,None]*3+np.minimum(np.arange(shape[1]),2)
    q=q.clip(0)
    q/=q.sum()
    p=p.clip(0)
    p/=p.sum()
    keep=np.full(q.size,S,np.uint64)
    pieces=[[] for _ in range(q.size)]
    for cat in range(9):
        mask=bins==cat
        if q[mask].sum()==0:continue
        assert p[mask].sum()>0
        origin=np.diff(cdf_rows(q[mask]),prepend=np.uint64(0)).astype(np.int64)
        target=np.diff(cdf_rows(p[mask]),prepend=np.uint64(0)).astype(np.int64)
        src={}
        dst={}
        for key,n,m in zip(ids[mask],origin,target):
            if n>m:
                src[int(key)]=int(n-m)
                keep[key]=(int(m)*S+int(n)//2)//int(n)
            if m>n:dst[int(key)]=int(m-n)
        def move(left,right):
            i=j=0
            while i<len(left) and j<len(right):
                a,b=left[i],right[j]
                mass=min(src[a],dst[b])
                if mass:pieces[a].append((b,mass))
                src[a]-=mass;dst[b]-=mass
                if not src[a]:i+=1
                if not dst[b]:j+=1
        for dim in (0,1):
            groups=defaultdict(lambda:[[],[]])
            for side,items in enumerate((src,dst)):
                for key,mass in items.items():
                    if mass:groups[divmod(key,shape[1])[dim]][side].append(key)
            for left,right in groups.values():move(left,right)
        move([k for k,v in src.items() if v],[k for k,v in dst.items() if v])
        assert not any(src.values()) and not any(dst.values())
    starts=np.zeros(q.size+1,np.int32)
    threshold=[];location=[]
    achieved=q.ravel()*(keep.astype(float)/S)
    changed=np.zeros(2)
    for i,choices in enumerate(pieces):
        starts[i]=len(threshold)
        if not choices:continue
        lab,masses=zip(*choices)
        cut=cdf_rows(np.array(masses,float))
        moved=q.ravel()[i]*(1-float(keep[i])/S)*realized(cut)
        for k,j in enumerate(lab):
            r,c=divmod(j,shape[1])
            location.append(r+(c<<16))
            achieved[j]+=moved[k]
            changed+=moved[k]*np.array([r!=i//shape[1],c!=i%shape[1]])
        threshold.extend(cut)
    starts[-1]=len(threshold)
    ae=achieved.reshape(shape)
    tv=abs(ae-p).sum()/2
    assert tv<5e-10 and abs(ae.sum()-1)<3e-13
    assert np.allclose([ae[bins==k].sum() for k in range(9)],[q[bins==k].sum() for k in range(9)],atol=1e-12,rtol=0)
    return keep,starts,np.array(threshold,np.uint64),np.array(location,np.uint32),ae,changed,float(tv)


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    s=np.load(CENSUS/'scan.npz')
    law=CountLaw(CENSUS/'chaos.dll')
    oracle=Optical()
    forced=[(36,107),(25,139)]
    reserved={x for pair in forced for x in pair}
    graph=nx.Graph()
    for a,b,v in zip(s['a'],s['b'],s['tv']):
        if a not in reserved and b not in reserved:graph.add_edge(int(a),int(b),weight=float(v))
    edges=forced+sorted(tuple(sorted(p)) for p in nx.max_weight_matching(graph,maxcardinality=True))
    keep=[];starts=[];cut=[];labels=[];begin=[];rows=[]
    cell=slot=0
    move=np.zeros(len(law.p))
    for a,b in edges:
        q,rem,_=law.pair(a,b,tol=1e-14)
        # Counts with zero integer marginal mass cannot occur in the native core.
        ref=oracle.pair(a,b)
        allowed=(law.p[a]>0)[:,None] & (law.p[b]>0)
        p=np.where(allowed,ref,0.)
        k,o,t,l,achieved,ch,tv=transport(q,p)
        keep.append(k);starts.append(o[:-1]+slot);cut.append(t);labels.append(l)
        begin.append(cell)
        cell+=k.size;slot+=len(t)
        move[[a,b]]=ch
        tv=float(abs(achieved-ref).sum()/2)
        rows.append(dict(groups=[a,b],raw_tv=float(abs(q-ref).sum()/2),new_tv=tv,series_remainder=rem,endpoint_change=ch.tolist()))
        assert tv<5e-10
    ptr=np.r_[np.concatenate(starts),slot].astype(np.int32)
    bounds=s['tv']+s['tail']+move[s['a']]+move[s['b']]
    for row in rows:
        a,b=row['groups']
        i=np.flatnonzero((s['a']==a)&(s['b']==b))[0]
        bounds[i]=row['new_tv']+2*row['series_remainder']+2e-12
    np.savez_compressed(out/'prepared.npz',edges=np.array(edges,np.int32),base=np.array(begin,np.int32),
        keep=np.concatenate(keep),starts=ptr,cuts=np.concatenate(cut),labels=np.concatenate(labels))
    np.savez_compressed(out/'bounds.npz',tv_upper=bounds,move=move)
    save(out,'summary.json',dict(pairs=rows,number=len(rows),full_count_pair_numerical_upper_max=float(bounds.max()),
         upper_percentiles=np.percentile(bounds,[50,90,99,100]).tolist(),coarse_invariant='all corrections preserve min(N,2) at both groups, for each record'))
    manifest(out)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):prepare(ap.parse_args().output.resolve())
