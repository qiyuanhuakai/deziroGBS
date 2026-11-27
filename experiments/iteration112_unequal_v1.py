"""Nonidentical, unequally populated detector cells with arbitrary shared-mode profiles."""
import argparse
import itertools
import json
import math
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from iteration111_memory_v1 import project, click_table
from iteration112_cases_v1 import optics
from iteration111_sampling_v1 import DATA


def triangle_table(x,k,b):
    # k and b are independent coherent bra/ket rows for a streamed dyad tile.
    d=len(x)//2
    v=np.ones((1<<d,len(k)),complex)
    ds=k+b
    for mask in range(1,1<<d):
        ids=np.flatnonzero(mask&(1<<np.arange(d)));ids=np.r_[ids,ids+d]
        q=np.eye(len(ids))+x[np.ix_(ids,ids)]
        a=ds[:,ids]
        v[mask]=np.exp(-np.sum((a@np.linalg.inv(q))*a,axis=1))/np.sqrt(np.linalg.det(q))
    law=v[::-1].copy()
    for j in range(d):
        u=law.reshape(-1,2,1<<j,len(k));u[:,1]-=u[:,0]
    return law


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    rows=[];arrays={}
    for buses in (1,2):
        local,xs,s,axes,bright,_=optics(buses)
        # Pick the smallest axis-specific grid meeting its allocated error.
        grids=[]
        for t in s:
            choices=[project(t,n) for n in range(2,16)]
            grids.append(next((p for p in choices if p[2]<1e-5/len(s)),choices[-1]))
        pos=np.array(list(itertools.product(*(p[0] for p in grids))))
        c=np.array([math.prod(v) for v in itertools.product(*(p[1] for p in grids))])
        bound=math.sqrt(-np.expm1(sum(np.log1p(-p[2]) for p in grids)))
        zz=np.split(1j*pos@axes.T,np.cumsum([len(a)//2 for a in local])[:-1],axis=1)
        for scale in (1024,8192):
            start=perf_counter();counts=np.array([2,3,5,7])*scale+np.array([1,2,7,3])
            base=[];ket=[];first=[];n=0;widths=[]
            for j,(x,z,raw) in enumerate(zip(xs,zz,local)):
                d=len(x)//2;widths.append(d)
                sel=np.r_[n:n+d,13+n:13+n+d];n+=d
                marginal=x+(bright[np.ix_(sel,sel)]-x)/counts[j]
                q=click_table(marginal,np.zeros((1,1,2*d),complex))[:,0,0].real
                obs=raw[np.r_[0:d,2*d:3*d]]
                z=z/np.sqrt(counts[j])
                ket.append(np.concatenate((z/2,z/(2j)),axis=1)@obs.T)
                base.append(q);first.append(np.zeros_like(q))
            # Simultaneous inversion and Hermitian conjugation share the same
            # real law contribution and squared L2 norm.
            size=len(c);ii,jj=np.tril_indices(size);use=ii+jj<=size-1
            ii,jj=ii[use],jj[use]
            higher=0.
            maximum=0.
            for off in range(0,len(ii),8192):
                i,j=ii[off:off+8192],jj[off:off+8192]
                gram=np.exp(-.5*((pos[i]-pos[j])**2).sum(1))
                mult=np.where(i==j,1,2)*np.where(i+j==size-1,1,2)
                w=c[i]*c[j]*gram*mult
                v=np.zeros(len(w))
                for g,(x,q,k) in enumerate(zip(xs,base,ket)):
                    diff=triangle_table(x,k[i],k[j].conj())-q[:,None]
                    first[g]+=diff.real@w
                    v+=counts[g]*(abs(diff)**2/q[:,None]).sum(0)
                higher+=np.dot(abs(w),v*np.exp(v/2)/math.sqrt(2))
                maximum=max(maximum,float(v.max()))
            one=math.sqrt(sum(n*np.dot(a/q,a) for n,a,q in zip(counts,first,base)))
            budget=min(1.,bound+.5*math.hypot(one,higher))
            rows.append(dict(buses=buses,nodes=[len(p[0]) for p in grids],branches=size,dyads=len(ii),
                       widths=widths,counts=counts.tolist(),ports=int(np.dot(widths,counts)),projection=bound,
                       first_order_l2=one,higher_order_l2_upper=higher,full_tv_budget=budget,
                       max_tilt=maximum,calculation_s=perf_counter()-start))
            for i,q in enumerate(base):arrays[f'b{buses}t{scale}g{i}']=q
            np.savez_compressed(out/'marginals.npz',**arrays)
            (out/'unequal.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
            print(buses,scale,size,budget,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2):run(ap.parse_args().output)
