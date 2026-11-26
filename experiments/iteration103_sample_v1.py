"""Two-field optical sampling, optional root bridges, and blinded whole-device benchmarks."""
import argparse
import ctypes as ct
import json
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
from scipy.special import ndtri
from threadpoolctl import threadpool_limits

from iteration100_spatial_v1 import ROOT,ORDER,HIST,Native,realify,split_joint,cdf_rows,realized,bind,I,U,D,J,W,B,S,SHORT,save,manifest,H
from iteration102_sample_v1 import Sampler as Previous, P, BLOCK
from iteration102_forest_v1 import FIT as OLD,RUN as ENGINE,BASE,DATA,VAC,Tables
from iteration103_fit_v1 import FIT

RUN=DATA/'iteration103_sample_v1'


def bridge_tables(tables):
    roots=[int(g) for g in tables.order if tables.parent[g]<0]
    scan=np.load(VAC/'scan.npz')
    pilot=np.load(DATA/'iteration102_release_v1/hold_0.npz')['vacuum_counts']/1048576
    graph=nx.Graph()
    graph.add_nodes_from(roots)
    forced=[(36,107),(25,139)]
    used=set(i for pair in forced for i in pair)
    for i,j,d in zip(scan['a'],scan['b'],abs(pilot-scan['optical'])):
        if i in roots and j in roots and i not in used and j not in used:
            graph.add_edge(int(i),int(j),weight=float(d))
    chosen=forced+sorted(tuple(sorted(pair)) for pair in nx.max_weight_matching(graph,maxcardinality=True))
    z=np.load(ORDER)
    v=z['v'][np.argsort(z['order'])]
    mats=[realify(v[8*g:8*g+8]) for g in range(len(tables.order))]
    grams=[m.T@m for m in mats]
    native=Native(BASE/'spatial.dll')
    cuts=[]
    for i,j in chosen:
        law=split_joint(native,grams[i],grams[j],z['x'],int(tables.caps[i]),int(tables.caps[j])).clip(0)
        law[law.sum(1)==0]=tables.marg[j]
        cuts.append(cdf_rows(law).ravel())
    return chosen,cuts


def residual_factor(corr,order,parent):
    m=len(order)
    transform=np.eye(2*m)
    for g in order:
        chain=[]
        p=parent[g]
        while p>=0:
            chain.extend([2*p,2*p+1])
            p=parent[p]
        local=[2*g,2*g+1]
        if chain:
            block=corr[np.ix_(chain,local)]
            transform[np.ix_(local,chain)]=-np.linalg.solve(corr[np.ix_(chain,chain)],block).T
    raw=transform@corr@transform.T
    diag=raw.reshape(m,2,m,2)[np.arange(m),:,np.arange(m),:]
    scale=np.linalg.inv(np.linalg.cholesky(diag))
    v=np.einsum('iaj,ijhk,hbk->iahb',scale,raw.reshape(m,2,m,2),scale,optimize=True).reshape(2*m,2*m)
    idx=np.c_[2*order,2*order+1].ravel()
    maximum=0.
    for g in order:
        p=parent[g]
        while p>=0:
            maximum=max(maximum,float(abs(v[2*g:2*g+2,2*p:2*p+2]).max()))
            p=parent[p]
    assert maximum<2e-14 and abs(v[2*np.arange(m),2*np.arange(m)+1]).max()<2e-14
    return np.ascontiguousarray(np.linalg.cholesky(v[np.ix_(idx,idx)]))


def transform_rows(cdf):
    # Integer subtraction preserves small positive-count masses at high vacuum probability.
    first=cdf[:,:1]
    pos=(cdf-first).astype(float)
    np.divide(pos,S-first,out=pos,where=first<S)
    ans=ndtri(pos)
    ans[:,0]=ndtri(first[:,0].astype(float)/S)
    return ans


class Sampler(Previous):
    def __init__(self,dll,bridges,linked):
        super().__init__(ENGINE/'forest.dll',depth=4 if linked else 2)
        t=Tables(OLD/'prepared.npz')
        corr=np.load(FIT/'fit.npz')['corr']
        self.newlib=ct.CDLL(str(dll))
        self.draw=bind(self.newlib,'continuation103_draw',[I,I,I,I,U,U,J,J,J,J,I,W,J,J,D,D,P,J,W,D,D,W,B,B,SHORT,SHORT])
        if linked:
            # Preserve the earlier nodes' explicit two-, three-, and four-site
            # conditional tables. Only former roots gain new count conditionals.
            extra=np.load(bridges)
            self.parent=self.parent.copy()
            self.offset=self.offset.copy()
            self.cdf=np.r_[self.cdf,extra['cdf']]
            base=len(self.cdf)-len(extra['cdf'])
            for (a,b),off in zip(extra['edges'],extra['offset']):
                assert self.parent[b]<0
                self.parent[b]=a
                self.offset[b]=base+off
            graph=nx.DiGraph()
            graph.add_nodes_from(self.order)
            graph.add_edges_from((p,g) for g,p in enumerate(self.parent) if p>=0)
            self.order=np.array(list(nx.topological_sort(graph)),np.int32)
        else:
            self.order=np.arange(len(t.order),dtype=np.int32)
            self.parent=np.full(len(t.order),-1,np.int32)
            rows=[cdf_rows(t.marg[g]) for g in self.order]
            self.offset=np.cumsum([0]+[len(row) for row in rows[:-1]]).astype(np.int32)
            self.cdf=np.concatenate(rows)
        self.factor=residual_factor(corr,self.order,self.parent)
        # Only active conditional tables need storage, even when the archived
        # parent model contains superseded pair and triple tables.
        pieces=[]
        offset=self.offset.copy()
        pos=0
        for g in self.order:
            n=1
            for ancestor in (self.parent[g],self.second[g]):
                if ancestor>=0:n*=int(self.caps[ancestor])+1
            if self.third[g]>=0:n*=self.classes.shape[1]
            width=self.caps[g]+1
            row=self.cdf[self.offset[g]:self.offset[g]+n*width].reshape(n,width)
            pieces.append(transform_rows(row).ravel())
            offset[g]=pos
            pos+=row.size
        self.offset=offset
        self.thresholds=np.concatenate(pieces)
        del self.cdf

    def sample(self,n,route,workers,seed,offset=0):
        groups=len(self.order)
        x=np.empty((n,8*groups),np.uint16)
        c=np.empty((n,groups),np.uint16)
        rc=self.draw(groups,n,int(route),workers,seed,offset,self.order,self.parent,self.second,self.third,
            self.classes.shape[1],self.classes,self.caps,self.offset,self.thresholds,self.factor,self.gemm,
            self.starts,self.masks,self.prior,self.weights,self.shortcut_cut,self.shortcut_mask,self.shortcut_fallback,x,c)
        if rc:raise RuntimeError(f'Continuation draw failed: {rc}')
        return x,c


def bench(out):
    out.mkdir(parents=True,exist_ok=False)
    src=ROOT/'native/continuation_draw103_v1.cpp'
    dll=out/'draw.dll'
    subprocess.run([shutil.which('g++'),'-std=c++20','-O3','-shared','-fopenmp','-ffp-contract=off',str(src),'-o',str(dll)],check=True)
    t=Tables(OLD/'prepared.npz')
    edges,rows=bridge_tables(t)
    offsets=np.cumsum([0]+[len(r) for r in rows[:-1]]).astype(np.int32)
    bridges=out/'bridges.npz'
    np.savez_compressed(bridges,edges=np.array(edges,np.int32),offset=offsets,cdf=np.concatenate(rows))
    target=np.load(FIT/'fit.npz')
    a,b=target['a'],target['b']
    hw=np.load(HIST)['total_click_histogram']
    photon=np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf']
    answers=[]
    for linked in (False,True):
        start=perf_counter()
        model=Sampler(dll,bridges,linked)
        prep=perf_counter()-start
        x,c=model.sample(2201,False,1,103010000+linked)
        y,d=model.sample(2138,True,8,103010000+linked,63)
        assert np.array_equal(x[63:],y>0) and np.array_equal(c[63:],d)
        assert np.array_equal(d,y.reshape(len(y),-1,8).sum(2))
        for route in (False,True):
            times=[]
            for i in range(8):
                start=perf_counter()
                model.sample(8192,route,16,103011000+2*linked+route,i*8192)
                if i:times.append((perf_counter()-start)*1e6/8192)
            n=262144
            hc=np.zeros(len(hw),np.int64)
            hp=np.zeros(4096,np.int64)
            cross=np.zeros((288,288))
            for off in range(0,n,BLOCK):
                x,c=model.sample(BLOCK,route,16,103012000+2*linked+route,off)
                if route:assert np.array_equal(x.reshape(BLOCK,-1,8).sum(2),c)
                hc+=np.bincount((x>0).sum(1),minlength=len(hc))
                hp+=np.bincount(c.sum(1).astype(int),minlength=len(hp))
                marks=np.concatenate([c>=1,c>=2],axis=1).astype(float)
                cross+=marks.T@marks
            count=np.column_stack([cross[144*i+a,144*j+b] for i in range(2) for j in range(2)])
            gap=count/n-target['target']
            k=np.unravel_index(abs(gap).argmax(),gap.shape)
            np.savez_compressed(out/f'pilot_{int(linked)}_{int(route)}.npz',click=hc,photon=hp,events=count.astype(np.int64))
            answers.append(dict(linked=linked,pnr=route,n=n,load_s=prep,us=times,median_us=float(np.median(times)),
                hardware_ks=float(abs(hc.cumsum()/n-hw.cumsum()/hw.sum()).max()),
                photon_ks=float(abs(hp.cumsum()/n-np.pad(photon,(0,len(hp)-len(photon))).cumsum()).max()),
                worst_pilot=[int(a[k[0]]),int(b[k[0]]),int(k[1])],gap=float(gap[k])))
    save(out,'summary.json',dict(rows=answers,root_bridges=len(edges),hashes={p.relative_to(ROOT).as_posix():H(p) for p in (src,Path(__file__),FIT/'fit.npz',HIST)}))
    manifest(out)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest='action',required=True)
    p=sub.add_parser('benchmark')
    p.add_argument('output',type=Path)
    p=sub.add_parser('draw')
    p.add_argument('output',type=Path)
    p.add_argument('--root-bridges',action='store_true')
    p.add_argument('--pnr',action='store_true')
    p.add_argument('--shots',type=int,default=2048)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--seed',type=int,required=True)
    args=ap.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.action=='benchmark':bench(args.output.resolve())
        else:
            m=Sampler(RUN/'draw.dll',RUN/'bridges.npz',args.root_bridges)
            x,c=m.sample(args.shots,args.pnr,args.workers,args.seed)
            with args.output.open('xb') as f:np.savez_compressed(f,records=x,spatial_photons=c)
