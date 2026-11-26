"""Optical conditional count tables on overlapping edges and Gaussian ancestor innovations."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
from scipy.special import ndtri
from threadpoolctl import threadpool_limits

from iteration100_spatial_v1 import ROOT, ORDER, Native, Tree, realify, split_joint, cdf_rows, realized, save, H, manifest
from iteration100_projection_v1 import Projector
from iteration101_copula_v1 import BASE, DATA

FIT=DATA/'iteration102_forest_v1'
RUN=DATA/'iteration102_sample_v1'
VAC=DATA/'iteration101_vacuum_v1'
COP=DATA/'iteration101_fit_v1'


def forest(size):
    scan=np.load(VAC/'scan.npz')
    error=abs(scan['model']-scan['optical'])
    weight={(int(a),int(b)):float(e) for a,b,e in zip(scan['a'],scan['b'],error)}
    rows=json.loads((COP/'summary.json').read_text())['rows']
    required=[r['groups'] for r in rows]+[json.loads((VAC/'summary.json').read_text())['worst_pair']]
    graph=nx.Graph()
    graph.add_nodes_from(range(size))
    for a,b in required:
        graph.add_edge(a,b,weight=weight[tuple(sorted((a,b)))])
    assert nx.is_forest(graph)
    for pair,w in sorted(weight.items(),key=lambda item:(-item[1],item[0])):
        if w<=.01:
            break
        if not nx.has_path(graph,*pair):
            graph.add_edge(*pair,weight=w)
    order=[]
    parent=np.full(size,-1,dtype=np.int32)
    for subset in nx.connected_components(graph):
        sub=graph.subgraph(subset)
        root=min(nx.center(sub),key=lambda i:(-sub.degree(i,weight='weight'),i))
        order.append(root)
        for a,b in nx.bfs_edges(sub,root,sort_neighbors=sorted):
            parent[b]=a
            order.append(b)
    return graph,np.array(order,dtype=np.int32),parent


def innovations(corr,order,parent):
    # The innovation of each child is orthogonal to every latent variable
    # used in constructing the parent's observed count.
    transform=np.eye(len(order))
    ancestors=[]
    maxdepth=0
    for g in order:
        chain=[]
        p=parent[g]
        while p>=0:
            chain.append(p)
            p=parent[p]
        maxdepth=max(maxdepth,len(chain))
        if chain:
            transform[g,chain]=-np.linalg.solve(corr[np.ix_(chain,chain)],corr[chain,g])
            ancestors.extend((g,i) for i in chain)
    cov=transform@corr@transform.T
    sd=np.sqrt(cov.diagonal())
    cov=cov/sd[:,None]/sd
    orth=max(abs(cov[g,i]) for g,i in ancestors)
    lower=np.linalg.cholesky(cov[np.ix_(order,order)])
    assert orth<2e-14
    return lower,orth,maxdepth


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    start=perf_counter()
    with np.load(BASE/'prepared.npz') as f:
        model=Tree(None,dict(f))
    projection=Projector(model)
    graph,order,parent=forest(model.leaves)
    corr=np.load(COP/'fit.npz')['corr']
    factor,orth,depth=innovations(corr,order,parent)
    z=np.load(ORDER)
    v=z['v'][np.argsort(z['order'])]
    t=np.array([realify(v[8*g:8*g+8]) for g in range(model.leaves)])
    gram=t.transpose(0,2,1)@t
    lib=Native(BASE/'spatial.dll')
    tables=[]
    offset=np.zeros(len(order),np.int32)
    pos=0
    marg={}
    max_edge=max_marg=neg=0.
    for g in order:
        p=parent[g]
        if p<0:
            table=cdf_rows(projection.totals[g][None,:])
            marg[g]=realized(table)[0]
        else:
            joint=split_joint(lib,gram[p],gram[g],z['x'],int(model.caps[p]),int(model.caps[g]))
            neg=max(neg,float(-joint[joint<0].sum()))
            clean=joint.clip(0)
            empty=clean.sum(1)==0
            clean[empty]=projection.totals[g]
            table=cdf_rows(clean)
            actual=realized(table)*marg[p][:,None]
            marg[g]=actual.sum(0)
            max_edge=max(max_edge,float(abs(actual-joint).sum()/2))
        max_marg=max(max_marg,float(abs(marg[g]-projection.totals[g]).sum()/2))
        offset[g]=pos
        tables.append(table.ravel())
        pos+=table.size
    cdf=np.concatenate(tables)
    assert max_edge<1e-11 and max_marg<1e-11
    np.savez_compressed(out/'prepared.npz',order=order,parent=parent,offset=offset,cdf=cdf,
                        factor=factor,caps=model.caps[:len(order)])
    save(out,'summary.json',dict(groups=model.leaves,edges=graph.number_of_edges(),components=nx.number_connected_components(graph),
        depth=depth,max_ancestor_correlation=orth,edge_count_tv_roundoff=max_edge,marginal_tv_roundoff=max_marg,
        negative_mass_max=neg,cdf_words=len(cdf),prepare_s=perf_counter()-start,
        required='five archived diagnostic pairs; then all prior absolute pair-vacuum errors above 0.01 in descending order, skipping cycles',
        inputs={p.relative_to(ROOT).as_posix():H(p) for p in [ORDER,BASE/'prepared.npz',COP/'fit.npz',VAC/'scan.npz']}))
    manifest(out)
    print((out/'summary.json').read_text(),flush=True)


class Tables:
    def __init__(self,path):
        with np.load(path) as f:
            self.__dict__.update(dict(f))
        self.graph=nx.Graph()
        self.graph.add_nodes_from(self.order)
        self.graph.add_edges_from((g,p) for g,p in enumerate(self.parent) if p>=0)
        self.marg={}
        self.transition={}
        for g in self.order:
            p=self.parent[g]
            a=self.row(g)
            if p<0:
                self.marg[g]=a[0]
            else:
                self.transition[p,g]=a
                self.marg[g]=self.marg[p]@a
                joint=self.marg[p][:,None]*a
                self.transition[g,p]=np.divide(joint.T,self.marg[g][:,None],out=np.zeros_like(joint.T),where=self.marg[g][:,None]>0)

    def row(self,g):
        p=self.parent[g]
        nr=self.caps[p]+1 if p>=0 else 1
        nc=self.caps[g]+1
        return realized(self.cdf[self.offset[g]:self.offset[g]+nr*nc].reshape(nr,nc))

    def pair(self,a,b):
        if not nx.has_path(self.graph,a,b):
            return self.marg[a][:,None]*self.marg[b]
        path=nx.shortest_path(self.graph,a,b)
        law=np.diag(self.marg[a])
        for i,j in zip(path,path[1:]):
            law=law@self.transition[i,j]
        return law


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):
        prepare(ap.parse_args().output.resolve())
