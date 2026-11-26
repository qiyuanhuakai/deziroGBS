"""Complete optical 16-bit laws for all conditioned edges and loops, and fresh residual panels."""
import argparse
import ctypes as ct
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from iteration100_projection_v1 import Projector,physical
from iteration100_spatial_v1 import I,D,bind,Native,Tree,ORDER,split_joint,realify,realized,save,manifest
from iteration101_copula_v1 import features
from iteration102_forest_v1 import FIT,BASE,DATA,RUN,COP,VAC,Tables
from iteration102_triads_v1 import TRI
from iteration102_four_v1 import FOUR

PROJ=DATA/'iteration102_projection_v1'


class Projection:
    def __init__(self):
        self.t=Tables(FIT/'prepared.npz')
        self.tri=dict(np.load(TRI/'prepared.npz'))
        self.full=Projector(Tree(None,dict(np.load(BASE/'prepared.npz'))))
        self.corr=np.load(COP/'fit.npz')['corr']
        self.four={}
        for item in json.loads((FOUR/'summary.json').read_text())['rows']:
            a,_,_,g=item['groups']
            self.four[a,g]=np.load(FOUR/f'endpoint_{a}_{g}.npz')['counts']

    def path(self,a,g):
        path=[g]
        while path[-1]!=a and self.t.parent[path[-1]]>=0:
            path.append(int(self.t.parent[path[-1]]))
        return path[::-1] if path[-1]==a else []

    def conditional(self,g):
        p=self.t.parent[g]
        a=self.t.parent[p]
        shape=tuple(self.t.caps[[a,p,g]]+1)
        off=self.tri['offset'][g]
        return realized(self.tri['cdf'][off:off+int(np.prod(shape))].reshape(shape))

    def pair(self,a,b,depth):
        t=self.t
        path=self.path(a,b) or self.path(b,a)
        if len(path)>1:
            if depth==4 and (path[0],path[-1]) in self.four:
                law=self.four[path[0],path[-1]]
            elif depth==2 or len(path)==2:
                law=t.pair(path[0],path[-1])
            else:
                if depth==4 and len(path)>4:return None
                first=t.pair(path[0],path[1])
                state=first[:,:,None]*self.conditional(path[2])
                if len(path)>3:
                    for i in path[3:]:
                        state=np.einsum('aij,ijk->ajk',state,self.conditional(i),optimize=True)
                law=state.sum(1)
            return law if path[0]==a else law.T
        if t.parent[a]<0 and t.parent[b]<0:
            left=features(t.marg[a])[1]
            right=features(t.marg[b])[1]
            return (left*self.corr[a,b]**np.arange(left.shape[1]))@right.T
        return None

    def bits(self,a,b,joint):
        return (self.full.mask[b].T@joint.T@self.full.mask[a]).ravel()


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    proj=Projection()
    t=proj.t
    spec=np.load(ORDER)
    v=spec['v'][np.argsort(spec['order'])]
    m=np.array([realify(v[g*8:g*8+8]) for g in range(len(t.parent))])
    grams=m.transpose(0,2,1)@m
    lib=Native(BASE/'spatial.dll')
    oracle=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    edge=[(int(p),int(g),'edge') for g,p in enumerate(t.parent) if p>=0]
    triangles=[(r['groups'][0],r['groups'][-1],'triad') for r in json.loads((TRI/'summary.json').read_text())['triads']]
    fours=[(a,g,'four') for a,g in proj.four]
    s=np.load(VAC/'scan.npz')
    selected=set()
    for depth in (2,3,4):
        # Both routes use disjoint seeds. The optical references predate all pilots.
        total=sum(np.load(RUN/f'hist_{depth}_1_{j}.npz')['vacuum_counts'] for j in (0,1))
        i=int(np.argmax(abs(total/524288-s['optical'])))
        selected.add((int(s['a'][i]),int(s['b'][i])))
    for j in (0,1):
        single=np.load(RUN/f'hist_4_1_{j}.npz')['vacuum_counts']/262144
        i=int(np.argmax(abs(single-s['optical'])))
        selected.add((int(s['a'][i]),int(s['b'][i])))
    required=selected|{(34,35),(35,69),(70,73),(74,80),(125,131),(11,44),(57,63),(34,69)}
    seen={tuple(sorted((a,b))) for a,b,_ in edge+triangles+fours}
    cases=edge+triangles+fours+[(a,b,'pilot') for a,b in sorted(selected-seen)]
    def run(case):
        a,b,kind=case
        a,b=sorted((a,b))
        jref=split_joint(lib,grams[a],grams[b],spec['x'],int(t.caps[a]),int(t.caps[b]))
        truth=np.empty(65536)
        ports=np.r_[8*a+np.arange(8),8*b+np.arange(8)]
        assert oracle(16,2,physical(spec,ports),np.zeros((2,32)),truth,np.empty(2))==0
        assert abs(jref[0,0]-truth[0])<2e-12
        row=dict(groups=[a,b],kind=kind,vacuum_optical=float(truth[0]),depths={})
        outputs={}
        for depth in (2,3,4):
            joint=proj.pair(a,b,depth)
            if joint is None:continue
            bits=proj.bits(a,b,joint)
            assert abs(bits.sum()-1)<1e-10
            row['depths'][str(depth)]=dict(count_tv=float(abs(joint-jref).sum()/2),click_tv=float(abs(bits-truth).sum()/2),
                                           vacuum_error=float(bits[0]-truth[0]))
            outputs['bits'+str(depth)]=bits
            outputs['counts'+str(depth)]=joint
        if (a,b) in required:
            np.savez_compressed(out/f'pair_{a}_{b}.npz',truth=truth,reference=jref,**outputs)
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(run,cases))
    for kind in ('edge','triad'):
        sample=[r['depths']['4'] for r in rows if r['kind']==kind]
        print(kind,len(sample),max(i['count_tv'] for i in sample),max(i['click_tv'] for i in sample),flush=True)
    save(out,'summary.json',dict(rows=rows,selected_from_pilot=[list(s) for s in sorted(selected)],
         complete_words=65536,analytic='Ancestor paths and distinct roots. A missing entry requires sampling of cross-branch innovations.'))
    manifest(out)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):
        main(ap.parse_args().output.resolve())
