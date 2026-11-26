"""Complete optical click projections of the raw, matched-tail and bridged models."""
import argparse
import ctypes as ct
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from iteration100_projection_v1 import physical
from iteration100_spatial_v1 import I,D,S,ORDER,bind,save,manifest,realized
from iteration102_projection_v1 import Projection,PROJ
from iteration103_census_v1 import CountLaw,Optical,CENSUS
from iteration103_sample_v1 import RUN,DATA
from iteration103_tail_v1 import TAIL


def main(out):
    out.mkdir(exist_ok=False,parents=True)
    proj=Projection()
    raw=CountLaw(CENSUS/'chaos.dll')
    oracle=Optical()
    spec=dict(np.load(ORDER))
    f=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    bridge=dict(np.load(RUN/'bridges.npz'))
    patch=dict(np.load(TAIL/'prepared.npz'))
    shifts=np.load(TAIL/'bounds.npz')['move']
    tails={tuple(map(int,e)):i for i,e in enumerate(patch['edges'])}
    bridges={tuple(map(int,e)):int(i) for e,i in zip(bridge['edges'],bridge['offset'])}
    previous={tuple(r['groups']):r['kind'] for r in json.loads((PROJ/'summary.json').read_text())['rows'] if r['kind']!='pilot'}
    selected={(0,1),(36,107),(25,139),(34,35),(64,73),(22,55),(70,73),(35,69)}
    worst={tuple(r['groups']) for r in json.loads((CENSUS/'summary.json').read_text())['largest']}
    pairs=sorted(set(tails)|set(bridges)|set(previous)|selected|worst)
    def evaluate(pair):
        a,b=pair
        ref=oracle.pair(a,b)
        joint,rem,_=raw.pair(a,b,tol=1e-14)
        ports=np.r_[8*a+np.arange(8),8*b+np.arange(8)]
        truth=np.empty(65536)
        assert f(16,2,physical(spec,ports),np.zeros((2,32)),truth,np.empty(2))==0
        def metrics(mat):
            bits=proj.bits(a,b,mat)
            return dict(count_tv=float(abs(mat-ref).sum()/2),click_tv=float(abs(bits-truth).sum()/2))
        row=dict(groups=[a,b],raw=metrics(joint),hermite_remainder=rem)
        row['patched_click_interval']=[max(0.,row['raw']['click_tv']-shifts[a]-shifts[b]-rem),min(1.,row['raw']['click_tv']+shifts[a]+shifts[b]+rem)]
        data=dict(reference=ref,truth=truth,raw=joint,ports=ports)
        if pair in tails:
            # Push through the actual prepared integer transport, including the two draws.
            base=patch['base'][tails[pair]]
            q=joint.clip(0).ravel()
            q/=q.sum()
            frac=patch['keep'][base:base+q.size].astype(float)/S
            kept=q*frac
            for i in np.flatnonzero(frac<1):
                lo,hi=patch['starts'][base+i:base+i+2]
                lab=patch['labels'][lo:hi]
                dest=(lab%65536)*joint.shape[1]+(lab>>16)
                np.add.at(kept,dest,q[i]*(1-frac[i])*realized(patch['cuts'][lo:hi]))
            m=kept.reshape(joint.shape)
            row['patched']=metrics(m)
            row['patched_click_interval']=[max(0.,row['patched']['click_tv']-rem-2e-12),row['patched']['click_tv']+rem+2e-12]
            data['patched']=m
        if pair in bridges:
            start=bridges[pair]
            m=raw.p[a][:,None]*realized(bridge['cdf'][start:start+joint.size].reshape(joint.shape))
            row['bridge']=metrics(m)
        if pair in previous:
            row['previous_kind']=previous[pair]
            row['retained']=metrics(proj.pair(a,b,4))
        if pair in selected:np.savez_compressed(out/f'pair_{a}_{b}.npz',**data)
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(evaluate,pairs))
    save(out,'summary.json',dict(rows=rows,unique_pairs=len(rows),words=65536))
    manifest(out)


if __name__=='__main__':
    arg=argparse.ArgumentParser()
    arg.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):main(arg.parse_args().output.resolve())
