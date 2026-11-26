"""Four consecutive optical groups, resolving the oldest count through a balanced virtual splitter."""
import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from iteration102_triads_v1 import TRI, triple
from iteration102_forest_v1 import FIT, DATA, BASE, Tables
from iteration100_spatial_v1 import ORDER, Native, compact, realify, cdf_rows, realized, save, manifest

FOUR=DATA/'iteration102_four_v1'
BINS=4


def occupancy(cap,ports):
    law=np.zeros((cap+1,ports+1))
    law[0,0]=1
    used=np.arange(ports+1)
    for n in range(cap):
        law[n+1]=law[n]*used/ports
        law[n+1,1:]+=law[n,:-1]*(ports-used[:-1])/ports
    return law


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    start=perf_counter()
    t=Tables(FIT/'prepared.npz')
    prev=dict(np.load(TRI/'prepared.npz'))
    second=prev['second']
    spec=np.load(ORDER)
    v=spec['v'][np.argsort(spec['order'])]
    maps=[realify(v[8*g:8*g+8]) for g in t.order]
    maps={g:m for g,m in zip(t.order,maps)}
    mats={g:compact(m.T@m) for g,m in maps.items()}
    lib=Native(BASE/'spatial.dll')
    tasks=[int(g) for g in t.order if second[g]>=0 and t.parent[second[g]]>=0]
    split=occupancy(int(t.caps.max()),BINS)
    def run(g):
        c=int(t.parent[g])
        b=int(second[g])
        a=int(t.parent[b])
        h=maps[a]
        tilted=[]
        # E[u^N_a 1{n_b,n_c,n_d}] is a subnormalized Gaussian law.
        for j in range(BINS+1):
            loss=1-j/BINS
            logp=0.
            xs=[]
            for x in spec['x']:
                cross=h@x
                m=np.eye(len(h))+loss*cross@h.T
                logp-=.5*np.linalg.slogdet(m)[1]
                xs.append(x-loss*cross.T@np.linalg.solve(m,cross))
            tilted.append(np.exp(logp)*triple(lib,[mats[i] for i in [g,c,b]],xs,t.caps[[g,c,b]]))
        joint=np.array([math.comb(BINS,k)*sum((-1)**(k-j)*math.comb(k,j)*tilted[j] for j in range(k+1))
                        for k in range(BINS+1)])
        neg=float(-joint[joint<0].sum())
        clean=joint.clip(0)
        clean[clean.sum(3)==0]=t.marg[g]
        cuts=cdf_rows(clean)
        # Separate optical inversion on the preceding triple supplies the
        # categorical separator distribution, independent of the tilted inversion.
        earlier=triple(lib,[mats[i] for i in [c,b,a]],spec['x'],t.caps[[c,b,a]])
        enc=split[:len(earlier)]
        separator=np.einsum('aq,abc->qbc',enc,earlier,optimize=True)
        sep_tv=float(abs(joint.sum(3)-separator).sum()/2)
        actual=separator[:,:,:,None]*realized(cuts)
        tv=float(abs(joint-actual).sum()/2)
        incoming=earlier.clip(0)
        endpoint=np.zeros((len(earlier),cuts.shape[-1]))
        for k in range(BINS+1):
            endpoint+=enc[:,k,None]*np.einsum('abc,bcd->ad',incoming,realized(cuts[k]),optimize=True)
        assert max(sep_tv,tv,neg)<5e-10,(a,b,c,g,sep_tv,tv,neg)
        np.savez_compressed(out/f'endpoint_{a}_{g}.npz',counts=endpoint)
        print('four',a,b,c,g,tv,flush=True)
        return g,cuts,dict(groups=[a,b,c,g],separator_tv=sep_tv,numeric_joint_tv=tv,negative_mass=neg)
    offsets=prev['offset'].copy()
    third=np.full(len(t.order),-1,np.int32)
    pos=len(prev['cdf'])
    extra=[]
    rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for g,cuts,row in pool.map(run,tasks):
            offsets[g]=pos
            pos+=cuts.size
            extra.append(cuts.ravel())
            third[g]=row['groups'][0]
            rows.append(row)
    np.savez_compressed(out/'extension.npz',extra=np.concatenate(extra),offset=offsets,third=third,classes=cdf_rows(split))
    save(out,'summary.json',dict(virtual_ports=BINS,rows=rows,additional_vacuum_pairs=len(rows),
         prepare_s=perf_counter()-start,extra_cdf_words=sum(e.size for e in extra)))
    manifest(out)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):
        prepare(ap.parse_args().output.resolve())
