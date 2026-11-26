"""Independent orthant, bridged-forest holdout and four-shot full-width CUDA comparisons."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtri
from threadpoolctl import threadpool_limits

from iteration100_spatial_v1 import save,manifest
from iteration103_fit_v1 import FIT,Orthant
from iteration103_census_v1 import CENSUS,CountLaw
from iteration103_transport_gpu_v1 import Sampler,RUN,OUT,Cuda,Correction,torch
from iteration103_sample_v1 import DATA


def validate(out):
    out.mkdir(parents=True,exist_ok=False)
    law=CountLaw(CENSUS/'chaos.dll')
    solver=Orthant(FIT/'orthant.dll')
    diff=[]
    for a,b in ((36,107),(34,35),(25,139),(64,73),(0,1),(12,51)):
        joint,rem,_=law.pair(a,b,1e-14)
        r=law.corr[2*a:2*a+2,2*b:2*b+2].ravel()
        keys=[(k,l) for k in (1,2,3,6) for l in (1,2,3,6)]
        def thresholds(g,k):
            p=law.p[g]
            return [ndtri(p[1:].sum()),ndtri(p[max(k,2):].sum()/p[1:].sum())]
        x=np.array([thresholds(a,k)+thresholds(b,l) for k,l in keys])
        found=solver.cdf(x,np.tile(r,(len(keys),1)),np.array([(min(k,2),min(l,2)) for k,l in keys]),32)
        diff.extend(abs(found-np.array([joint[k:,l:].sum() for k,l in keys])))
    assert max(diff)<2e-10
    model=Sampler(RUN/'draw.dll',RUN/'bridges.npz',True)
    rows=json.loads((DATA/'iteration103_projection_v1/summary.json').read_text())['rows']
    pairs=np.array([r['groups'] for r in rows if 'bridge' in r or r.get('previous_kind') in ('edge','triad')])
    assert len(pairs)==129
    with np.load(CENSUS/'scan.npz') as s:
        lookup={(a,b):i for i,(a,b) in enumerate(zip(s['a'],s['b']))}
        truth=s['coarse_optical'][[lookup[tuple(p)] for p in pairs]]
    n=524288
    observed=np.zeros((len(pairs),49),np.int64)
    for off in range(0,n,2048):
        z,c=model.sample(2048,True,16,103101000,off)
        assert np.array_equal(z.reshape(len(z),144,8).sum(2),c)
        cc=c.clip(0,6)
        code=cc[:,pairs[:,0]]*7+cc[:,pairs[:,1]]+49*np.arange(len(pairs))
        observed+=np.bincount(code.ravel(),minlength=observed.size).reshape(observed.shape)
    tv=abs(observed.reshape(-1,7,7)/n-truth).sum((1,2))/2
    radius=float(np.sqrt((49*np.log(2)+np.log(len(pairs)/.01))/(2*n)))
    assert tv.max()<radius
    np.savez_compressed(out/'linked.npz',pairs=pairs,counts=observed)
    base=Sampler(RUN/'draw.dll',RUN/'bridges.npz',False)
    torch.set_float32_matmul_precision('highest')
    gpu=Cuda(OUT/'gpu.dll',base,False)
    tail=Correction(OUT/'gpu.dll',base)
    timings=[]
    for i in range(25):
        tic=perf_counter()
        z,c=gpu.sample(4,False,103102000,4*i,tail)
        if i>3:timings.append((perf_counter()-tic)*1e6/4)
        assert z.shape==(4,1152)
    save(out,'summary.json',dict(orthant_checks=len(diff),orthant_max_error=max(diff),linked_pairs=len(pairs),linked_n=n,linked_max_7by7_tv=float(tv.max()),linked_radius99=radius,gpu_four_shot_click_us=float(np.median(timings)),gpu_timings=timings))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):validate(p.parse_args().output.resolve())
