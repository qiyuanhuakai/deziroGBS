"""Replay the published two-stage decisions with the compiled contractions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
import iteration119_clamps as clamps
from iteration120_contractions import amplitudes

OUT=clamps.OUT.parent/'iteration120_contractions'


def stable(S,v,J,cap,threshold,compensated=False):
    width=np.linalg.norm(J,axis=1)*np.sqrt(np.linalg.norm(S,2))
    weak=np.flatnonzero(width>threshold)
    device='cuda' if cap>=9 and len(J)>=32 else None
    if not len(weak):return amplitudes(S,v,J,cap,device,compensated),0
    scale=1+width[weak]/threshold
    total=np.zeros(cap+1,complex)
    for phase in np.exp(2j*np.pi*(np.arange(len(weak)+1)+.25)/(len(weak)+1)):
        den=1+scale*phase;rows=J.copy();rows[weak]/=den[:,None]
        total+=den.prod()*amplitudes(S,v,rows,cap,device,compensated)
    return total/(len(weak)+1),len(weak)


def init(path,optics=None):
    import torch
    torch.set_num_threads(1)
    clamps.initialize(path,{},optics)
    clamps.stable=stable


def run(workers,shots):
    OUT.mkdir(exist_ok=True)
    source=clamps.OUT/'exact_bands_64.npz'
    reference=json.loads((clamps.OUT/'clamped_bands_64.json').read_bytes())['decisions']
    report=[]
    with ProcessPoolExecutor(workers,initializer=init,initargs=(source,)) as pool:
        for route in range(2):
            start=perf_counter()
            rows=list(pool.map(clamps.process,[(route,i,False) for i in range(shots)]))
            elapsed=perf_counter()-start
            ref=[x for x in reference if x['direct']==bool(route)][:shots]
            discrepancy=0.
            for a,b in zip(ref,rows,strict=True):
                assert (a['index'],a['accepted'],a['stages'])==(b['index'],b['accepted'],b['stages'])
                for x,y in zip(a['weights'],b['weights'],strict=True):
                    discrepancy=max(discrepancy,abs(x-y))
                    assert abs(x-y)<1e-7*max(1,abs(x)),(a,b)
            result=dict(direct=bool(route),candidates=shots,accepted=sum(x['accepted'] for x in rows),
                        scoring_wall_s=elapsed,max_absolute_weight_discrepancy=discrepancy,rows=rows)
            report.append(result)
            print({k:v for k,v in result.items() if k!='rows'},flush=True)
    (OUT/f'replay_{shots}_{workers}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--shots',type=int,default=64)
    a=p.parse_args();run(a.workers,a.shots)
