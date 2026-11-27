"""Generate fresh full-width band candidates and independently corrected records."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
import iteration119_mixture as band
import iteration119_clamps as clamp
from iteration120_replay import OUT,stable,init

FIRST=2**20


def init_band():
    band.initialize()
    band.stable=stable


def generate(route,n,workers):
    OUT.mkdir(exist_ok=True)
    path=OUT/f'base_{route}_{n}.npz'
    if path.exists():raise FileExistsError(path)
    start_time=perf_counter()
    init_band()
    t,ss=band.hardware()
    setups=[band.quartic(t,s) for s in ss]
    arrays={};stats=[]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp,ProcessPoolExecutor(workers,initializer=init_band) as pool:
        lib=band.native(tmp)
        for j,setup in enumerate(setups):
            start=FIRST;selected=[];log=[]
            while len(selected)<n:
                # Schedule independent proposals across multiple batches before
                # waiting for rare expensive bands. Always retain proposal order.
                need=n-len(selected);cost=band.models[j][-1].sum()
                count=max(16,int(np.ceil(need*cost/16))*16)
                tasks=chain.from_iterable(band.generate_batch(lib,setup,route,j,i,16)
                                          for i in range(start,start+count,16))
                for row,info in pool.map(band.task,tasks):
                    log.append(info)
                    if row is not None:selected.append(row)
                start+=count
                print(route,j,'accepted',len(selected),'of',start-FIRST,flush=True)
            key=f'{route}_{j}'
            records,fields,latent=map(np.stack,zip(*selected[:n]))
            arrays['row'+key]=records;arrays['v'+key]=fields;arrays['z'+key]=latent;arrays['log'+key]=np.array(log)
            stats.append(dict(spectrum=j,proposals=len(log),successful=len(selected),band_counts=np.bincount(
                np.array(log)[:,1].astype(int),minlength=len(band.models[j][1])).tolist()))
        if os.name=='nt':
            import _ctypes
            _ctypes.FreeLibrary(lib._handle)
    a,b=[arrays[f'row{route}_{j}'] for j in (0,1)]
    arrays['click' if route else 'pnr']=a|b if route else np.add(a,b,dtype=np.uint32)
    np.savez_compressed(path,**arrays)
    info=dict(direct=bool(route),candidates=n,first_proposal=FIRST,workers=workers,
              wall_seconds=perf_counter()-start_time,spectra=stats)
    path.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
    print(info,flush=True)


def score(route,n,workers):
    path=OUT/f'base_{route}_{n}.npz';name='click' if route else 'pnr'
    target=OUT/f'full_{route}_{n}.npz'
    if target.exists():raise FileExistsError(target)
    timer=perf_counter()
    with ProcessPoolExecutor(workers,initializer=init,initargs=(path,)) as pool:
        rows=list(pool.map(clamp.process,[(route,i,False,1.8,120) for i in range(n)]))
    keep=[x['index'] for x in rows if x['accepted']]
    with np.load(path) as f:np.savez_compressed(target,**{name:f[name][keep],'accepted_indices':keep})
    base=json.loads(path.with_suffix('.json').read_bytes())
    duration=perf_counter()-timer
    info=dict(direct=bool(route),candidates=n,accepted=len(keep),base_workers=base['workers'],scoring_workers=workers,
              base_wall_seconds=base['wall_seconds'],scoring_wall_seconds=duration,
              seconds_per_accepted=(base['wall_seconds']+duration)/len(keep),decisions=rows)
    target.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
    print({k:v for k,v in info.items() if k!='decisions'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('base','score'));p.add_argument('route',type=int,choices=(0,1))
    p.add_argument('--candidates',type=int,default=256);p.add_argument('--workers',type=int,required=True)
    p.add_argument('--output',type=Path,default=OUT)
    a=p.parse_args()
    OUT=a.output.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        (generate if a.stage=='base' else score)(a.route,a.candidates,a.workers)
