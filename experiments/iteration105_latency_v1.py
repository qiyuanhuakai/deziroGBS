"""Match the four-record 1,152-port Frugal job and separate CPU parallelism from optical approximation."""
import argparse
from pathlib import Path
from time import perf_counter_ns
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save, manifest
from iteration105_jiuzhang_v1 import inputs
from iteration105_two_mode_v1 import TwoMode, OUT


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    _,transfer,r=inputs()
    start=perf_counter_ns()
    sampler=TwoMode(OUT/'two.dll',transfer,np.repeat(r[:,None],2,axis=1),np.ones(len(r),bool))
    prepare=(perf_counter_ns()-start)*1e-9
    rows=[]
    for batch in (4,2048):
        for workers in (1,16):
            for pnr in (False,True):
                times=[]
                for rep in range(33 if batch==4 else 11):
                    seed=105030000+int(pnr)+2*workers+100*batch
                    start=perf_counter_ns()
                    x=sampler.draw(batch,pnr,seed,rep*batch,workers)
                    elapsed=(perf_counter_ns()-start)/(batch*1000)
                    if rep>=2:times.append(elapsed)
                    assert x.shape==(batch,1152) and (pnr or x.max()<=1)
                rows.append(dict(batch=batch,workers=workers,pnr=pnr,median_us=float(np.median(times)),quartiles=np.quantile(times,[.25,.75]).tolist(),times=times))
    save(out,'summary.json',dict(prepare_s=prepare,rows=rows))
    manifest(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):run(parser.parse_args().output.resolve())
