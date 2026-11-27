"""Run complete 1,152-port records with independent auxiliary, field and readout streams."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration118_auxiliary import OUT,draw,prepare,hardware


def initialize(k,strong_only=False):
    global models,limit,cap
    from threadpoolctl import threadpool_limits
    limit=threadpool_limits(limits=1);cap=k
    t,ss=hardware()
    models=[prepare(t,s,k if j==0 or not strong_only else 5) for j,s in enumerate(ss)]


def record(args):
    route,index,strong_only=args
    start=perf_counter();lst=[];tries=[];checks=[]
    for j,model in enumerate(models):
        k=5 if strong_only and j else cap
        diag={};row,tr=draw(model,k,route,[118,610,cap,int(strong_only),route,j,index],diag)
        checks.append(diag)
        lst.append(row);tries.append(tr)
    combined=(lst[0]|lst[1]) if route else (lst[0]+lst[1])
    return combined,perf_counter()-start,tries,checks


def run(k,shots,workers,strong_only=False):
    OUT.mkdir(exist_ok=True)
    initialize(k,strong_only);bounds=[m[-1] for m in models];rows=[];arrays={}
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(k,strong_only)) as pool:
        list(pool.map(record,[(0,2**30+i,strong_only) for i in range(workers)]))
        for route in (0,1):
            start=perf_counter()
            values=list(pool.map(record,[(route,i,strong_only) for i in range(shots)]))
            wall=perf_counter()-start
            data=np.stack([v[0] for v in values]);tries=np.array([v[2] for v in values])
            times=np.array([v[1] for v in values])
            arrays["click" if route else "pnr"]=data
            rows.append(dict(direct=bool(route),shots=shots,elapsed_s=wall,seconds_per_row=wall/shots,
                      median_single_worker_s=float(np.median(times)),times_s=times.tolist(),
                      prefix_checks=[v[3] for v in values],
                      heterodyne_attempts=tries.tolist(),mean_clicks=float((data>0).sum(1).mean()),
                      mean_photons=None if route else float(data.sum(1).mean())))
            print(route,shots,wall/shots,rows[-1]["mean_clicks"],flush=True)
    name=f"cap{2*k}"+("_weak10" if strong_only else "")+f"_{shots}"
    np.savez_compressed(OUT/(name+".npz"),**arrays)
    (OUT/(name+".json")).write_text(json.dumps(dict(pairs_per_spectrum=[k,5 if strong_only else k],
          retained=bounds,full_TV_upper=float(np.sqrt(1-np.prod(bounds))),workers=workers,rows=rows),indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--pairs",type=int,required=True);a.add_argument("--shots",type=int,required=True)
    a.add_argument("--workers",type=int,default=4);a.add_argument("--weak-ten",action="store_true");args=a.parse_args()
    run(args.pairs,args.shots,args.workers,args.weak_ten)
