"""Layered likelihood bounds for an independent sub-half-percent optical sampler."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits

from iteration118_auxiliary import hardware,prepare
from iteration119_benchmark import OUT,stable


def excess(scale,overlap_sq):
    # Positive eigenvalue of rho-scale*sigma for two normalized pure states.
    d=1-overlap_sq
    return 2*scale*d/(np.sqrt((scale-1)**2+4*scale*d)+scale-1)


def initialize(path,pilot,optics=None):
    global models,data,BLAS,reference
    BLAS=threadpool_limits(limits=1)
    with np.load(path) as f:data={k:f[k] for k in f}
    reference=pilot
    t,ss=hardware() if optics is None else optics;models=[]
    for s,caps in zip(ss,[(7,9,10),(5,6,6)]):
        q,S,_,fields,_=prepare(t,s,caps[0])
        retention=[prepare(t,s,k)[-1] for k in caps]
        models.append((q,S,fields,caps,retention))


def process(args):
    route,index,pilot=args[:3];timer=perf_counter()
    conditioning=args[3] if len(args)>3 else 1.8
    domain=args[4] if len(args)>4 else 119
    t=4*np.random.default_rng([domain,92,int(pilot),route,index]).random()
    stage=0;weights=[];accepted=True
    if t>=.5:
        inputs=[]
        for j,(q,S,fields,caps,R) in enumerate(models):
            key=f"{route}_{index}_{j}" if pilot else f"{route}_{j}"
            get=lambda name:data[name+key] if pilot else data[name+key][index]
            alpha=fields@get("z");v=get("v");sites=np.repeat(np.arange(len(q)),get("row"))
            inputs.append((S,v,q[sites]/alpha[sites,None],caps,R))
        if pilot:
            mid=reference[f"{route}_{index}"]
        else:
            mid=1.
            for S,v,J,caps,R in inputs:
                A,_=stable(S,v,J,caps[1],conditioning)
                mid*=abs(A[caps[1]]/A[caps[0]])**2*R[0]/R[1]
        if not np.isfinite(mid) or mid<0:
            raise ArithmeticError(("invalid middle likelihood ratio",route,index,mid))
        stage=1;weights.append(float(mid))
        if t<.8*mid:accepted=True
        elif t>=1.25*mid:accepted=False
        else:
            stage=2;hi=1.
            for S,v,J,caps,R in inputs:
                A,_=stable(S,v,J,caps[2],conditioning)
                hi*=abs(A[caps[2]]/A[caps[0]])**2*R[0]/R[2]
            if not np.isfinite(hi) or hi<0:
                raise ArithmeticError(("invalid final likelihood ratio",route,index,hi))
            weights.append(float(hi));accepted=t<hi
    return dict(direct=bool(route),index=index,threshold=float(t),stages=stage,
                weights=weights,accepted=bool(accepted),seconds=perf_counter()-timer)


def run(shots,workers,pilot):
    path=OUT/(f"high_proposals_{shots}.npz" if pilot else f"exact_bands_{shots}.npz")
    reference={}
    if pilot:
        prior=json.loads((OUT/f"subpercent_{shots}.json").read_text())["scores"]
        reference={f"{int(s['direct'])}_{s['index']}":s["w"] for s in prior}
    tasks=[(r,i,pilot) for i in range(shots) for r in (0,1)]
    start=perf_counter()
    with ProcessPoolExecutor(workers,initializer=initialize,initargs=(path,reference)) as pool:
        rows=list(pool.map(process,tasks))
    elapsed=perf_counter()-start;initialize(path,reference)
    R=np.prod([m[-1] for m in models],axis=0);base,mid=R[:2]/R[-1]
    loss=excess(4,base)+excess(1.25,mid)
    extra=.5*excess(2,base)+.8*excess(1/.8,mid)
    intrinsic=np.sqrt(1-R[-1]);bound=intrinsic+max(loss,extra)
    out={}
    for route,name in enumerate(("pnr","click")):
        accepted=[x["index"] for x in rows if x["direct"]==bool(route) and x["accepted"]]
        if not accepted:continue
        if not pilot:out[name]=data[name][accepted];continue
        items=[]
        for i in accepted:
            a,b=(data[f"row{route}_{i}_{j}"] for j in (0,1))
            items.append(a|b if route else a+b)
        out[name]=np.stack(items)
    suffix=("pilot" if pilot else "bands")+f"_{shots}"
    np.savez_compressed(OUT/f"clamped_{suffix}.npz",**out)
    info=dict(base_pair_caps=[7,5],middle_pair_caps=[9,6],target_pair_caps=[10,6],retention=R.tolist(),
              outer_clamp=[.5,4],inner_clamp=[.8,1.25],positive_deficit_upper=loss,positive_surplus_upper=extra,
              intrinsic_tv_upper=intrinsic,full_record_tv_upper=bound,elapsed_s=elapsed,workers=workers,
              decisions=rows,accepted_by_readout={k:len(a) for k,a in out.items()})
    (OUT/f"clamped_{suffix}.json").write_text(json.dumps(info,indent=2)+"\n",encoding="utf8")
    print("full upper",bound,info["accepted_by_readout"],"last stage",sum(x["stages"]==2 for x in rows),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--shots",type=int,default=8);p.add_argument("--workers",type=int,default=4)
    p.add_argument("--pilot",action="store_true");a=p.parse_args()
    with threadpool_limits(limits=1):run(a.shots,a.workers,a.pilot)
