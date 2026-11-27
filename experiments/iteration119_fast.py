"""Independent clipped draws and simultaneous uncertainty from the 4,096-row audit."""
import json
import numpy as np
from iteration119_benchmark import OUT


def run():
    n=4096;first=4096;M=8
    stats=json.loads((OUT/f"score_{first}_{n}_7.json").read_text())
    full=.03091351917028392
    saved={};rows=[]
    with np.load(OUT/f"proposal_{first}_{n}.npz") as prop,np.load(OUT/f"score_{first}_{n}_7.npz") as scores:
        for route,key in enumerate(("pnr","click")):
            w=abs(scores["z"+str(route)])**2
            u=np.array([np.random.default_rng([119,91,route,i]).random() for i in range(first,first+n)])
            keep=M*u<w
            a,b=prop[f"row{route}_0"][keep],prop[f"row{route}_1"][keep]
            saved[key]=a|b if route else a.astype(np.uint16)+b
            saved[f"indices{route}"]=np.flatnonzero(keep)+first
            # E_Q w=1. The missing unnormalized target mass is 1-E_Q min(w,M).
            # Cover both readouts and all five reported ceilings simultaneously.
            upper={}
            for m in (1,2,4,8,16):
                h=np.minimum(w,m);log=np.log(2*10/.01)
                radius=np.sqrt(2*h.var(ddof=1)*log/n)+7*m*log/(3*(n-1))
                upper[str(m)]=float(min(1,1-h.mean()+radius))
            seconds=float(prop[f"elapsed{route}"])+stats[route]["score_s_per_row"]*n
            rows.append(dict(route=key,candidates=n,accepted=int(keep.sum()),ceiling=M,wall_s=seconds,
                         seconds_per_delivered_row=seconds/keep.sum(),clipped_mass_upper99=upper,
                         full_record_tv_upper99=min(1,upper[str(M)]+full)))
    np.savez_compressed(OUT/"clipped_fast.npz",**saved)
    (OUT/"clipped_fast.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf8")
    print(rows,flush=True)


if __name__=="__main__":run()
