"""Entire eight-detector threshold law on every bank, for each official optical input."""
import argparse
import json
import numpy as np
from threadpoolctl import threadpool_limits

from iteration115_projection_review_v3 import factor
from iteration115_visible_v1 import hardware
from iteration117_power_inputs import load_power
from iteration117_quartic_laws import physical_models,marginal,contraction
from iteration117_quartic import OUT


def invert(v):
    q=v[:,::-1].astype(np.longdouble)
    for j in range(8):
        view=q.reshape(len(q),-1,2,1<<j);view[:,:,1]-=view[:,:,0]
    assert q.min()>-1e-12
    return np.array(q,float)


def run(power):
    optical=hardware() if power=="high" else load_power(power)
    truth=factor(*optical)
    mods=[physical_models(optical,transport=b) for b in (False,True)]
    subsets=[np.flatnonzero(mask&(1<<np.arange(8))) for mask in range(256)]
    res=[]
    for bank in range(144):
        ports=8*bank+np.arange(8)
        axes=np.r_[ports,1152+ports]
        target=[(u[axes]*e)@u[axes].T for u,e in truth]
        parts=[[marginal(model,ports) for model in pair] for pair in mods]
        vv=np.ones((3,256))
        for mask,ids in enumerate(subsets[1:],1):
            ix=np.r_[ids,8+ids];ii=np.ix_(ix,ix);eye=np.eye(len(ix))
            vv[0,mask]=np.exp(-.5*sum(np.linalg.slogdet(eye+x[ii])[1] for x in target))
            for j,variants in enumerate(parts,1):
                for S,norm,t,x in variants:
                    den=eye+x[ii]
                    val=contraction(S,norm,t[ids],np.linalg.inv(den),np.exp(-.5*np.linalg.slogdet(den)[1]))
                    assert abs(val.imag)<1e-11
                    vv[j,mask]*=val.real
        res.append(invert(vv))
    laws=np.array(res);tv=abs(laws[:,1:]-laws[:,:1]).sum(2)/2
    row=dict(power=power,banks=len(tv),model_order=["quartic","coherence_matched"],
             mean_tv=tv.mean(0).tolist(),max_tv=tv.max(0).tolist(),median_tv=np.median(tv,axis=0).tolist(),
             improved_with_transport=int((tv[:,1]<tv[:,0]).sum()),tv_by_bank=tv.tolist())
    np.savez_compressed(OUT/("banks_"+power+".npz"),optical=laws[:,0],quartic=laws[:,1],transport=laws[:,2])
    (OUT/("banks_"+power+".json")).write_text(json.dumps(row,indent=2)+"\n",encoding="utf8")
    print(power,row["mean_tv"],row["max_tv"],row["improved_with_transport"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("power",choices=("low","median","high"))
    with threadpool_limits(limits=1):run(p.parse_args().power)
