"""Two-bank void correlations for all 10,296 distinct bank pairs in each input."""
import argparse
import json
import numpy as np
from threadpoolctl import threadpool_limits
from iteration115_visible_v1 import hardware
from iteration115_projection_review_v3 import factor
from iteration117_power_inputs import load_power
from iteration117_quartic_laws import physical_models,void
from iteration117_quartic import OUT


def run(power):
    optical=hardware() if power=="high" else load_power(power)
    truth=factor(*optical)
    models=[physical_models(optical,transport=b) for b in (False,True)]
    single=np.load(OUT/("banks_"+power+".npz"))
    empty=np.stack([single[key][:,0] for key in ("optical","quartic","transport")])
    pairs=np.array(np.triu_indices(144,1)).T
    joint=np.empty((len(pairs),3));tv=np.zeros((len(pairs),2));errors=tv.copy()
    for index,(i,j) in enumerate(pairs):
        ports=(8*np.array([i,j])[:,None]+np.arange(8)).ravel()
        idx=np.r_[ports,1152+ports]
        joint[index,0]=np.exp(-.5*sum(np.linalg.slogdet(np.eye(len(idx))+(u[idx]*e)@u[idx].T)[1] for u,e in truth))
        for a,mods in enumerate(models,1):joint[index,a]=np.prod([void(m,ports) for m in mods])
        v=joint[index]
        law=np.stack([v,empty[:,i]-v,empty[:,j]-v,1-empty[:,i]-empty[:,j]+v],axis=1)
        assert law.min()>-1e-12
        tv[index]=abs(law[1:]-law[0]).sum(1)/2
        connected=v-empty[:,i]*empty[:,j]
        errors[index]=abs(connected[1:]-connected[0])
    d=dict(power=power,pairs=len(pairs),model_order=["quartic","coherence_matched"],
           mean_pair_tv=tv.mean(0).tolist(),max_pair_tv=tv.max(0).tolist(),
           mean_abs_connected_error=errors.mean(0).tolist(),max_abs_connected_error=errors.max(0).tolist(),
           pairs_improved_tv=int((tv[:,1]<tv[:,0]).sum()),
           pairs_improved_connected=int((errors[:,1]<errors[:,0]).sum()))
    np.savez_compressed(OUT/("pairs_"+power+".npz"),banks=pairs,joint_empty=joint,tv=tv,connected_error=errors)
    (OUT/("pairs_"+power+".json")).write_text(json.dumps(d,indent=2)+"\n",encoding="utf8")
    print(d,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("power",choices=("low","median","high"))
    with threadpool_limits(limits=1):run(p.parse_args().power)
