"""Apply the identical untrained core model to all three independently published inputs."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from iteration114_pairs_v1 import sector
from iteration115_visible_v1 import hardware
from iteration115_displacement_v1 import conditional
from iteration115_projection_review_v3 import factor
from iteration116_sector_floor import BANKS
from iteration117_power_inputs import load_power
from iteration117_quartic import OUT
from iteration117_quartic_laws import physical_models,marginal,total_curve,contraction,void,GRID


def invert_void(v):
    ans=np.array(v)
    for bit in range(8):
        for i in range(256):
            if not i&(1<<bit):ans[i]-=ans[i|(1<<bit)]
    assert ans.min()>0
    return ans


def run(transport=False):
    rows={}
    for power in ("low","median","high"):
        t,ss=hardware() if power=="high" else load_power(power)
        raw=factor(t,ss);mods=physical_models((t,ss),transport)
        z=np.exp(-2j*np.pi*np.arange(GRID//2+1)/GRID);w=1-z
        target=np.prod([np.exp(-.5*np.log1p(w[:,None]*e).sum(1)) for _,e in raw],axis=0)
        model=np.prod([total_curve(marginal(m,None),GRID) for m in mods],axis=0)
        photon_tv=float(np.abs(np.fft.irfft(model-target,n=GRID)).sum()/2)
        radius=1.08
        true_tail=radius**-GRID*np.exp(-.5*sum(np.log1p((1-radius)*e).sum() for _,e in raw))
        model_tail=radius**-GRID
        for m in mods:
            S,norm,u,x=marginal(m,None);ev,basis=np.linalg.eigh(x);v=1-radius
            model_tail*=contraction(S,norm,u,(basis*(v/(1+v*ev)))@basis.T,np.exp(-.5*np.log1p(v*ev).sum())).real
        true_word=[];quartic=[]
        for m in range(256):
            b=np.array([v for j,v in enumerate(BANKS) if m&(1<<j)],int)
            sites=(8*b[:,None]+np.arange(8)).ravel()
            axes=np.r_[sites,1152+sites]
            true_word.append(np.exp(-.5*sum(np.linalg.slogdet(np.eye(len(axes))+(u[axes]*e)@u[axes].T)[1]
                               for u,e in raw)))
            quartic.append(np.prod([void(spec,sites) for spec in mods]))
        word_tv=float(abs(invert_void(true_word)-invert_void(quartic)).sum()/2)
        gains=[]
        for s in ss:
            _,v,_,_,_=conditional(t,s)
            gains.append(.5*np.log(np.linalg.eigvalsh(v)[50:]))
        rows[power]=dict(mean_optical=float(sum(e.sum() for _,e in raw)/2),
                         mean_quartic=float(np.fft.irfft(model,n=GRID)@np.arange(GRID)),
                         total_tv=[photon_tv,float(photon_tv+true_tail+model_tail)],
                         eight_bank_word_tv=word_tv,
                         full_TV_upper_by_core_cap={k:float(np.sqrt(1-np.prod([sector(g,k)[0] for g in gains])))
                             for k in (0,2,4,6,8,12,16,20)})
        if transport:
            rows[power]["original_core_upper_by_cap"]=rows[power].pop("full_TV_upper_by_core_cap")
            from iteration117_transport import corrected
            diagnostics=[]
            for s,model,gain in zip(ss,mods,gains):
                _,_,_,resp,_=conditional(t,s)
                _,info=corrected(model[1],resp,sector(gain,4)[0]);diagnostics.append(info)
            rows[power]["transport"]=diagnostics
            rows[power]["full_TV_upper"]=float(np.sqrt(1-np.prod([v["root_fidelity_lower"] for v in diagnostics])**2))
        print(power,rows[power],flush=True)
    (OUT/("transport_laws.json" if transport else "power_laws.json")).write_text(json.dumps(rows,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--transport",action="store_true")
    with threadpool_limits(limits=2):run(parser.parse_args().transport)
