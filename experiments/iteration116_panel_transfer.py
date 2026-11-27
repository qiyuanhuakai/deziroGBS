"""Full panel PNR brackets and click laws after a possibly remote vacuum-word filter."""
import itertools
import json

import numpy as np
from threadpoolctl import threadpool_limits

from iteration106_field_v1 import actual_input
from iteration107_cliques_v1 import optical_x
from iteration115_panel_accuracy_v2 import bank_counts, dark_projection, tail, tv
from iteration116_sector_floor import BANKS, DATA, OUT

GRID=128
Z=1-np.exp(2j*np.pi*np.arange(GRID)/GRID)
POWER=np.array([np.ones(GRID),Z,Z*Z])
PARTS=[(sub,sum(i<2 for i in sub),sum(i>=2 for i in sub))
       for size in range(1,5) for sub in itertools.combinations(range(4),size)]


def pair_pgf(x):
    # Principal-minor expansion of det(I + X diag(a,a,b,b)).
    coef=np.zeros((3,3));coef[0,0]=1.
    for ids,ia,ib in PARTS:
        coef[ia,ib]+=np.linalg.det(x[np.ix_(ids,ids)])
    a=coef@POWER
    det=a[0]+Z[:,None]*a[1]+Z[:,None]**2*a[2]
    # Continue from the origin of the unit-angle torus. Check the sampling of each phase walk.
    phase=np.unwrap(np.unwrap(np.angle(det),axis=0),axis=1)
    for axis in (0,1):
        assert abs(np.diff(phase,axis=axis)).max()<1.5
    return np.exp(-.5j*phase)/np.sqrt(abs(det))


def split_words(weights):
    # Stochastic channel from n incident photons to the set of occupied ports.
    p=np.zeros((GRID,256));p[0,0]=1.
    for n in range(1,GRID):
        for j,w in enumerate(weights):
            dest=np.arange(256)|(1<<j)
            p[n]+=w*np.bincount(dest,weights=p[n-1],minlength=256)
    np.testing.assert_allclose(p.sum(1),1.,atol=2e-13,rtol=0)
    return p


def law_for_void(pair,other,model):
    sites=(2*np.array(pair)[:,None]+np.arange(2)).ravel()
    remote=(2*np.array(other,dtype=int)[:,None]+np.arange(2)).ravel()
    grid=np.zeros((GRID,GRID),complex)
    for w, component in zip(model["weights"],model["marginal"]):
        term=w
        for mat in component:
            x=mat[np.ix_(sites,sites)]
            if len(remote):
                b=mat[np.ix_(sites,remote)]
                d=np.eye(len(remote))+mat[np.ix_(remote,remote)]
                term=term*np.exp(-.5*np.linalg.slogdet(d)[1])
                x=x-b@np.linalg.solve(d,b.T)
            term=term*pair_pgf(x)
        grid+=term
    p=np.fft.fft2(grid)/GRID**2
    assert abs(p.imag).max()<1e-12
    return p.real


def run():
    model=dict(np.load(DATA/"iteration115_tilt_v1/model.npz"))
    prior=np.load(DATA/"iteration115_review_v3/projection_panels.npz")
    rows=json.loads((OUT/"filtered.json").read_text())["filters"]
    raw,gains=actual_input()
    results={};arrays={}
    for pair in [(34,35),(71,72),(102,143)]:
        tag="_".join(map(str,pair))
        transfer=raw.reshape(144,8,-1)[list(pair)]
        opt=optical_x(transfer.reshape(16,-1),gains)
        truth=bank_counts(opt,[*range(8),*range(16,24)])
        simple,leak,split=dark_projection(transfer,gains)
        np.testing.assert_allclose(split,np.diff(model["split"][list(pair)],prepend=0,axis=1),atol=1e-12)
        compressed=bank_counts(simple,[0,1])
        ptail=tail(simple)
        axes=(2*np.array(pair)[:,None]+np.arange(2)).ravel()
        base_tail=sum(w*tail(x[:,axes][:,:,axes]) for w,x in zip(model["weights"],model["marginal"]))
        empty=[i for i,b in enumerate(BANKS) if b not in pair]
        outside=[BANKS[i] for i in empty]
        joint=[]
        for mask in range(1<<len(outside)):
            joint.append(law_for_void(pair,[b for j,b in enumerate(outside) if mask & (1<<j)],model))
        joint=np.array(joint)
        base=joint[0].copy()
        for j in range(len(outside)):
            b=1<<j
            for m in range(len(joint)):
                if not m & b:joint[m]-=joint[m|b]
        assert joint.min()>-2e-13
        np.testing.assert_allclose(joint.sum(0),base,atol=1e-13)
        keys=np.array([sum(1<<i for j,i in enumerate(empty) if m & (1<<j)) for m in range(len(joint))])
        ids=keys[:,None,None]+np.zeros((1,GRID,GRID),int)
        for j,bank in enumerate(pair):
            if bank in BANKS:
                if j==0:ids[:,0,:]|=1<<BANKS.index(bank)
                else:ids[:,:,0]|=1<<BANKS.index(bank)
        split1,split2=map(split_words,split)
        case={}
        for k in (0,2,8):
            if k:
                row=rows[str(k)]
                w=np.array(row["target_word"])/row["proposal_word"]
                law=(joint*w[ids%len(w)]).sum(0)
                # For the wrapped table, a multiple of GRID could be mistaken for vacuum.
                # Bound both the wrong multiplier and lifting the wrapped counts back to N.
                err=3*float(w.max())*base_tail
            else:
                law=base;err=base_tail
            assert abs(law.sum()-1)<2e-11 and law.min()>-1e-12
            click=(split2.T@law.T@split1).ravel()
            lo=max(tv(law,truth)-err,0.)
            upper=tv(law,compressed)+err+ptail+np.sqrt(leak)
            cc=tv(click,prior["truth_"+tag])
            case[k]=dict(pnr_lower=float(lo),pnr_upper=float(upper),full_click_tv=cc,
                         click_wrap_bound=err,local_dark_probability=float(leak))
            arrays[f"pnr_{tag}_{k}"]=law
            arrays[f"click_{tag}_{k}"]=click
        if pair==(34,35):
            assert abs(case[2]["full_click_tv"]-rows["2"]["full_panel_click_tv"])<1e-10
            prev=json.loads((DATA/"iteration115_panel_accuracy_v2/pnr_bounds.json").read_text())
            assert abs(case[0]["pnr_lower"]-prev["full_pnr_tv_lower"])<1e-10
        results[tag]=case
        print(tag,[(k,v["full_click_tv"],v["pnr_lower"],v["pnr_upper"]) for k,v in case.items()],flush=True)
    np.savez_compressed(OUT/"panel_transfer.npz",**arrays)
    (OUT/"panel_transfer.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    with threadpool_limits(limits=3):
        run()
