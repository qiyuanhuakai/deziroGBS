"""Spatial vacuum-word filters shared by PNR and threshold readout on the full device."""
import ctypes as ct
import json
from time import perf_counter

import numpy as np
import torch
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from iteration115_projection_review_v3 import factor
from iteration115_tilt_v1 import Device, setup
from iteration115_visible_v1 import DATA, I, J, P, U, hardware
from iteration103_gpu_v1 import build
from iteration116_sector_floor import BANKS, OUT


def independent_word(model, sites, banks):
    # Fit the filters using unconditional voids only. No total-N table enters.
    probs = np.empty(1<<len(banks))
    for mask in range(len(probs)):
        cols = sites(np.array([v for j,v in enumerate(banks) if mask & (1<<j)], int))
        lp = 0.
        for u,e in model:
            b = u[cols]
            sign, val = np.linalg.slogdet(np.eye(len(b))+(b*e)@b.T)
            assert sign>0
            lp -= val/2
        probs[mask] = np.exp(lp)
    for j in range(len(banks)):
        for m in range(len(probs)):
            if not m & (1<<j):
                probs[m] -= probs[m | (1<<j)]
    assert probs.min()>0 and abs(probs.sum()-1)<1e-12
    return probs


def bank_axes(banks):
    return (2*banks[:,None]+np.arange(2)).ravel()


def optical_axes(banks):
    ports = (8*banks[:,None]+np.arange(8)).ravel()
    return np.r_[ports,1152+ports]


def tv_interval(p,q):
    # Unrepresented tails can add at most their summed half mass.
    head=np.abs(p-q).sum()/2
    return [float(head), float(head+(max(0,1-p.sum())+max(0,1-q.sum()))/2)]


def words(rows,banks):
    masks = np.zeros(len(rows),np.uint16)
    for j,b in enumerate(banks):
        masks |= (rows[:,8*b:8*b+8].max(1)==0).astype(np.uint16)<<j
    return masks


def run():
    t,ss=hardware()
    target=factor(t,ss)
    begin=perf_counter()
    args, fit, distributions=setup(t,ss)
    setup_seconds=perf_counter()-begin
    spectra=[[eigh(x) for x in component] for component in distributions["marginal"]]
    models=[[(u,e) for e,u in component] for component in spectra]
    joint=np.load(OUT/"joint.npz"); p=joint["optical"]
    base=np.maximum(0,joint["base"])
    n=np.arange(30,271)
    fit_z=np.array(fit["log_tilts"])
    # Extract base covariance from the actual pre-tilt field and the untilted local blocks.
    from iteration115_banks_v1 import bank_model
    a,e,_,_=bank_model(t,ss,8)[0]
    eig=[]
    for j in range(2):
        f=a.reshape(len(a),144,4)[:,:,2*j:2*j+2].reshape(len(a),-1)
        eig.extend(eigh(2*f.T@f+np.diag(e[:,2*j:2*j+2].ravel()),eigvals_only=True))
    lg=-.5*np.log1p((1-np.exp(fit_z[:,None]))*eig).sum(1)
    previous=base[n]*(np.exp(n[:,None]*fit_z-lg)@fit["weights"])[:,None]
    held=p[n]; digit=np.arange(256)
    filters={}; result={}
    for k in (2,8):
        banks=BANKS[:k]; begin=perf_counter()
        target_word=independent_word(target,optical_axes,banks)
        proposal=sum(w*independent_word(m,bank_axes,banks) for w,m in zip(fit["weights"],models))
        weight=target_word/proposal
        training=perf_counter()-begin
        filt=weight[digit%(1<<k)]
        calibrated=previous*filt
        check=np.bincount(digit%(1<<k),weights=p.sum(0))
        np.testing.assert_allclose(target_word,check,atol=3e-13,rtol=0)
        check=np.bincount(digit%(1<<k),weights=previous.sum(0))
        omitted=1-previous.sum()
        assert np.all(proposal-check > -1e-11) and abs((proposal-check).sum()-omitted)<1e-11 and omitted<1e-5
        filters[k]=weight
        row=dict(banks=banks,filter_fit_seconds=training,theory_acceptance=float(1/weight.max()),
                 joint_tv=tv_interval(held,calibrated), total_tv=tv_interval(held.sum(1),calibrated.sum(1)),
                 fixed_116_tv=float(abs(held[86]/held[86].sum()-calibrated[86]/calibrated[86].sum()).sum()/2),
                 old_fixed_116_tv=float(abs(held[86]/held[86].sum()-previous[86]/previous[86].sum()).sum()/2),
                 target_word=target_word.tolist(),proposal_word=proposal.tolist())
        if k==2:
            panel=np.load(DATA/"iteration115_panel_v1/panel.npz")
            pattern=np.arange(65536)
            code=(pattern%256==0).astype(int)+2*(pattern//256==0)
            approx=panel["model"]*weight[code]
            assert abs(approx.sum()-1)<1e-11
            row["full_panel_click_tv"]=float(abs(approx-panel["optical"]).sum()/2)
        result[k]=row
    # Timed work includes full output transfer, scoring, rejection, and copying accepted records.
    build(OUT,"homodyne_bank115_v1.cu")
    lib=ct.CDLL(str((OUT/"gpu.dll").resolve()))
    lib.normal103_launch.argtypes=[I,I,U,J,I,P,P]
    lib.tilt115_latent.argtypes=[I]*4+[U,J]+[P]*6
    lib.tilt115_banks.argtypes=[I]*5+[U,J]+[P]*9
    torch.set_float32_matmul_precision("highest")
    device=Device(lib,args,False)
    output={}
    for k in (2,8):
        weight=filters[k]
        acceptance=weight/weight.max()
        bench=[]
        for pn in (1,0):
            batch=4096; timings=[]; kept=0; total_time=0.
            h=np.zeros(len(weight),int)
            for rep in range(33):
                rng=np.random.default_rng(np.random.SeedSequence([116,1,k,pn,rep]))
                start=perf_counter()
                rows=device.draw(batch,pn,116001000+2*k+pn,rep*batch)
                word=words(rows,BANKS[:k])
                choose=rng.random(batch)<acceptance[word]
                dest=rows[choose]
                seconds=perf_counter()-start
                if rep==0:continue
                kept+=len(dest);total_time+=seconds
                timings.append(seconds*1e6/len(dest))
                h+=np.bincount(word[choose],minlength=len(h))
                if rep==1:output[f"rows_{k}_{pn}"]=dest[:32].copy()
            assert device.error.item()==0
            proposals=batch*len(timings)
            alpha=result[k]["theory_acceptance"]
            assert abs(kept/proposals-alpha)<6*np.sqrt(alpha*(1-alpha)/proposals)
            # A multinomial check tests actual post-selected records against the independently fitted law.
            from scipy.stats import chisquare
            prob=np.array(result[k]["target_word"])
            frequent=prob*kept>=10
            actual=h[frequent]; expected=prob[frequent]*kept
            if (~frequent).any():
                actual=np.r_[actual,h[~frequent].sum()]
                expected=np.r_[expected,(prob[~frequent]*kept).sum()]
            pvalue=float(chisquare(actual,expected).pvalue)
            bench.append(dict(route="PNR" if pn else "direct_click",proposals=proposals,accepted=kept,
                       median_us=float(np.median(timings)),aggregate_us=float(total_time*1e6/kept),
                       batches_us=timings,word_fit_pvalue=pvalue,accepted_word_counts=h.tolist()))
        result[k]["bench"]=bench
        print(k,result[k]["joint_tv"],result[k]["fixed_116_tv"],[(v["route"],v["median_us"],v["word_fit_pvalue"]) for v in bench],flush=True)
    np.savez_compressed(OUT/"example_rows.npz",**output)
    (OUT/"filtered.json").write_text(json.dumps(dict(base_setup_seconds=setup_seconds,filters=result),indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    with threadpool_limits(limits=4):
        run()
