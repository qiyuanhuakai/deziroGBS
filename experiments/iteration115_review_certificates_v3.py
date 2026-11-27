"""Conditional-probability integration for positive-P and independent fixed-N checks."""
import json
import numpy as np
from scipy.linalg import eigh
from scipy.special import gammaln
from threadpoolctl import threadpool_limits
from iteration115_goodman_review_v3 import OUT,DATA,PORTS,fields,corrected
from iteration115_projection_review_v3 import hardware,factor,project,TEST_N
from iteration115_visible_v1 import energy_law

BLOCK=4096
ROWS=1048576


def words(v):
    # Bit j indexes channel j. Columns are conditional probabilities of the eight-bit word.
    tab=np.ones((len(v),1))
    for col in v.T:
        z=col[:,None]
        tab=np.column_stack((tab*z,tab*(1-z)))
    return tab


def capped_poisson(rate):
    probs=np.empty((len(rate),9))
    probs[:,0]=np.exp(-rate)
    for n in range(1,8):probs[:,n]=probs[:,n-1]*rate/n
    probs[:,8]=1-probs[:,:8].sum(1)
    return probs


def photon_sector(models,axes):
    retained=[];original=[];p0=1.
    for u,e in models:
        b=u[axes]
        den=np.eye(len(b))+(b*e)@b.T
        be=b*e
        small=np.diag(e)-be.T@np.linalg.solve(den,be)
        gram=np.eye(len(e))-b.T@b
        a,v=eigh(gram);root=(v*np.sqrt(np.maximum(0.,a)))@v.T
        retained.extend(np.linalg.eigvalsh(root@small@root))
        original.extend(e)
        p0*=np.exp(-.5*np.linalg.slogdet(den)[1])
    keep,_=energy_law(retained,160)
    total,_=energy_law(original,160)
    return p0*keep[TEST_N]/total[TEST_N]


def interval(halves,ref,alpha):
    n=ROWS
    estimate=halves.mean(0)
    tv=float(np.abs(estimate-ref).sum()/2)
    # Upper: convexity + bounded differences for the TV of the Rao--Blackwell estimator.
    high=tv+np.sqrt(np.log(1/alpha)/(2*n))
    # Lower: independent second half assesses a sign witness selected on the first.
    event=halves[0]>ref
    delta=float((halves[1]-ref)[event].sum())
    low=max(0.,delta-np.sqrt(np.log(1/alpha)/n))
    return dict(point=tv,lower=low,upper=high,heldout_event_gap=delta)


def run():
    data=np.load(OUT/'positive_p_panel.npz')
    opt=np.load(DATA/'iteration115_panel_v1/panel.npz')['optical']
    pn=np.load(DATA/'iteration115_panel_accuracy_v2/pnr_laws.npz')['optical_pair']
    t,s=hardware(); stats=[]; arrays={}
    for route in (0,1):
        rng=np.random.default_rng(np.random.SeedSequence([115,385,route]))
        layers=list(zip(data[f'{route}_layers'],data[f'{route}_offsets']))
        h=np.zeros((2,65536));p=np.zeros((2,81))
        for off in range(0,ROWS,BLOCK):
            y=corrected(fields(rng,BLOCK,t[PORTS],s),layers,route)
            v=np.exp(-y) if route else y
            a,b=words(v[:,:8]),words(v[:,8:])
            half=off//(ROWS//2)
            h[half]+=(b.T@a).ravel()
            if route:p[half]+=(capped_poisson(y[:,8]).T@capped_poisson(y[:,0])).ravel()
        h/=ROWS//2;p/=ROWS//2
        assert np.max(abs(h.sum(1)-1))<1e-10
        # 3 estimands, 2 one-sided statements each; simultaneous coverage >=99%.
        alpha=.01/6
        row=dict(full_click=interval(h,opt,alpha))
        if route:
            assert np.max(abs(p.sum(1)-1))<1e-10
            row['two_channel_pnr']=interval(p,pn,alpha)
        arrays[f'{route}_click']=h
        if route:arrays['1_pnr_pair']=p
        stats.append(row)
        print(row,flush=True)
    full=factor(t,s)
    _,red,_=project(t,s,1)
    pinfo=json.loads((OUT/'projection.json').read_bytes())
    errs=[]
    for panel in pinfo['panels'][:2]:
        banks=np.array(panel['banks']);ports=(8*banks[:,None]+np.arange(8)).ravel()
        for inputs,axes,key in [(full,np.r_[ports,ports+1152],'optical_fixed_N_vacuum'),
                              (red,np.r_[banks,banks+144],'projected_fixed_N_vacuum')]:
            errs.append(float(max(abs(photon_sector(inputs,axes)-panel[key]))))
    assert max(errs)<1e-10
    np.savez_compressed(OUT/'conditional_means.npz',**arrays)
    result=dict(trajectories_per_route=ROWS,nominal_simultaneous_coverage=.99,
                conditional_on_fixed_training=True,stats=stats,independent_fixed_N_max_abs=max(errs))
    (OUT/'certificates.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
