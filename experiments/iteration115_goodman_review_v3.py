"""Independent 16-channel adaptation of Goodman et al. (2026), Eqs. 8--12."""
import json
from time import perf_counter
import numpy as np
from scipy.linalg import eigh
from scipy.stats import beta
from threadpoolctl import threadpool_limits
from iteration115_visible_v1 import DATA,hardware
from iteration106_field_v1 import actual_input
from iteration107_cliques_v1 import optical_x
from iteration115_panel_accuracy_v2 import tv

OUT=DATA/'iteration115_review_v3'
PORTS=np.arange(272,288)
BATCH=8192
TRAIN=131072
EVAL=524288


def fields(rng,n,t,spectra):
    intensity=np.zeros((n,len(t)),complex)
    for s in spectra:
        nbar=np.sinh(s)**2
        m=np.sinh(s)*np.cosh(s)
        x=rng.standard_normal((n,len(s)))*np.sqrt((nbar+m)/2)
        y=rng.standard_normal((n,len(s)))*np.sqrt((m-nbar)/2)
        intensity+=((x-y)@t.T)*((x+y)@t.conj().T)
    return intensity


def targets(raw,gains):
    x=optical_x(raw,gains);m=len(raw)
    def vac(indices,weights):
        axes=np.r_[indices,np.array(indices)+m]
        w=np.tile(np.sqrt(weights),2)
        sub=x[:,axes][:,:,axes]*w[:,None]*w[None,:]
        return np.exp(-np.linalg.slogdet(np.eye(len(w))+sub)[1].sum()/2)
    v=np.array([vac([i],[1]) for i in range(m)])
    second=np.array([[vac([i,j],[1,1]) if i!=j else vac([i],[2]) for i in range(m)] for j in range(m)])
    norm=(x[:,:m,:m]+x[:,m:,m:]+1j*(x[:,m:,:m]-x[:,:m,m:]))/2
    an=(x[:,:m,:m]-x[:,m:,m:]+1j*(x[:,m:,:m]+x[:,:m,m:]))/2
    rates=np.diagonal(norm,axis1=1,axis2=2).real.sum(0)
    return [(v,second-np.outer(v,v)),(rates,(abs(norm)**2+abs(an)**2).sum(0))]


def matrix_power(c,power):
    e,u=eigh((c+c.T)/2)
    keep=e>1e-13*e.max()
    return (u[:,keep]*e[keep]**power)@u[:,keep].T,e


def clip(x,route):
    return np.clip(x.real,0,1 if route==0 else np.inf)


def raw_prob(intensity,route):
    return np.exp(-intensity) if route==0 else intensity


def fit(raw,route,mean,cov):
    y=clip(raw,route)
    coloring,eig=matrix_power(cov,.5)
    layers=[]
    for _ in range(10):
        center=y.mean(0)
        whitening,_=matrix_power(np.cov(y.T,bias=True),-.5)
        mat=whitening@coloring
        layers.append((mat,mean-center@mat))
        y=clip(y@mat+layers[-1][1],route)
    return layers,dict(min_target_cov_eigenvalue=float(eig.min()),max_target_cov_eigenvalue=float(eig.max()),
        negative_target_eigenvalues=int((eig< -1e-12).sum()),
        postfit_mean_residual=float(abs(y.mean(0)-mean).max()),
        postfit_cov_residual=float(abs(np.cov(y.T,bias=True)-cov).max()),
        raw_mean_check=float(abs(raw.mean(0)-mean).max()))


def corrected(intensity,layers,route):
    y=clip(raw_prob(intensity,route),route)
    for mat,offset in layers:y=clip(y@mat+offset,route)
    return y


def evaluate(route,t,spectra,layers,optical,tilted):
    rng=np.random.default_rng(np.random.SeedSequence([115,333,route]))
    draw_rng=np.random.default_rng(np.random.SeedSequence([115,334,route]))
    counts=[];times=[];pairs=[];bank_hist=[];raw_examples=[]
    for rep in range(EVAL//BATCH+1):
        start=perf_counter()
        y=corrected(fields(rng,BATCH,t,spectra),layers,route)
        rows=(draw_rng.random(y.shape)>y) if route==0 else draw_rng.poisson(y)
        rows=rows.astype(np.uint8)
        elapsed=(perf_counter()-start)*1e6/BATCH
        if rep==0:continue
        times.append(elapsed)
        if rep==1:raw_examples=rows.copy()
        code=(rows>0)@(1<<np.arange(16))
        counts.append(np.bincount(code,minlength=65536))
        if route:
            cnt=np.minimum(rows[:,0],8).astype(int)+9*np.minimum(rows[:,8],8).astype(int)
            pairs.append(np.bincount(cnt,minlength=81))
        h=(rows>0).reshape(-1,2,8).sum(2)
        bank_hist.append(np.bincount(9*h[:,0]+h[:,1],minlength=81))
    counts=np.array(counts)
    # Select an event from half the records, measure its discrepancy on independent rows.
    half=len(counts)//2
    event=(counts[:half].sum(0)/(EVAL/2)>optical)
    hits=int(counts[half:,event].sum());n=EVAL//2
    lower=beta.ppf(.01,hits,n-hits+1)
    pop=np.array([v.bit_count() for v in range(256)])
    words=np.arange(65536);code=9*pop[words&255]+pop[words>>8]
    coarse=np.bincount(code,weights=optical,minlength=81)
    old=np.bincount(code,weights=tilted,minlength=81)
    bank_hist=np.array(bank_hist)
    stats=dict(rows=EVAL,train=TRAIN,replicate_times_us=times,median_us=float(np.median(times)),
        empirical_full_click_tv=tv(counts.sum(0)/EVAL,optical),event_size=int(event.sum()),test_records=n,
        test_hits=hits,optical_event=float(optical[event].sum()),one_sided_99_full_click_tv_lower=max(0.,float(lower-optical[event].sum())),
        bank_click_empirical_tv=tv(bank_hist.sum(0)/EVAL,coarse),round115_bank_click_model_tv=tv(old,coarse))
    if route:
        pn=np.load(DATA/'iteration115_panel_accuracy_v2/pnr_laws.npz')
        stats['two_channel_pnr_empirical_tv']=tv(np.sum(pairs,axis=0)/EVAL,pn['optical_pair'])
    return stats,dict(hist=counts,bank_hist=bank_hist,examples=raw_examples,pair_hist=np.array(pairs))


def run():
    OUT.mkdir(exist_ok=True)
    t,ss=hardware();t=t[PORTS];raw,r=actual_input();ref=targets(raw[PORTS],r)
    src=np.load(DATA/'iteration115_panel_v1/panel.npz')
    stats=[];arrays={}
    # The deterministic transformation is trained once and applied to fresh trajectories.
    rng=np.random.default_rng(np.random.SeedSequence([115,332]))
    start=perf_counter()
    ensemble=np.vstack([fields(rng,BATCH,t,ss) for _ in range(TRAIN//BATCH)])
    pre=perf_counter()-start
    for route,(mean,cov) in enumerate(ref):
        start=perf_counter()
        layers,diag=fit(raw_prob(ensemble,route),route,mean,cov)
        fit_s=perf_counter()-start
        row,output=evaluate(route,t,ss,layers,src['optical'],src['model'])
        row.update(route=['threshold','pnr'][route],ensemble_setup_s=pre,fit_s=fit_s,diagnostics=diag)
        arrays.update({f'{route}_{k}':v for k,v in output.items()})
        arrays[f'{route}_layers']=np.stack([x for x,_ in layers])
        arrays[f'{route}_offsets']=np.stack([b for _,b in layers])
        stats.append(row)
        print(route,row['median_us'],row['one_sided_99_full_click_tv_lower'],diag,flush=True)
    np.savez_compressed(OUT/'positive_p_panel.npz',**arrays)
    (OUT/'positive_p.json').write_text(json.dumps(stats,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
