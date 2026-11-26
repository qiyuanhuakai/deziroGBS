"""Fresh fixed-size output batches, whole-machine holdouts and same-projection precision intervals."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import HIST,save,manifest
from iteration103_sample_v1 import DATA
from iteration104_optical_v1 import OPT
from iteration104_sample_v1 import Filter,PILOT,FIT,MAX


def interval(truth,counts,trial,n,alpha):
    # Complete TV lies above the gap of a frozen event. For an upper bound
    # retain top optical words and merge the tail, then add its reference mass.
    order=np.argsort(-truth)
    tail=np.maximum(0.,1-np.cumsum(truth[order]))
    k=np.arange(1,len(truth)+1)
    radii=np.sqrt(((k+1)*np.log(2)+np.log(1/alpha))/(2*n))
    kk=int(np.argmin(tail+radii))
    ids=order[:kk+1]
    diff=counts[ids]/n-truth[ids]
    collapsed=(abs(diff).sum()+abs(diff.sum()))/2
    event=trial/trial.sum()>truth
    gap=float(counts[event].sum()/n-truth[event].sum())
    # The empirical complete TV is upward-biased by convexity. Replacing
    # one draw changes it by at most 1/n; bounded differences give a second,
    # dimension-free one-sided upper bound on the population TV.
    plug=float(abs(counts/n-truth).sum()/2)
    margin=np.sqrt(np.log(1/alpha)/(2*n))
    return dict(lower=max(0.,gap-margin),upper=min(1.,collapsed+tail[kk]+radii[kk],plug+margin),empirical_tv=plug,event_gap=gap,retained=len(ids),tail=float(tail[kk]),radius=float(radii[kk]))


def main(out):
    out.mkdir(exist_ok=False,parents=True)
    model=Filter(PILOT/'gpu.dll')
    tables=dict(np.load(FIT/'tables.npz'))
    witness=np.load(FIT/'witness.npz')
    hw=witness['test']
    h=witness['h']
    upper=float(witness['upper'])
    opt=np.load(OPT/'moments.npz')
    ph=np.r_[opt['prob'][:MAX-1],opt['prob'][MAX-1:].sum()]
    mu=opt['click_mass']/np.maximum(opt['prob'],1e-300)
    true_var=(opt['factorial_mass']+opt['click_mass'])/np.maximum(opt['prob'],1e-300)-mu**2
    groups=[(36,107),(34,35),(25,139),(0,1)]
    panels=[np.load(DATA/f'iteration103_projection_v1/pair_{a}_{b}.npz') for a,b in groups]
    pref=np.load(DATA/'iteration103_baselines_v1/pnr.npz')['reference']
    fit=np.load(DATA/'iteration103_fit_v1/fit.npz')
    ga,gb=fit['a'],fit['b']
    rows=[]
    times=[]
    states=[(label,pnr) for label in ('optical','hardware') for pnr in (False,True)]
    # Common domains within a route-equivalence check, distinct seeds between experiments.
    for i,(label,pnr) in enumerate(states):
        model.set_table(tables[label])
        rate=float(tables[label+'_rate'])
        x,c,end=model.draw(521,False,104010000+i,rate=rate)
        y,d,pend=model.draw(521,True,104010000+i,rate=rate)
        assert end==pend and np.array_equal(x,y>0) and np.array_equal(c,d)
    rng=np.random.default_rng(104011000)
    for rep in range(10):
        for i in rng.permutation(len(states)):
            label,pnr=states[i]
            model.set_table(tables[label])
            # Finish table upload before timing. Include each complete result on the host.
            import torch
            torch.cuda.synchronize()
            tic=perf_counter()
            x,c,end=model.draw(16384,pnr,104012000+100*rep+int(i),rate=float(tables[label+'_rate']))
            elapsed=(perf_counter()-tic)*1e6/16384
            if rep>=2:times.append(dict(label=label,pnr=pnr,us=elapsed,proposed=end))
    weights=(1<<np.arange(16)).astype(np.uint16)
    for i,(label,pnr) in enumerate(states):
        model.set_table(tables[label])
        rate=float(tables[label+'_rate'])
        train=[np.zeros(65536,np.int64) for _ in panels]
        actual=[v.copy() for v in train]
        joint=np.zeros((MAX,MAX),np.int64)
        spatial=np.zeros((288,288))
        fock=np.zeros(169,np.int64)
        n=1048576
        accepted=used=0
        for count,seed,destination in [(131072,104013000+i,train),(n,104014000+i,actual)]:
            off=0
            for start in range(0,count,8192):
                x,c,off=model.draw(8192,pnr,seed,off,rate)
                clicked=x>0
                for ids,panel,counter in zip(groups,panels,destination):
                    code=clicked[:,panel['ports']]@weights
                    counter+=np.bincount(code,minlength=65536)
                if destination is train:continue
                nc=c.sum(1,dtype=np.int32)
                cc=clicked.sum(1,dtype=np.int32)
                joint+=np.bincount(nc.clip(0,MAX-1)*MAX+cc.clip(0,MAX-1),minlength=MAX*MAX).reshape(MAX,MAX)
                marks=np.concatenate([c>=1,c>=2],axis=1).astype(float)
                spatial+=marks.T@marks
                if pnr:
                    assert np.array_equal(x.reshape(len(x),144,8).sum(2),c)
                    fock+=np.bincount(x[:,[289,861]].clip(0,12)@np.array([1,13]),minlength=169)
            if destination is actual:used=off
        observed=np.column_stack([spatial[144*a+ga,144*b+gb] for a in range(2) for b in range(2)])/n
        clicks=joint.sum(0)/n
        photons=joint.sum(1)/n
        target=np.r_[hw[:MAX-1],hw[MAX-1:].sum()]/hw.sum()
        error=abs(observed-fit['target'])
        loc=np.unravel_index(error.argmax(),error.shape)
        panel_stats=[dict(groups=list(g),**interval(p['truth'],v,t,n,.0001)) for g,p,t,v in zip(groups,panels,train,actual)]
        obs=float(np.r_[clicks[:len(h)-1],clicks[len(h)-1:].sum()]@h)
        nr=np.arange(55,191)
        pr=joint[nr]/np.maximum(1,joint[nr].sum(1))[:,None]
        cond=pr@np.arange(MAX)
        condvar=(pr@(np.arange(MAX,dtype=float)**2))-cond**2
        score=ph[nr]/ph[nr].sum()
        timing=[v['us'] for v in times if v['label']==label and v['pnr']==pnr]
        row=dict(model=label,pnr=pnr,records=n,seed=104014000+i,proposals_including_batch_overdraw=used,us=float(np.median(timing)),timing=timing,mean_clicks=float(clicks@np.arange(MAX)),hardware_holdout_ks=float(abs(clicks.cumsum()-target.cumsum()).max()),photon_ks=float(abs(photons.cumsum()-ph.cumsum()).max()),optical_witness_lower99=max(0.,obs-upper-np.sqrt(np.log(10000)/(2*n))),witness_observed=obs,weighted_conditional_click_mean_rmse=float(np.sqrt(score@(cond-mu[nr])**2)),weighted_conditional_click_variance_rmse=float(np.sqrt(score@(condvar-true_var[nr])**2)),all_spatial_survival_max=float(error[loc]),worst_spatial=[int(ga[loc[0]]),int(gb[loc[0]]),int(loc[1])],panels=panel_stats)
        if pnr:
            row['two_port_fock']=interval(pref,fock,fock*0+np.rint(pref*n),n,.0001)
            # Only the upper interval is used for this pre-specified PNR projection.
            row['two_port_fock'].pop('lower')
            row['two_port_fock'].pop('event_gap')
        rows.append(row)
        print(label,pnr,row['us'],row['hardware_holdout_ks'],row['photon_ks'],flush=True)
        np.savez_compressed(out/f'hold_{label}_{int(pnr)}.npz',joint=joint,spatial=observed,pnr=fock,**{f'trial_{j}':t for j,t in enumerate(train)},**{f'panel_{j}':v for j,v in enumerate(actual)})
    save(out,'summary.json',dict(rows=rows,batch=16384,hardware_holdout=int(hw.sum()),hardware_radius99=float(np.sqrt(np.log(2/.0001)/(2*n))+np.sqrt(np.log(2/.0001)/(2*hw.sum()))),photon_radius99=float(np.sqrt(np.log(2/.0001)/(2*n))),spatial_radius99=float(np.sqrt(np.log(2*fit['target'].size/.0001)/(2*n))),tail_bin=MAX-1,conditional_photon_range=[55,190],study='Isolated per-record rejection; all empirical intervals use fresh records with fixed rules. TV intervals cover each entire 16-bit law.'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):main(p.parse_args().output.resolve())
