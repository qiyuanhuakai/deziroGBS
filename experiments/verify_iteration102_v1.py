"""New full-device records, complete-pair audits, held-out 16-bit events, and baseline comparisons."""
import argparse
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import beta
from threadpoolctl import threadpool_limits

from iteration100_spatial_v1 import H, HIST, ORDER, Tree, save, manifest
from iteration100_baselines_v1 import frugal
from iteration100_projection_v1 import Projector,physical
from iteration102_projection_v1 import PROJ
from iteration102_forest_v1 import FIT, RUN, BASE, DATA, VAC, Tables
from iteration102_triads_v1 import TRI
from iteration102_four_v1 import FOUR
from iteration102_sample_v1 import Sampler,BLOCK


def validate(out):
    out.mkdir(parents=True,exist_ok=False)
    for folder in (FIT,TRI,FOUR,RUN,PROJ):
        for f,digest in json.loads((folder/'manifest.json').read_text()).items():
            assert H(folder/f)==digest,(folder,f)
    t=Tables(FIT/'prepared.npz')
    proj=json.loads((PROJ/'summary.json').read_text())
    scan=np.load(VAC/'scan.npz')
    a,b=scan['a'],scan['b']
    target=scan['optical']
    pilot=sum(np.load(RUN/f'hist_4_1_{j}.npz')['vacuum_counts'] for j in (0,1))/524288
    worst=int(abs(pilot-target).argmax())
    pair=[int(a[worst]),int(b[worst])]
    refs=[]
    for i in range(5):
        with np.load(DATA/f'iteration100_projection_v1/panel_{i}.npz') as f:
            refs.append((f['ports'],f['truth']))
    g,h=pair
    ref=np.load(PROJ/f'pair_{g}_{h}.npz')
    refs.append((np.r_[8*g+np.arange(8),8*h+np.arange(8)],ref['truth']))
    bitweights=(1<<np.arange(16)).astype(np.uint16)
    hw=np.load(HIST)['total_click_histogram']
    photon=np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf']
    n=1048576
    model=Sampler(RUN/'forest.dll',depth=4)
    full=Projector(Tree(None,dict(np.load(BASE/'prepared.npz'))))
    # Choose the partition size using the optical distribution alone. With
    # one extra tail category, multinomial TV concentration costs 2^(k+1).
    jtruth=ref['reference'].clip(0)
    rank=np.argsort(-jtruth.ravel())
    tail=1-np.cumsum(jtruth.ravel()[rank])
    ks=np.arange(8,min(500,len(rank)))
    radius=np.sqrt(((ks+1)*np.log(2)+np.log(1/.0002))/(2*n))
    k=int(ks[np.argmin(radius+tail[ks-1])])
    eps=float(np.sqrt(((k+1)*np.log(2)+np.log(1/.0002))/(2*n)))
    rows=[]
    for pnr in (False,True):
        train=[np.zeros_like(q,dtype=np.int64) for _,q in refs]
        hold=[v.copy() for v in train]
        counts=np.zeros_like(jtruth,dtype=np.int64)
        both=np.zeros((len(t.order),len(t.order)))
        hc=np.zeros(len(hw),np.int64)
        hp=np.zeros(4096,np.int64)
        for pilot_phase,records,seed,hists in [(True,131072,102040000+pnr,train),(False,n,102050000+pnr,hold)]:
            for offset in range(0,records,BLOCK):
                x,ns=model.sample(BLOCK,pnr,True,16,seed,offset)
                if pnr:assert np.array_equal(x.reshape(BLOCK,-1,8).sum(2),ns)
                clicked=x>0
                for j,(ports,_) in enumerate(refs):
                    word=clicked[:,ports]@bitweights[:len(ports)]
                    hists[j]+=np.bincount(word,minlength=len(hists[j]))
                if pilot_phase:continue
                hc+=np.bincount(clicked.sum(1),minlength=len(hc))
                hp+=np.bincount(ns.sum(1).astype(np.int64),minlength=len(hp))
                zero=(ns==0).astype(float)
                both+=zero.T@zero
                code=ns[:,g].astype(np.int32)*counts.shape[1]+ns[:,h]
                counts+=np.bincount(code,minlength=counts.size).reshape(counts.shape)
        events=[]
        for j,((ports,truth),trainh,holdh) in enumerate(zip(refs,train,hold)):
            event=trainh/trainh.sum()>truth
            hits=int(holdh[event].sum())
            q=float(truth[event].sum())
            lo=float(beta.ppf(.0002,hits,n-hits+1))
            events.append(dict(ports=ports.tolist(),hits=hits,optical=q,signed_difference=hits/n-q,lower99=max(0.,lo-q)))
            np.savez_compressed(out/f'panel_{int(pnr)}_{j}.npz',pilot=trainh,hold=holdh)
        phat=counts/n
        tv=abs(phat-jtruth).sum()/2
        ctv=abs(full.mask[h].T@phat.T@full.mask[g]-ref['truth'].reshape(256,256)).sum()/2
        tailhat=float(1-phat.ravel()[rank[:k]].sum())
        allowance=eps+tailhat
        probs=both[a,b]/n
        worstgap=float(abs(probs-target).max())
        # 2*10,296 vacuum estimates jointly cost 0.003; 5 CDFs cost
        # 0.002; 12 fixed events and 2 categorical TV bounds cost 0.0028.
        ve=float(np.sqrt(np.log(2*2*len(a)/.003)/(2*n)))
        hw_radius=float(np.sqrt(np.log(2*5/.002)/(2*n))+np.sqrt(np.log(2*5/.002)/(2*hw.sum())))
        ph_radius=float(np.sqrt(np.log(2*5/.002)/(2*n)))
        rows.append(dict(pnr=pnr,n=n,seed=102050000+pnr,heldout=events,selected_pair=pair,
            max_vacuum_gap=worstgap,all_pair_99_interval=[max(0.,worstgap-ve),worstgap+ve],
            hardware_ks=float(abs(hc.cumsum()/n-hw.cumsum()/hw.sum()).max()),hardware_radius99=hw_radius,
            photon_ks=float(abs(hp.cumsum()/n-np.pad(photon,(0,len(hp)-len(photon))).cumsum()).max()),photon_radius99=ph_radius,
            selected_click_tv99=[max(0.,ctv-allowance),ctv+allowance],
            selected_count_tv99=[max(0.,tv-allowance),tv+allowance],partition_size=k+1,tail_mass=tailhat))
        np.savez_compressed(out/f'hold_{int(pnr)}.npz',vacuum_counts=both[a,b].astype(np.int64),click=hc,photon=hp,pair_counts=counts)
    bench=json.loads((RUN/'summary.json').read_text())
    numeric={}
    for typ in ('edge','triad','four'):
        group=[r['depths']['4'] for r in proj['rows'] if r['kind']==typ]
        numeric[typ]=dict(number=len(group),max_count_tv=max(r['count_tv'] for r in group),
            max_click_tv=max(r['click_tv'] for r in group),max_vacuum_error=max(abs(r['vacuum_error']) for r in group))
    diagnostic=[r for r in proj['rows'] if r['groups'] in ([34,35],[70,73],[35,69],[11,44],[57,63],pair)]
    save(out,'validation.json',dict(verified_manifests=5,heldout=rows,complete_panels=numeric,diagnostic=diagnostic,
        archived_benchmark=bench['rows'],confidence='All intervals in this audit have simultaneous nominal coverage at least 99% under independent records.'))
    manifest(out)


def benchmark(out):
    out.mkdir(parents=True,exist_ok=False)
    t=Tables(FIT/'prepared.npz')
    spec=np.load(ORDER)
    rows=[]
    for j,(a,b) in enumerate([(34,35),(73,70),(11,44)]):
        g,h=sorted((a,b))
        table=np.load(PROJ/f'pair_{g}_{h}.npz')
        law=table['counts4'].clip(0)
        truth=table['truth']
        if a!=g:
            law=law.T.copy()
            truth=truth.reshape(256,256).T.ravel()
        start=perf_counter()
        sampler=Sampler(RUN/'forest.dll',[a,b],law)
        sampler.sample(2048,False,False,1,102060000+j)
        cold=(perf_counter()-start)*1e6/2048
        ports=np.r_[8*a+np.arange(8),8*b+np.arange(8)]
        q=physical(spec,ports)
        times=[]
        for i in range(8):
            start=perf_counter()
            sampler.sample(2048,False,False,1,102061000+j,i*2048)
            if i:times.append((perf_counter()-start)*1e6/2048)
        fr=ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'))
        baselines=[]
        for k in (2,4):
            dist,us=frugal(fr,q,k,102062000+1000*j+100*k)
            baselines.append(dict(k=k,tv=float(abs(dist-truth).sum()/2),median_us=float(np.median(us)),us=us))
        rows.append(dict(groups=[a,b],us=times,median_us=float(np.median(times)),cold_us=cold,
            click_tv=float(abs(table['bits4']-table['truth']).sum()/2),frugal=baselines))
    save(out,'summary.json',dict(rows=rows,comparison='prepared two-group marginal versus Frugal on the same 16 observed detectors and two spectra; single thread, 2048 records per call'))
    manifest(out)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=('validate','baseline'))
    ap.add_argument('output',type=Path)
    args=ap.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        (validate if args.action=='validate' else benchmark)(args.output.resolve())
