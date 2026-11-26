"""Independent bivariate normal integration, detector-level PNR baselines and evidence checks."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import beta
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
from threadpoolctl import threadpool_limits

from iteration101_sample_v1 import Sampler, FIT, RUN, DATA, BASE
from iteration101_copula_v1 import features
from iteration100_spatial_v1 import Tree,H,save,manifest
from iteration100_projection_v1 import Projector


class PairProjection(Projector):
    def response(self,group,feature):
        groups=sorted(group)
        g=groups[0]
        left,ids=feature(g)
        if len(groups)==1:
            return self.totals[g]@left,ids
        h=groups[1]
        right,more=feature(h)
        coeff=features(self.totals[g])[1]
        other=features(self.totals[h])[1]
        weights=self.corr[g,h]**np.arange(coeff.shape[1])
        return ((right.T@other*weights)@(coeff.T@left)).ravel(),ids+more


def check(out):
    out.mkdir(parents=True,exist_ok=False)
    vac=DATA/'iteration101_vacuum_v1'
    for directory in (FIT,RUN,vac):
        hashes=json.loads((directory/'manifest.json').read_text())
        assert all(H(directory/f)==digest for f,digest in hashes.items())
    model=Tree(None,dict(np.load(BASE/'prepared.npz')))
    proj=PairProjection(model)
    proj.corr=np.load(FIT/'fit.npz')['corr']
    fit=json.loads((FIT/'summary.json').read_text())
    vacuum=json.loads((vac/'summary.json').read_text())
    samples=json.loads((RUN/'summary.json').read_text())
    maxdiff=0.
    pairs=[(row['groups'],FIT/f'pair_{j}.npz') for j,row in enumerate(fit['rows'])]
    pairs.append((vacuum['worst_pair'],vac/'worst.npz'))
    for (g,h),path in pairs:
        joint=np.load(path)['counts']
        rho=proj.corr[g,h]
        for i,j in ((0,0),(1,2),(3,1),(5,5),(10,8)):
            a=ndtri(proj.totals[g][:i+1].sum())
            b=ndtri(proj.totals[h][:j+1].sum())
            f=lambda x: np.exp(-x*x/2)/np.sqrt(2*np.pi)*ndtr((b-rho*x)/np.sqrt(1-rho*rho))
            val,_=quad(f,-np.inf,a,epsabs=2e-13,epsrel=2e-13)
            maxdiff=max(maxdiff,abs(val-joint[:i+1,:j+1].sum()))
    assert maxdiff<2e-12
    previous=DATA/'iteration100_baselines_v1'
    pe=json.loads((previous/'summary.json').read_text())['pnr']
    pnrs=[]
    for j,row in enumerate(pe):
        ports=np.array(row['ports'])
        groups=np.unique(ports//8)
        sampler=Sampler(RUN/'copula.dll',groups)
        idx=[int(np.where(groups==p//8)[0][0])*8+p%8 for p in ports]
        law=proj.photons(ports.tolist(),12)
        truth=np.load(previous/f'pnr_{j}.npz')['truth']
        tv=float(abs(law-truth).sum()/2)
        speeds=[]
        for i in range(8):
            start=perf_counter()
            records,_=sampler.sample(2048,True,False,1,101050000+j,2048*i)
            result=records[:,idx].copy()
            if i:speeds.append((perf_counter()-start)*1e6/len(result))
        x,_=sampler.sample(32768,True,False,1,101060000+j)
        code=x[:,idx].clip(0,12)@np.array([1,13])
        empirical=abs(np.bincount(code,minlength=169)/len(x)-truth).sum()/2
        pnrs.append(dict(ports=ports.tolist(),full_tv_interval=[tv,tv+row['reference_tail']],
            median_us=float(np.median(speeds)),us=speeds,empirical_tv=empirical,
            archived_pe_empirical=row['pe_empirical'],archived_pe_us=float(np.median(row['pe_us'])),
            records=len(x)))
    for row in samples['rows']:
        f=np.load(RUN/f"hist_{int(row['locked'])}_{int(row['pnr'])}.npz")
        assert f['click'].sum()==f['photon'].sum()==row['records']
        if not row['locked']:
            tvs=np.array([p['click_tv'] for p in fit['rows']])
            assert max(abs(tvs-row['click_event_difference']))<samples['optical_ks_radius99']
    cold=[]
    for j in range(5):
        start=perf_counter()
        reduced=Sampler(RUN/'copula.dll',fit['rows'][0]['groups'])
        reduced.sample(2048,False,False,1,101070000+j)
        cold.append((perf_counter()-start)*1e6/2048)
    # The analytic optical scan picked the pair and events before these new draws.
    g,h=vacuum['worst_pair']
    worst=np.load(vac/'worst.npz')
    click_event=worst['bits']>worst['truth']
    count_event=worst['counts']>worst['reference']
    qclick=worst['truth'][click_event].sum()
    qcount=worst['reference'][count_event].sum()
    sampler=Sampler(RUN/'copula.dll')
    powers=(1<<np.arange(8)).astype(np.uint16)
    n=262144
    fresh=[]
    for route in (False,True):
        hit=nhit=0
        for off in range(0,n,2048):
            x,_=sampler.sample(2048,route,True,16,101080000+int(route),off)
            left=x[:,8*g:8*g+8]
            right=x[:,8*h:8*h+8]
            code=((left>0)@powers)+256*((right>0)@powers)
            hit+=int(click_event[code].sum())
            if route:nhit+=int(count_event[left.sum(1),right.sum(1)].sum())
        def event(k,q):
            low=beta.ppf(.01/3,k,n-k+1)
            return dict(hits=k,optical=q,difference=k/n-q,lower99=max(0.,low-q))
        fresh.append(dict(pnr=route,records=n,click=event(hit,qclick),count=event(nhit,qcount) if route else None))
    save(out,'validation.json',dict(checked_manifests=3,independent_bivariate_integrals=len(pairs)*5,
        max_integral_error=maxdiff,full_device_records=sum(x['records'] for x in samples['rows']),
        pnr_same_ports=pnrs,fresh_locked_worst=fresh,reduced_cold_first_batch_us=cold,pe_evidence=H(previous/'summary.json'),
        covariance_unlocked=fit['fit'],vacuum_scan=vacuum,benchmark=samples))
    manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):
        check(ap.parse_args().output.resolve())
