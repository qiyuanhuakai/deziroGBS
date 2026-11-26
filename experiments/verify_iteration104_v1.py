"""Recompute published confidence, portable native traces, selected metrics and the rank-reduction limits."""
import argparse
import json
from pathlib import Path
import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,ORDER,HIST,save,manifest,H
from iteration103_sample_v1 import DATA
from iteration104_spectral_v1 import Allocation,OUT as SPECTRAL,count_law


def philox(seed,sector,purpose,row,position):
    mask=2**32-1
    ctr=[(position//2)&mask,(position//2)>>32,row,sector*65536+purpose]
    key=[seed&mask,seed>>32]
    for _ in range(10):
        x=0xd2511f53*ctr[0];y=0xcd9e8d57*ctr[2]
        ctr=[(y>>32)^ctr[1]^key[0],y&mask,(x>>32)^ctr[3]^key[1],x&mask]
        key=[(key[0]+0x9e3779b9)&mask,(key[1]+0xbb67ae85)&mask]
    i=2*(position%2)
    return ctr[i]+2**32*ctr[i+1]


def main(out):
    out.mkdir(exist_ok=False,parents=True)
    names=('optical','pilot','fit','validate','packed','audit','contour','spectral','transfer')
    stages={}
    hashes={}
    for name in names:
        stage=DATA/f'iteration104_{name}_v1'
        verified=json.loads((stage/'manifest.json').read_text())
        assert set(verified)=={f.name for f in stage.iterdir() if f.name!='manifest.json'}
        assert all(H(stage/f)==v for f,v in verified.items())
        stages[name]=json.loads((stage/'summary.json').read_text())
        hashes[name]=H(stage/'manifest.json')
    native=SPECTRAL/'allocation.dll'
    z=np.load(SPECTRAL/'input16.npz')
    model=Allocation(native,z['intensity'],z['squeezing'],z['efficiency'])
    records=model.draw(128,True,104080001,17,4)
    def pick(u,bits,cuts,alt):
        bits=int(bits)
        i=u%2**bits
        return i if u>>bits<int(cuts[i]) else int(alt[i])
    for i,x in enumerate(records):
        ref=np.zeros_like(x)
        for s in range(model.s):
            start=model.starts[s];base=s*2**model.port_bits
            n=pick(philox(104080001,s,199,17+i,0),model.nbits[s],model.tc[start:],model.ta[start:])
            for k in range(n):
                j=pick(philox(104080001,s,200,17+i,k),model.port_bits,model.pc[base:],model.pa[base:])
                ref[j]+=1
        assert np.array_equal(x,ref)
    for efficiencies,verify in [([0.,0.],'vacuum'),([1.,1.],'even')]:
        sparse=np.zeros((2,3));sparse[:,2]=1
        a=Allocation(native,sparse,[1.65,1.31],efficiencies)
        x=a.draw(4096,True,104080002)
        assert not x[:,:2].any()
        assert not x.any() if verify=='vacuum' else not (x[:,2]%2).any()
    original=np.load(ORDER)
    r=np.repeat(original['squeezing'].T,2,axis=1)
    k=np.argmax(r,axis=1)
    t=original['T']
    opt=np.load(DATA/'iteration104_optical_v1/moments.npz')
    true=opt['prob']
    estim=np.ones(1)
    with mp.workdps(65):
        for s,j in enumerate(k):
            b=(t[:,j//2*2]+(1 if j%2==0 else -1)*t[:,j//2*2+1])/np.sqrt(2)
            eff=float(np.vdot(b,b).real)
            p,_=count_law(mp.mpf(float(r[s,j])),mp.mpf(eff))
            estim=np.convolve(estim,np.asarray(p,float))
    lim=min(len(estim),len(true))
    df=estim.cumsum()[:lim]-true.cumsum()[:lim]
    witness_i=int(df.argmax())
    nn=np.flatnonzero(true>1e-6)
    mu=opt['click_mass'][nn]/true[nn]
    var=(opt['factorial_mass'][nn]+opt['click_mass'][nn])/true[nn]-mu**2
    # C cannot exceed N, and conditional Cantelli bounds P(C<=k | N).
    witnesses=[]
    for threshold in range(lim):
        opt_upper=1-true[nn].sum()+true[nn]@(var/(var+np.maximum(mu-threshold,0.)**2))
        witnesses.append(estim.cumsum()[threshold]-opt_upper)
    raw104=stages['validate']
    flow=stages['packed']
    fit=stages['fit']
    audit=stages['audit']
    jz={}
    hw=np.load(DATA/'iteration104_fit_v1/witness.npz')['test']
    hardware=np.r_[hw[:319+1],hw[320:].sum()]/hw.sum()
    for row in raw104['rows']:
        label=row['model'];pnr=row['pnr']
        h=np.load(DATA/f'iteration104_validate_v1/hold_{label}_{int(pnr)}.npz')['joint']
        n=row['records'];assert h.sum()==n
        hc=h.sum(0)/n;hp=h.sum(1)/n
        optical=np.r_[true[:320],true[320:].sum()]
        assert max(hp[-1],optical[-1])<row['photon_ks']
        assert max(hc[-1],hardware[-1])<row['hardware_holdout_ks']
        assert abs(abs(hp.cumsum()-optical.cumsum()).max()-row['photon_ks'])<1e-13
        assert abs(abs(hc.cumsum()-hardware.cumsum()).max()-row['hardware_holdout_ks'])<1e-13
        timing=next(r for r in flow['timing'] if (r['label'],r['pnr'],r['batch'])==(label,pnr,16384))
        assert np.median(timing['us'])==timing['median_us']
        measured=next(r for r in audit['rows'] if (r['label'],r['pnr'])==(label,pnr))
        matched=next(r for r in audit['observed_speed'] if (r['label'],r['pnr'])==(label,pnr))
        jz[label+'_'+('pnr' if pnr else 'click')]=dict(records=n,us=timing['median_us'],quartiles=timing['quartiles'],rate_ratio_to_jiuzhang=1.27/timing['median_us'],hardware_click_ks=row['hardware_holdout_ks'],photon_ks=row['photon_ks'],optical_click_ks_witness_lower99=row['optical_witness_lower99'],max_spatial_survival_gap=row['all_spatial_survival_max'],weighted_conditional_mean_rmse=row['weighted_conditional_click_mean_rmse'],complete_16bit_36_107=row['panels'][0],port_pair_289_861=row.get('two_port_fock'),projected_task_us=matched['output_us'],hardware_three_8bit_empirical_tv=measured['hw_tv'])
    comp=stages['spectral']['comparison']
    prev=json.loads((DATA/'iteration103_release_v1/validation.json').read_text())
    small=[{k:r[k] for k in ('m','pnr','workers','batch','us','full_tv_upper')} for r in stages['spectral']['rows']]
    data=dict(input='Jiuzhang 3.0 high-power nominal double-spectrum covariance; 144 spatial modes and 1,152 detector bins',hardware_us=1.27,gpu=flow['device'],batch=16384,old_routes=prev['routes'],routes=jz,hardware_holdout_records=int(hw.sum()),hardware_ks_radius99=raw104['hardware_radius99'],photon_ks_radius99=raw104['photon_radius99'],spatial_radius99=raw104['spatial_radius99'],witness=fit['cdf_witness'],independent_optical_contour=stages['contour'],optical_mean_clicks=stages['optical']['mean_clicks'],optical_click_variance=stages['optical']['click_variance'],acceptance={label:fit[label]['rate'] for label in ('optical','hardware')},replay=flow['record_equivalence'],previous_16bit_frugal=prev['click_baselines'][0],previous_pnr_pe=prev['pnr_baseline'],arbitrary_width=small,arbitrary_width_comparison=comp,near_rank_one=stages['transfer']['rows'],jiuzhang_rank_reduction=dict(**stages['transfer']['jiuzhang_rank_reduction'],full_pnr_tv_lower_from_total=float(df.max()-2e-12),photon_threshold=witness_i,full_click_tv_lower_from_total=float(max(witnesses)-2e-12),click_threshold=int(np.argmax(witnesses))),native_reference_records=len(records),manifests=hashes)
    save(out,'validation.json',data)
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):main(p.parse_args().output.resolve())
