"""Verify published raw files, independent few-photon amplitudes, numerical projections and timings."""
import argparse
import ctypes as ct
import json
import math
from pathlib import Path

import numpy as np
from iteration100_spatial_v1 import ROOT,H,save,manifest,bind,I,D,W
from iteration107_cliques_v1 import optical_x
from iteration107_fast_v1 import oracle
from iteration106_sample_v1 import DATA,read

STAGES=('tables','device','holdout','cliques','rank','projection','patterns','readout','resolved_law','fast','scale','cross','global','peers')


def few_photons():
    oracle()  # Registers the GCC runtime path on Windows.
    fn=bind(ct.CDLL(str(DATA/'iteration107_patterns_v1/fock.dll')),'patterns107',[I,I,I,D,W,D])
    states=[(a,b) for a in range(6) for b in range(6-a)]
    codes=np.array([a+16*b for a,b in states],np.uint64)
    max_error=0.
    for theta in (0.,.19,np.pi/4,1.09):
        c,s=np.cos(theta),np.sin(theta)
        u=np.array([[c,1j*s],[1j*s,c]])@np.diag([np.exp(.41j),np.exp(-.29j)])
        r=.73;lam=np.tanh(r)
        x=optical_x(u,np.array([[r]]))[0];out=np.zeros(len(states))
        assert fn(2,len(codes),2,np.ascontiguousarray(np.eye(4)+x),codes,out)==0
        for (a,b),value in zip(states,out):
            target=0.
            if (a+b)%2==0:
                n=(a+b)//2
                amplitude=sum(math.comb(n,j)*math.comb(n,a-j)*u[0,0]**j*u[1,0]**(n-j)*u[0,1]**(a-j)*u[1,1]**(n-a+j)
                              for j in range(max(0,a-n),min(n,a)+1))
                target=(1-lam**2)*lam**(2*n)*abs(amplitude)**2*math.factorial(a)*math.factorial(b)/math.factorial(n)**2
            max_error=max(max_error,abs(value-target))
    assert max_error<1e-12
    return float(max_error)


def main(out):
    summaries={};hashes={}
    for s in STAGES:
        p=DATA/f'iteration107_{s}_v1'
        lst=read(p/'manifest.json');assert set(lst)=={f.name for f in p.iterdir() if f.name!='manifest.json'}
        for name,h in lst.items():assert H(p/name)==h,(s,name)
        hashes[s]=H(p/'manifest.json')
        if (p/'summary.json').exists():summaries[s]=read(p/'summary.json')
    for stage,source in [('device','local_transport'),('cliques','clique_transport'),('readout','photon_readout'),('fast','fast_cliques')]:
        p=DATA/f'iteration107_{stage}_v1'
        assert H(ROOT/f'native/{source}107_v1.cu')==read(p/'build.json')['source_sha256']
    s=summaries;pr=s['patterns']['rows'];tab=s['tables']['rows']
    assert len(pr)==len(tab)==72 and sorted(k for v in pr for k in v['pair'])==list(range(144))
    assert all(x['pair']==y['pair'] for x,y in zip(pr,tab))
    assert s['readout']['layout_cases']==20349 and max(x['max_sector_reference_discrepancy'] for x in pr)<1e-12
    assert s['scale']['correction_only_overflow']==299
    # A frozen optical input, its prefix and three independent remote projections.
    with np.load(DATA/'iteration107_fast_v1/prefix4096.npz') as z:
        tv=float(abs(z['optical']-z['model']).sum()/2)
        assert abs(tv-s['fast']['rows'][-1]['primary_corrected_tv'])<1e-14 and z['empirical'].sum()==65536
    with np.load(DATA/'iteration107_cross_v1/projection.npz') as z:
        center=float(abs(z['primary_only']-z['target']).sum()/2)
        impact=sum(v['change_probability'] for v in s['cross']['remote'])
        assert abs(center+impact-s['cross']['upper'])<1e-15
    global_rows=s['global']['rows']
    with np.load(DATA/'iteration107_global_v1/totals.npz') as z:
        for row in global_rows:
            m=row['m'];n=row['shots'];obs=z['observed_'+str(m)];ref=z['optical_'+str(m)]
            assert obs.sum()==n and abs(abs(obs.cumsum()/n-ref.cumsum()).max()-row['pnr_total_ks'])<1e-12
    with np.load(DATA/'iteration107_readout_v1/holdout.npz') as z:
        hardware=np.load(DATA/'iteration104_record_order_v1/windows.npz')['histograms'][32:].sum(0)
        for row in s['readout']['runs']:
            key=row['profile']+('_pnr' if row['pnr'] else '_click')
            counts=z[key+'_click_total'];assert counts.sum()==row['records']
            assert abs(abs(counts.cumsum()/counts.sum()-hardware.cumsum()/hardware.sum()).max()-row['hardware_ks'])<1e-12
    prev=read(DATA/'iteration106_release_v1/validation.json')
    selected=dict(device=read(DATA/'iteration107_device_v1/build.json')['device'],hardware_us=prev['hardware_us'],
        hardware_holdout_records=int(hardware.sum()),jiuzhang_speed=[{k:r[k] for k in ('profile','batch','route','us')} for r in s['readout']['timing']],
        jiuzhang_holdout=s['readout']['runs'],analytic_panels=s['projection']['rows'],resolved_panels=s['resolved_law'],
        matched_count=72,matched_16bit_direct_max_table_tv=max(r['click_table_tv_after'] for r in tab),
        matched_16bit_model_prior_max_tv=max(r['click_tv_before'] for r in tab),
        matched_pnr_threshold_max_tv_before_patterns=max(r['old_click_tv'] for r in pr),
        matched_pnr_threshold_max_tv_after_patterns=max(r['new_click_tv'] for r in pr),
        resolved_34_35=next(r for r in pr if r['pair']==[34,35]),
        five_photon_sector_mass_range=[min(r['sector_mass'] for r in pr),max(r['sector_mass'] for r in pr)],
        fock_reference_max_error=few_photons(),full_frugal_cpu_us=prev['limits']['full_frugal_cpu_draw_us'],
        full_frugal_cpu_prefix_tv=prev['limits']['full_frugal_cpu_prefix_tv'],
        jiuzhang_frugal16=[v for v in prev['frugal_projected'] if v['panel'] in (0,5)],
        benchmark=s['peers'],small_tiles_4096=s['rank']['rows'][-1],frugal16_by_width=s['rank']['frugal'],
        direct_tile_scaling=s['fast']['rows'],large_width=s['scale']['rows'],cross_tile_bound=s['cross'],global_observables=global_rows,
        verified_manifests=hashes,previous_quantum_two_sources_131072=prev['width_q']['rows'][-1],
        interpretation='TV and KS are fractions. Analytic TV evaluates projected model laws. Empirical TV includes finite-shot error. Hardware-calibrated and optical profiles have separate targets.')
    out.mkdir(parents=True,exist_ok=False);save(out,'validation.json',selected);manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output.resolve())
