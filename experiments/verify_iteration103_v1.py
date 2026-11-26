"""Recompute published scalar checks from fixed evidence, and verify every stage manifest."""
import argparse
import json
from pathlib import Path
import numpy as np
from iteration100_spatial_v1 import H,HIST,ROOT,manifest,save
from iteration103_sample_v1 import DATA


def validate(out):
    out.mkdir(parents=True,exist_ok=False)
    summaries={}
    hashes={}
    for name in ('fit','census','tail','sample','gpu','transport_gpu','projection','baselines','holdout','frugal144'):
        directory=DATA/f'iteration103_{name}_v1'
        entries=json.loads((directory/'manifest.json').read_text())
        assert set(entries)=={x.name for x in directory.iterdir() if x.name!='manifest.json'}
        for f,digest in entries.items():assert H(directory/f)==digest,(name,f)
        summaries[name]=json.loads((directory/'summary.json').read_text())
        for f,digest in summaries[name].get('hashes',{}).items():assert H(ROOT/f)==digest
        hashes[name]=H(directory/'manifest.json')
    s=summaries
    device=s['transport_gpu']
    hardware=np.load(HIST)['total_click_histogram']
    ref=np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf']
    route=[]
    for row in device['rows']:
        pnr=row['pnr']
        h=np.load(DATA/f'iteration103_transport_gpu_v1/hold_{int(pnr)}.npz')
        n=row['records']
        assert n==h['photon'].sum()==h['click'].sum()==1048576
        click_ks=float(abs(h['click'].cumsum()/n-hardware.cumsum()/hardware.sum()).max())
        photon_ks=float(abs(h['photon'].cumsum()/n-np.pad(ref,(0,len(h['photon'])-len(ref))).cumsum()).max())
        assert abs(click_ks-row['hardware_ks'])<1e-13 and abs(photon_ks-row['photon_ks'])<1e-13
        timing=next(t for t in device['timing'] if t['patch'] and not t['wide'] and t['pnr']==pnr)
        assert np.median(timing['us'])==timing['median_us']
        route.append(dict(pnr=pnr,records=n,us=timing['median_us'],quartiles=timing['quartiles'],relative_to_hardware_rate=1.27/timing['median_us'],hardware_click_ks=click_ks,optical_photon_ks=photon_ks,heldout_optical_click_tv_lower99=max(r['lower99'] for r in row['heldout']),empirical_all_survival_gap=row['all_survival_events_max']))
    rows=s['projection']['rows']
    matched=[r for r in rows if 'patched' in r]
    assert len(matched)==72 and len({i for r in matched for i in r['groups']})==144
    old=[r['retained'] for r in rows if r.get('previous_kind') in ('edge','triad')]
    bridges=[r['bridge'] for r in rows if 'bridge' in r]
    assert len(old)==93 and len(bridges)==36
    disputed=next(r for r in rows if r['groups']==[36,107])
    previous=json.loads((DATA/'iteration102_projection_v1/summary.json').read_text())['rows']
    before=next(r for r in previous if r['groups']==[36,107])['depths']['4']
    residue=max(rows,key=lambda r:r['patched_click_interval'][0])
    baselines=s['baselines']
    compact=dict(scope='Jiuzhang 3.0 double-spectrum high-power input, 144 spatial groups x 8 threshold-detector bins. Two-group TV values use the numerical optical reference. Hardware KS uses the measured total-click histogram.',device=device['device'],batch=device['batch'],hardware_us=1.27,hardware_records=int(hardware.sum()),routes=route,hardware_ks_radius99=device['hardware_radius99'],optical_photon_ks_radius99=device['photon_radius99'],first_2048_build_and_draw_s=device['cold_first_batch_us']*2048/1e6,fp_mismatches=device['fp_mismatches'],fp32_vs_fp64_tv_upper99=device['fp_coupling99'],pairs=s['fit']['pairs'],three_bin_max_tv=s['fit']['max_count3_tv'],raw_full_count_tv_percentiles=s['census']['percentiles'],patched_pairs=72,patched_count_max_tv=max(r['patched']['count_tv'] for r in matched),patched_click_max_tv=max(r['patched']['click_tv'] for r in matched),upper_any_two_group_count_tv=s['tail']['full_count_pair_numerical_upper_max'],click_lower_witness=dict(groups=residue['groups'],tv_interval=residue['patched_click_interval']),previous_pair_36_107=before,current_pair_36_107=disputed,linked=dict(previous_pairs=len(old),previous_max_click_tv=max(r['click_tv'] for r in old),bridges=len(bridges),bridges_max_click_tv=max(r['click_tv'] for r in bridges),independent_holdout=s['holdout']),click_baselines=[dict(groups=r['groups'],patched=r['patched'],us=r['us'],tv=r['complete_tv'],frugal=[dict(K=v['K'],us=v['us'],tv=v['complete_tv']) for v in r['frugal']]) for r in baselines['click']],pnr_baseline=baselines['pnr'][0],frugal144={k:v for k,v in s['frugal144'].items() if k not in ('gpu_times','times','frugal_phases')},manifests=hashes)
    save(out,'validation.json',compact)
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    validate(p.parse_args().output.resolve())
