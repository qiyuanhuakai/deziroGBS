"""Rebuild the release summary, check the Gaussian generating function against Fock permanents, and verify the pair census."""
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import H,save,manifest
from iteration106_groups_v1 import DATA,forest,synthetic
from iteration106_reference_v1 import Marked,OUT as REF

STAGES=['build','reference','weak','sample','field','packed','device','width','rankwidth','frugal','words','witness','figure']


def rd(path):return json.loads(path.read_text())


def permanent(a):
    return sum(np.prod(a[np.arange(len(a)),idx]) for idx in itertools.permutations(range(len(a))))


def fock_test():
    t,r,_=synthetic(4)
    nodes=forest(2,[[0,1]],np.ones((2,2)),{2:2})
    full=Marked(REF/'laws.dll',t,r,nodes);base=Marked(REF/'laws.dll',t,r,nodes[:2])
    ids=[0,2,3];z=np.array([[0.,0.,1.],[.4+.13j,-.1j,.8],[.9,0.,.12]])
    observed=full.evaluate(ids,z)-base.evaluate(ids,z)
    transfer=np.concatenate(t,axis=1)[ids];lam=np.tanh(r);vac=np.prod(1-lam**2)
    predicted=[]
    for zz in z:
        gram=np.eye(4)+transfer.conj().T@((zz-1)[:,None]*transfer)
        result=0j
        for k in (1,2):
            for left in range(k+1):
                a=(left,k-left)
                occ=np.repeat(np.arange(4),np.repeat(a,2))
                norm=math.prod(math.factorial(i) for i in a)
                labelled=1.
                for h,count in enumerate(a):
                    ii=np.repeat([2*h,2*h+1],count)
                    if count:labelled*=permanent(gram[np.ix_(ii,ii)])/math.factorial(count)**2
                result-=np.prod(lam**(2*np.array(a)))*labelled
                for right in range(k+1):
                    b=(right,k-right);occ2=np.repeat(np.arange(4),np.repeat(b,2))
                    norm2=math.prod(math.factorial(i) for i in b)
                    result+=np.prod(lam**(np.array(a)+b))*permanent(gram[np.ix_(occ,occ2)])/(norm*norm2)
        predicted.append(result*vac)
    err=float(np.max(abs(observed-predicted)));assert err<1e-12
    return err


def main(out):
    summaries={};manifests={}
    for code in STAGES:
        d=DATA/f'iteration106_{code}_v1'
        manifest_path=d/'manifest.json'
        for name,h in rd(manifest_path).items():assert H(d/name)==h,(code,name)
        manifests[code]=H(manifest_path)
        if (d/'summary.json').exists():summaries[code]=rd(d/'summary.json')
    st=summaries;pack=st['packed'];device=st['device'];words=st['words']
    d=DATA/'iteration106_packed_v1'
    values=np.load(d/'census_1.npz')['two_bit_tv'];m=1152
    assert len(values)==m*(m-1)//2
    check=0.;tested=0
    for i,panel in enumerate(pack['panels']):
        x=np.load(d/f'panel_{i}.npz');words_idx=np.arange(len(x['target']))
        gap=x['1']-x['target'];panel=np.array(panel)
        for lo,hi in itertools.combinations(range(len(panel)),2):
            labels=((words_idx>>lo)&1)+2*((words_idx>>hi)&1)
            tv=float(abs(np.bincount(labels,weights=gap,minlength=4)).sum()/2)
            a,b=sorted(panel[[lo,hi]])
            check=max(check,abs(tv-values[b*(b-1)//2+a]));tested+=1
    assert check<1e-10
    assert all(r['dense_packed_different_cells']<20 for r in device['optical'])
    assert all(r['n']==1048576 and max(map(abs,r['panel_gaps']))<.004 for r in device['runs'])
    assert words['high_rate_max']>255 and st['width']['high_address_discordant']>300
    assert max(x['total_alias'] for x in pack['rows'])<1e-12
    p=next(r for r in pack['rows'] if r['rank']==1)
    gpu=rd(DATA/'iteration104_device_release_v1/validation.json')
    cpu=rd(DATA/'iteration105_release_v1/validation.json')['full_1152']
    hardware_us=cpu['hardware_us'];perf=[]
    for r in words['rows']:
        perf.append({**{k:r[k] for k in ('label','pnr','packed','batch','us','quartiles')},'rate_vs_hardware':hardware_us/r['us']})
    q=st['rankwidth']['rows'][-1]
    assert q['m']==131072 and q['ideal_arithmetic_full_bound']<.00036
    limits=dict(worst_examined_click_tv=max(p['click_tvs']),full_click_and_pnr_tv_numeric_lower=max(p['click_tvs']),
                same_gpu_8= gpu['rows'],same_gpu_widths=device['external_gpu'],
                full_frugal_cpu_draw_us=cpu['frugal_K2']['draw_us'],full_frugal_cpu_total_us=cpu['frugal_K2']['total_us'],full_frugal_cpu_prefix_tv=cpu['prefix']['full_target_prefix_tv'])
    out.mkdir(parents=True,exist_ok=False)
    save(out,'validation.json',dict(hardware_us=hardware_us,device=gpu['device'],profile=pack['profiles']['1'],
        spectrum_transfer_fraction=pack['redistributed_weak_fraction'],minimum_record_bits_click=1152,
        full_optical=dict(photon_total_tv=p['total_tv'],pairwise_census=dict(pairs=p['pairs'],two_bit_max_tv=p['max_tv'],two_bit_mean_tv=p['mean_tv'],two_bit_p99_tv=p['p99_tv']),panel_tvs=p['click_tvs'],panels=pack['panels']),
        speed=perf,fast_holdout=words['runs'],paired_model_audit=device['optical'],native_host_holdout=device['runs'],
        hardware_calibration=device['fit'],holdout_records=device['hardware_holdout_records'],hardware_ks_radius=device['two_sample_ks_radius'],
        full_pnr_pair=device['pnr'],native_event_validation=st['witness'],limits=limits,frugal_projected=st['frugal']['rows'],pe=st['sample']['pair']['photonic_engines'],
        fully_quantum_sectors=dict(native_weak_tv=st['weak']['click_tv'],weak_full_fallback_mass=st['weak']['full_output_fallback_mass'],
        weak_8_source_two_photon_records=st['weak']['eight_source_two_photon_records'],weak_frugal=st['weak']['frugal'],weak_pe_us=st['weak']['pe_us'],
        high_power_panels=st['reference']['rows'][1:],high_power_total=st['reference']['total']),
        width_q=st['rankwidth'],width_field=st['width'],overflow_max=words['high_rate_max'],conditional_click_rounding_tv=words['conditional_threshold_uniform_full_tv_bound'],
        conditional_permanent_error=fock_test(),census_projection_max_error=check,census_projected_pairs=tested,manifests=manifests))
    manifest(out)
    print('Independent Fock amplitudes and',tested,'census projections verified; worst census discrepancy',check)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):main(p.parse_args().output.resolve())
