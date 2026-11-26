"""Verify fixed raw evidence and numerical marginals, and publish a compact comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import H, realify, save, manifest
from iteration103_sample_v1 import DATA
from iteration105_jiuzhang_v1 import inputs, xpair

STAGES=('iteration105_two_mode_v1','iteration105_accuracy_v1','iteration105_jiuzhang_v1',
        'iteration105_source_groups_v1','iteration105_full_prefix_v1','iteration105_latency_v1',
        'iteration103_frugal_full_v1')


def checked(name):
    directory=DATA/name
    index=json.loads((directory/'manifest.json').read_text())
    for file,expected in index.items():
        assert H(directory/file)==expected,(name,file)
    return json.loads((directory/'summary.json').read_text())


def total_tail_bounds(groups):
    # For non-negative photon count, P(N>=L) <= E[z^N]/z^L, z>1.
    z,modes,strength=inputs()
    t=z['T'][np.argsort(z['order'])]
    basis=realify(np.linalg.qr(t,mode='reduced')[1])
    elements=[]
    for s in range(2):
        per_source=[]
        for i in range(25):
            b=basis[:,[2*i,2*i+1,50+2*i,51+2*i]]
            per_source.append(b@xpair(strength[25*s+i])@b.T)
        elements.append(np.array(per_source))
    bounds=[]
    for partition in groups:
        eig=np.concatenate([np.linalg.eigvalsh(row[g].sum(0)) for row in elements for g in partition])
        dz=np.linspace(.05,.95,128)/eig.max()
        log_bound=-.5*np.log1p(-dz[:,None]*eig).sum(1)-1024*np.log1p(dz)
        bounds.append(float(np.exp(log_bound.min())))
    return bounds


def run(out):
    width,accuracy,jz,group,prefix,latency,frugal=map(checked,STAGES)
    raw=DATA/STAGES[-1]
    with np.load(raw/'samples.npz') as z:
        spectra=z['spectra']
        assert spectra.shape==(2,4,1152) and np.isin(spectra,[0,1]).all()
        assert np.array_equal(spectra[0]|spectra[1],z['clicks'])
        counts=z['clicks'].sum(1).tolist()
    replay=np.load(DATA/STAGES[4]/'law.npz')['seed_replay']
    assert np.array_equal(replay,spectra[:,:,:16])
    n=accuracy['click_records']
    assert n==2097152 and accuracy['pnr_records']==1048576
    assert abs(accuracy['native_click']['event_gap'])<.003 and abs(accuracy['local_pnr']['event_gap'])<.004
    assert accuracy['native_click']['upper']<min(r['tv'] for r in accuracy['frugal'])
    bounds=total_tail_bounds([row['groups'] for row in group['rows']])
    assert max(bounds)<1e-10,bounds
    jrows=jz['rows']
    assert all(abs(row['mean_photon']-116.28612516734)<1e-7 for row in jrows)
    for row in jrows:
        assert row['records']==262144
        assert max(abs(panel['event_bias']) for panel in row['panel'])<.007
    a=latency['rows']
    def us(batch,workers,pnr):return next(row['median_us'] for row in a if (row['batch'],row['workers'],row['pnr'])==(batch,workers,pnr))
    peer_gpu=json.loads((DATA/'iteration104_device_release_v1/validation.json').read_text())
    previous=json.loads((DATA/'iteration104_release_v1/validation.json').read_text())
    photon=next(row for row in jrows if row['pnr'] and row['label']=='coherent_pairs')
    optical=next(row for row in jrows if not row['pnr'] and row['label']=='coherent_pairs')
    assert abs(optical['panel'][1]['full_target_tv']-group['rows'][0]['click_tvs'][-1])<1e-9
    full=dict(frugal_K2=frugal,merged_click_counts=counts,prefix=prefix,replayed_prefix_bits=int(replay.size),
              coherent_pair_cpu=dict(batch4_single_click_us=us(4,1,False),batch4_single_pnr_us=us(4,1,True),
              prepare_s=latency['prepare_s'],batch4_setup_inclusive_us=(latency['prepare_s']*1e6/4+us(4,1,False)),
              batch2048_16threads_click_us=us(2048,16,False),batch2048_16threads_pnr_us=us(2048,16,True)),
              draw_speed_ratio_single_thread=frugal['draw_us']/us(4,1,False),
              setup_inclusive_ratio_single_thread=frugal['total_us']/(latency['prepare_s']*1e6/4+us(4,1,False)),
              cpu_model_prefix_tv=optical['panel'][1]['full_target_tv'],photon_total_tv=photon['photon_total_tv_vs_full'],
              photon_total_ks=photon['photon_total_ks_vs_full'],nominal_mean=photon['mean_photon'],
              hardware_click_ks=optical['empirical_hardware_click_ks'],hardware_radius=jz['hardware_radius99'],
              hardware_records=jz['hardware_records'],hardware_us=previous['hardware_us'])
    out.mkdir(parents=True,exist_ok=False)
    save(out,'validation.json',dict(full_1152=full,synthetic_16=accuracy,
         synthetic_timing=[{k:x[k] for k in ('m','pnr','workers','batch','us')} for x in width['rows']],
         source_group_models=[dict(cap=x['cap'],method=x['method'],photon_tv=x['photon_tv'],click_tvs=x['click_tvs'],photon_alias_tail=bound) for x,bound in zip(group['rows'],bounds)],
         max_panel_reference_tv_discrepancy=group['extended_precision_max_tv_vs_binary128'],panel_ports=group['panels'],
         previous_same_gpu_144=[row for row in peer_gpu['rows'] if row['modes']==144],device=peer_gpu['device'],
         archives={name:H(DATA/name/'manifest.json') for name in STAGES}))
    manifest(out)
    print(json.dumps(full,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):run(parser.parse_args().output.resolve())
