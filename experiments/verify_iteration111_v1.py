"""Recompute release metrics from frozen arrays and exercise the complete rejection envelope."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from iteration111_sampling_v1 import DATA,ROOT,prepare,branches,dyads,click_table,complete_clicks


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    def folder(name):return DATA/f'iteration111_{name}_v1'
    def read(name,file):return json.loads((folder(name)/file).read_text(encoding='utf8'))
    op=np.load(DATA/'iteration110_collective_v1/optical.npz')
    ref=np.load(DATA/'iteration110_bus_v1/laws.npz')
    raw=read('memory','projection.json')
    matched=read('matched','matched.json')
    sample=read('sampling','sampling.json')
    peers=read('peers','peers.json')
    hw=read('transport','windows.json')
    redraw=read('windows','windows.json')
    het=read('heterogeneous','heterogeneous.json')
    wide=read('wide8','wide.json')
    small=read('wide','wide.json')
    stored=np.load(folder('peers')/'reference.npz')
    shots=np.load(folder('matched')/'matched.npz')
    hh=np.load(folder('transport')/'holdout.npz')
    assert sum(hh['hold_head'])==hw['holdout'] and sum(hh['train_head'])==hw['train']
    w=[]
    for i,route in enumerate(hw['routes']):
        before,after=route['before'],route['after']
        for prefix,dct in ((f'pre{i}',before),(f'post{i}',after)):
            delta=hh[prefix+'_joint']/route['shots']-hh['hold_joint']/hw['holdout']
            d=delta.diagonal()
            tv4=.5*(abs(delta)+abs(d[:,None]-delta)+abs(d[None,:]-delta)+abs(delta-d[:,None]-d[None,:]))
            vals=tv4[np.triu_indices(144,1)]
            np.testing.assert_allclose([vals.mean(),vals.max(),tv4[36,107]],
                    [dct['pair_mean_tv'],dct['pair_max_tv'],dct['anchor_pair_tv']],rtol=1e-12)
        assert after['pair_mean_tv']<before['pair_mean_tv'] and after['pair_max_tv']<before['pair_max_tv']
        assert np.array_equal(hh[f'pre{i}_total'],hh[f'post{i}_total'])
        assert np.array_equal(hh[f'pre{i}_head'],hh[f'post{i}_head'])
        w.append(dict(pnr=bool(i),us=route['timing'][1]['us'],shots=route['shots'],before=before,
                      after=after,changed_fraction=route['changed_fraction'],redraw=redraw['routes'][i]['after']))
    s,axes,obs=prepare(op,ref['bus'],1.)
    z,c,g,bound=branches(s,axes,6)
    local=click_table(op['core_x'],dyads(z,obs,4))
    model=complete_clicks(local,np.outer(c,c)*g)
    denom=np.zeros_like(model)
    for i,coef in enumerate(c):
        p=local[:,i,i].real
        denom+=abs(coef)*sum(abs(c))*np.kron(np.kron(p,p),np.kron(p,p))
    ratio=model/denom
    assert ratio.min()>=0 and ratio.max()<=1+1e-12
    acceptance=1/denom.sum()
    assert abs(acceptance-1/sum(abs(c))**2)<1e-12
    model_tv=float(abs(model-ref['truth4']).sum()/2)
    assert abs(model_tv-sample[1]['model_tv'])<1e-12
    event=(model<stored['frugal4'])
    dist=float(model[event].sum()-ref['truth4'][event].sum())
    fr_event=float(stored['frugal4'][event].sum()-ref['truth4'][event].sum())
    click=matched['native'][0]
    for k,v in [('hist0',click)]+[(f'frugal{r["k"]}',r) for r in matched['frugal_gpu']]:
        h=shots[k];n=v['shots']
        assert h.sum()==n
        assert abs(abs(h/n-ref['truth4']).sum()/2-v.get('empirical_tv',v.get('empirical_threshold_tv')))<1e-12
    for row in peers['frugal']:
        assert abs(abs(stored['frugal'+str(row['k'])]-ref['truth4']).sum()/2-row['model_tv'])<1e-12
    native=matched['native'][-1]
    for name,coeff,pkey in (('pair',.5,'pair_empirical_tv'),('total',1,'total_empirical_ks')):
        h=shots['pnr_'+name];p=stored[name];dif=h/native['shots']-p
        v=coeff*abs(dif).sum() if name=='pair' else abs(dif.cumsum()).max()
        assert abs(v-native[pkey])<1e-12
    for name,row in [('wide',small),('wide8',wide)]:
        z=np.load(folder(name)/'counts.npz')
        assert np.array_equal(z['first'].sum(1),z['totals'][:8])
        assert z['first'].shape==(8,131072) and (z['first']>1).any()
        assert abs(row['total_tv_budget']-(row['state_bound']+row['overflow_bound']+row['cells']*row['discarded_marginal_mass']))<1e-10
    assert shots['wide_click_sample'].shape==(8,16384)
    def med(times):return {str(k):float(np.median(v)) for k,v in times.items()}
    result=dict(input='1-radian global phase on a complex supermode, 4 detectors per lossy cell',
        device='NVIDIA GeForce RTX 5080 Laptop GPU',jiuzhang_reference_us=1.27,
        click=dict(omitted_bus_tv=float(abs(ref['base']-ref['truth4']).sum()/2),model_tv=model_tv,
            measured_us=med(click['timing']),shots=click['shots'],empirical_tv=click['empirical_threshold_tv'],
            cached_extra_preparation_s=sample[1]['cache_extra_setup_s'],cached_us=med(sample[1]['cached_us_by_threads']),
            acceptance=acceptance,acceptance_range=[float(ratio.min()),float(ratio.max())],fixed_event_bias=[dist,fr_event],
            cpu_frugal=[{k:r[k] for k in ('k','us','model_tv')} for r in peers['frugal']],
            gpu_frugal=[{k:r[k] for k in ('k','us','empirical_tv','shots')} for r in matched['frugal_gpu']]),
        pnr=dict(matched={k:v for k,v in native.items() if k!='timing'},us=med(native['timing']),
            photonic_engines=[{k:v for k,v in r.items() if k!='times'} for r in peers['pe']]),
        projections=raw,heterogeneous=het,wide_click=dict(**{k:v for k,v in matched['native'][1].items() if k!='timing'},
            us=med(matched['native'][1]['timing'])),wide_pnr=[{**{k:v for k,v in r.items() if k!='us_by_threads'},
            'us':med(r['us_by_threads'])} for r in (small,wide)],
        jiuzhang=dict(train=hw['train'],holdout=hw['holdout'],spatial_pairs=144*143//2,routes=w,
            split_reference=hw['split_reference'],frozen_spatial=hw['frozen']))
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    fig,ax=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    data=[r for r in raw if r['angle']==1.]
    ax[0].semilogy([r['nodes_per_axis'] for r in data],[100*r['click_model_tv'] for r in data],'o-',label='Shared mode')
    ax[0].axhline(100*peers['frugal'][1]['model_tv'],linestyle='--',color='#a64b22',label='Frugal K4')
    ax[0].axhline(100*result['click']['omitted_bus_tv'],linestyle=':',color='#555',label='Omitted phase')
    ax[0].set(xlabel='Coherent nodes per squeezed mode',ylabel='Complete click TV (%)',title='16 detectors, 1-radian phase')
    ax[0].legend(frameon=False)
    for j,(key,label) in enumerate((('before','Base'),('redraw','Redraw'),('after','Transport'))):
        ax[1].bar(np.arange(4)+(j-1)*.23,[100*r[key][f] for r in w for f in ('pair_mean_tv','pair_max_tv')],width=.23,label=label)
    ax[1].set_xticks(range(4),['mean','max','mean','max'])
    ax[1].set(xlabel='Direct clicks                    PNR to clicks',ylabel='Spatial pair TV (%)',title='Jiuzhang 3.0 holdout, 10,296 pairs')
    ax[1].legend(frameon=False)
    fig.savefig(out/'comparison.png',dpi=180)
    paths=[p for p in DATA.glob('iteration111*/*') if p.is_file() and p.name!='manifest.json']
    paths+=list((ROOT/'experiments').glob('*iteration111*.py'))+list((ROOT/'native').glob('*111*'))
    for path in (DATA/'iteration110_bus_v1/laws.npz',DATA/'iteration110_collective_v1/optical.npz',
            DATA/'iteration110_profile_v1/training.npz',DATA/'iteration110_joint_v1/tables.npz'):
        paths.append(path)
    files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
    (out/'manifest.json').write_text(json.dumps(files,indent=2)+'\n',encoding='utf8')
    print('verified',len(files),'source and evidence hashes',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=3):run(ap.parse_args().output.resolve())
