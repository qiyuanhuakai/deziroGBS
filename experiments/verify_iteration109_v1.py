"""Reconcile timing, full-law, all-port and finite-shot comparisons into the selected release."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iteration100_spatial_v1 import save,manifest
from iteration106_field_v1 import DATA
from iteration109_sample_v1 import tv
from iteration109_twins_v1 import optics

def read(name):return json.loads((DATA/name/'summary.json').read_text('utf-8'))

def physical_guards(m):
    l,f=optics(m);nbar=l/(1-l)
    eig=np.linalg.eigvalsh(np.einsum('kaj,kbj->jab',f,f))
    # Union bound across pairs, valid with correlated shared displacements.
    t=.495/eig[:,-1]
    log_rate=-150*t-.5*np.log1p(-2*t[:,None]*eig).sum(1)
    x=nbar[:,None]+2*eig;s=1+.98/x[:,-1]
    log_count=-1025*np.log(s)-.5*np.log1p((1-s[:,None])*x).sum(1)
    risk=float(np.sum(l**257)+np.exp(log_rate).sum()+np.exp(log_count).sum())
    # Displacements differing by delta on the idler have squared overlap
    # exp[-(2*nbar+1)|delta|^2] on a pure twin beam. Convexity and Jensen.
    rounded=f.astype(np.float32).astype(float)
    rounding=float(np.sqrt(((f-rounded)**2*(2*nbar+1)[None,None,:]).sum()))
    return dict(modes=m,ideal_guard_union_bound=risk,factor_rounding_tv_upper=rounding)

def fig(out,g,twin,peers,dev):
    fig,axes=plt.subplots(1,3,figsize=(12.8,3.7),layout='constrained')
    ax=axes[0];xx=np.arange(3);width=.28
    fr=read('iteration106_frugal_v1')['rows'];fr={p['panel']:p for p in fr}
    for i,(label,v,color) in enumerate([
        ('Local residual',[p['local_tv'] for p in g['edge_sixteen']],'#b78f4c'),
        ('Conditional edge',[p['edge_tv'] for p in g['edge_sixteen']],'#167369'),
        ('Frugal CPU K4',[fr[j]['K4']['tv'] for j in (0,1,5)],'#7754a6')]):
        ax.bar(xx+(i-1)*width,np.array(v)*100,width,label=label,color=color)
    ax.set_xticks(xx,['0 + 1','36 + 107','34 + 35']);ax.set_yscale('log');ax.set_ylabel('16-bit model TV (%)');ax.set_xlabel('Jiuzhang spatial outputs');ax.legend(fontsize=8)
    ax=axes[1];ms=[z['modes'] for z in twin['scaling']]
    for k,c in [('click','#167369'),('pnr','#9b5138')]:
        ax.loglog(ms,[z[k]['us'] for z in twin['scaling']],marker='o',label=k.upper(),color=c)
    ax.set_xlabel('Physical outputs');ax.set_ylabel('Full GPU record (µs)');ax.legend(fontsize=8)
    ax=axes[2];xx=np.arange(3)
    ax.bar(xx,[z['us'] for z in peers['synthetic_gpu']],color=['#167369','#9b84b6','#7754a6'])
    ax.set_yscale('log');ax.set_xticks(xx,['Twin beam','Frugal K2','Frugal K4']);ax.set_ylabel('Full 16-bit GPU record (µs)')
    for ax,title in zip(axes,['Measured optical input','Displaced twin beams','Same RTX 5080 Laptop']):
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(out/'accuracy.png',dpi=155);plt.close(fig)

def run(out):
    out.mkdir(exist_ok=False,parents=True)
    tree=read('iteration109_tree_v1');dev=read('iteration109_device_v1');gl=read('iteration109_global_v1')
    tw=read('iteration109_twins_v1');peer=read('iteration109_peers_v1');prior=read('iteration107_peers_v1')
    tr=np.load(DATA/'iteration109_tree_v1/laws.npz');devraw=np.load(DATA/'iteration109_device_v1/observed.npz')
    glob=np.load(DATA/'iteration109_global_v1/global.npz');twin=np.load(DATA/'iteration109_twins_v1/laws.npz')
    held=np.load(DATA/'iteration109_peers_v1/samples.npz')
    assert len(tree['edges'])==len(gl['edges'])==143
    for p,g in gl['priority']:assert [p,g] in gl['edges'] or [g,p] in gl['edges']
    for j,item in enumerate(gl['edge_sixteen']):assert abs(item['edge_tv']-tv(glob[f'edge{j}'],glob[f'edge_target{j}']))<1e-13
    for m in ('click','pnr'):
        assert tr[m+'_total'].sum()==32768 and devraw[m+'_total'].sum()==262144
        for n in ('first8','space36','edge47_0','edge47_1'):assert tr[m+'_'+n].sum()==32768
    assert np.array_equal(tr['hardware'],devraw['hardware']) and tr['hardware'].sum()==746306
    ct=glob['count_target'];assert tv(glob['count_model'],ct)<.00002
    tree_pair_tv=tv(tr['pnr_pair'].T.ravel()/32768,ct);gpu_pair_tv=tv(devraw['pnr_pair']/262144,ct)
    for z in tw['scaling']:assert z['first_pair_partial_transpose_symplectic_min']<1
    assert tv(twin['model'],twin['target'])<1e-10
    assert all(abs(twin[k].sum()-1)<1e-12 for k in ('model','target','frugal2','frugal4'))
    for row in peer['synthetic_gpu']:
        h=held['gpu_'+str(row['k'])];assert h.sum()==262144 and abs(tv(h/262144,twin['target'])-row['empirical_tv'])<1e-12
    assert gl['controls'][1]['max_readout']>=7 and all(z['unequal']==0 for z in gl['controls'])
    assert gl['recurrence']['pnr_max_cell_error']<1e-12 and peer['fock_matrix_max_abs_error']<1e-12
    guards=[physical_guards(z['modes']) for z in tw['scaling']]
    hashes={}
    for folder in ('tree','device','global','twins','peers','frugal_build'):
        for p in sorted((DATA/f'iteration109_{folder}_v1').iterdir()):
            hashes[p.relative_to(DATA).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    full_gpu=[dict(route=row['route'],us_2048=row['timing'][0]['us'],us_16384=row['timing'][1]['us'],
                   first8_empirical_tv=row['empirical_tv'][0],hardware_total_click_ks=row['hardware_ks']) for row in dev['actual']]
    actual_peers=[z for z in prior['upstream_gpu'] if z['panel']==0]
    selected=gl['edge_sixteen'];fr={z['panel']:z for z in read('iteration106_frugal_v1')['rows']}
    optical_edges=[dict(spatial_ports=ij,conditional_tv=z['edge_tv'],local_tv=z['local_tv'],frugal_projected_k4=fr[k]['K4'])
                   for ij,z,k in zip(gl['priority'],selected,(0,1,5))]
    save(out,'validation.json',dict(device=dev['device'],hardware_us=1.27,hardware_holdout=746306,
        jiuzhang=dict(complete_gpu_records=full_gpu,equal_eight_bit_host_outputs=peer['actual_gpu'],frugal_gpu=actual_peers,
                      conditional_edges=optical_edges,edge_count=143,edge_four_bit_max_tv=gl['four_bit_edge_max_tv'],
                      edge_pnr_tv=gl['two_port_pnr_tv'],tree_empirical_pnr_tv_32768=tree_pair_tv,
                      local_empirical_pnr_tv_262144=gpu_pair_tv,pe_reduced_pair=prior['pe_pair'],full_total_photon=gl['total_photon'],
                      tree_us=[v['timing']['us'] for v in tree['runs']],ablation=tree['panels'][:2]),
        entangled_twins=dict(width=tw['scaling'],same_device=peer['synthetic_gpu'],full_16_model_and_pnr=tw['comparison'],
                             concentrated_histogram_tv_95=peer['ideal_empirical_tv_95'],conditional_bounds=guards),
        local_squeezing=dev['scaling'],verification=dict(tree_recurrence=gl['recurrence'],twins_fock=peer['fock_matrix_max_abs_error'],
             twin_control=gl['controls'],scoped_guards='bounds cover the ideal-input probability of exceeding specified physical caps and isolated FP32 factor storage, not all arithmetic or pseudorandom error'),
        artifact_sha256=hashes))
    fig(out,gl,tw,peer,dev);manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);run(ap.parse_args().output.resolve())
