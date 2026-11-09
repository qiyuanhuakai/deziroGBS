"""Separate reused exploration and predeclared new-seed fixed-degree results."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration14_run import save
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/iteration18_analysis'; out.mkdir(exist_ok=False); panel=ROOT/'results/iteration18_panel'
    read=lambda p:json.loads(p.read_text())
    inputs=read(panel/'inputs.json'); rows=read(panel/'rows.json'); controls=read(panel/'controls.json')
    assert len(inputs)==60 and len(rows)==7800 and len(controls)==120
    assert all(r['status']=='complete' for r in rows+controls)
    lookup={(r['index'],r['policy'],r['degree']):r for r in rows}; control={(r['index'],r['k']):r for r in controls}
    phases={}; groups=[]; margin=1e-10
    for phase in ('reused_exploration','new_seed_confirmation'):
        ids=[r['index'] for r in inputs if r['phase']==phase]; comparisons=[]
        for policy in ('paper','physical_mean'):
            for degree in (8,16,32,64):
                table=[lookup[i,policy,degree] for i in ids]
                comparisons.append(dict(policy=policy,degree=degree,tv_median=float(np.median([r['tv_repaired'] for r in table])),tv_max=max(r['tv_repaired'] for r in table),
                    **{f'beats_k{k}':sum(lookup[i,policy,degree]['tv_repaired']<control[i,k]['tv_extended']-margin for i in ids) for k in (2,4)},
                    beats_other_reference=sum(lookup[i,policy,degree]['tv_repaired']<lookup[i,'paper' if policy=='physical_mean' else 'physical_mean',degree]['tv_repaired']-margin for i in ids)))
        bound_degrees=[next((d for d in range(65) if lookup[i,'physical_mean',d]['repaired_tv_bound_estimate']<=.01),None) for i in ids]
        phases[phase]=dict(inputs=len(ids),comparisons=comparisons,bound_target=.01,bound_met=sum(d is not None for d in bound_degrees),first_bound_degrees=bound_degrees)
        for m in sorted({inputs[i]['modes'] for i in ids}):
            for profile in ('lossy','bright','stress'):
                sub=[i for i in ids if inputs[i]['modes']==m and inputs[i]['profile']==profile]
                group=dict(phase=phase,modes=m,profile=profile,n=len(sub))
                for policy in ('paper','physical_mean'):
                    for d in (8,16,32,64): group[f'{policy}_D{d}']=float(np.median([lookup[i,policy,d]['tv_repaired'] for i in sub]))
                for k in (2,4): group[f'k{k}']=float(np.median([control[i,k]['tv_extended'] for i in sub]))
                group['radius_paper']=lookup[sub[0],'paper',0]['spectral_radius']; group['radius_mean']=lookup[sub[0],'physical_mean',0]['spectral_radius']
                groups.append(group)
    refs=read(ROOT/'results/iteration18_reference/rows.json')
    summary=dict(inputs=60,method_runs=120,degree_rows=len(rows),controls=len(controls),comparison_margin=margin,phases=phases,groups=groups,
        old_wick_max_error=max(r.get('old_wick_regression_error',0) for r in rows),
        reference_maxima={k:max(float(r[k]) for r in refs) for k in ('precision_convergence','raw_law_abs_error','target_abs_error')},
        max_negative_mass=max(r['negative_mass'] for r in rows),negative_mass_gt_1e_minus_10=sum(r['negative_mass']>1e-10 for r in rows),
        max_retained_failure=max(r['failure_mass'] for r in controls),
        mean_matched_increasing_even_steps=sum(lookup[i,'physical_mean',d]['tv_repaired']>lookup[i,'physical_mean',d-2]['tv_repaired']+margin for i in range(60) for d in range(2,65,2)),
        mean_matched_bound_violation_max=max(r['tv_repaired']-r['repaired_tv_bound_estimate'] for r in rows if r['policy']=='physical_mean'),
        timing_scope='One full-law construction per input/policy, current exact enumeration measured separately; no sample throughput or equal-quality resource claim.',
        timing_medians={str(m):dict(exact=float(np.median([r['target_seconds'] for r in inputs if r['modes']==m])),**{p:float(np.median([lookup[r['index'],p,0]['total_seconds'] for r in inputs if r['modes']==m])) for p in ('paper','physical_mean')}) for m in (4,8,12)})
    save(out/'summary.json',summary)
    lines=['# Iteration 18 fixed-degree results','','TV is after explicit positive repair; k4 includes failure symbol. Mean = physical mean reference. Four cases per group (two ensembles x two seeds). Comparisons use an absolute 1e-10 margin to avoid counting numerical-floor differences.','','|phase|m|profile|paper D8|mean D8|mean D16|mean D32|mean D64|k4|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for g in groups: lines.append('|'+ '|'.join([g['phase'],str(g['modes']),g['profile']]+[f'{g[k]:.6g}' for k in ('paper_D8','physical_mean_D8','physical_mean_D16','physical_mean_D32','physical_mean_D64','k4')])+'|')
    (out/'summary_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
    for ax,(m,p) in zip(axes.flat,((m,p) for m in (8,12) for p in ('lossy','bright','stress'))):
        ids=[r['index'] for r in inputs if r['phase']=='new_seed_confirmation' and r['modes']==m and r['profile']==p]
        for policy,color in (('paper','tab:orange'),('physical_mean','tab:blue')):
            ax.plot(range(0,65,2),[np.median([lookup[i,policy,d]['tv_repaired'] for i in ids]) for d in range(0,65,2)],color=color,label=policy)
        ax.axhline(np.median([control[i,4]['tv_extended'] for i in ids]),ls='--',color='black',label='retained k4')
        ax.axhspan(1e-15,1e-10,color='gray',alpha=.12)
        ax.set(yscale='log',title=f'm={m}, {p}',xlabel='total degree',ylabel='repaired TV',ylim=(1e-14,1))
        ax.grid(alpha=.2)
    axes[0,0].legend(); fig.suptitle('New-seed panel: fixed reference policies, 4-case medians\nAll-subset enumeration; shaded region below comparison margin')
    fig.tight_layout(); fig.savefig(out/'new_seed_convergence.png',dpi=160); plt.close(fig)
    shutil.copy2(__file__,out/'iteration18_analyze.py')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('groups',)},indent=2))

if __name__=='__main__': main()
