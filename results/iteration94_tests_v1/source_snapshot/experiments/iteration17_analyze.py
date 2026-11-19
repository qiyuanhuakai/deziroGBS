"""Matched fixed-degree comparisons; extension remains exploratory."""
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
    out=ROOT/'results/iteration17_analysis'; out.mkdir(exist_ok=False)
    panel=ROOT/'results/iteration17_panel'; extension=ROOT/'results/iteration17_extension'
    read=lambda p:json.loads(p.read_text())
    inputs=read(panel/'inputs.json'); rows=read(panel/'rows.json'); extra=read(extension/'rows.json'); controls=read(panel/'controls.json')
    assert len(inputs)==36 and len(rows)==276 and len(extra)==216 and len(controls)==72
    assert all(r['status']=='complete' for r in rows+extra+controls)
    lookup={(r['index'],r['degree']):r for r in rows+extra}
    control={(r['index'],r['k']):r for r in controls}
    groups=[]
    for m in (4,8,12):
        for profile in ('lossy','bright','stress'):
            ids=[r['index'] for r in inputs if r['modes']==m and r['profile']==profile]
            group=dict(modes=m,profile=profile,n=len(ids),thermal_median=float(np.median([inputs[i]['thermal_tv'] for i in ids])))
            for d in (2,6,8):
                values=[lookup[i,d]['tv_repaired'] for i in ids]
                group[f'D{d}_median']=float(np.median(values))
                for k in (2,4): group[f'D{d}_beats_k{k}']=sum(lookup[i,d]['tv_repaired']<control[i,k]['tv_extended'] for i in ids)
            for k in (2,4): group[f'k{k}_median']=float(np.median([control[i,k]['tv_extended'] for i in ids]))
            groups.append(group)
    combined=list(lookup.values()); comparisons={}
    for d in (2,6,8):
        comparisons[f'D{d}']={f'beats_k{k}':sum(lookup[i,d]['tv_repaired']<control[i,k]['tv_extended'] for i in range(36)) for k in (2,4)}
        comparisons[f'D{d}']['beats_thermal']=sum(lookup[i,d]['tv_repaired']<inputs[i]['thermal_tv'] for i in range(36))
    refs=read(ROOT/'results/iteration17_reference/rows.json')
    summary=dict(initial_rows=len(rows),extension_rows=len(extra),unique_laws=len(combined),controls=len(controls),inputs=36,
        fixed_degree_comparisons=comparisons,
        D8_improves_D6=sum(lookup[i,8]['tv_repaired']<lookup[i,6]['tv_repaired'] for i in range(36)),
        D8_improves_D2=sum(lookup[i,8]['tv_repaired']<lookup[i,2]['tv_repaired'] for i in range(36)),
        increasing_steps=sum(lookup[i,d]['tv_repaired']>lookup[i,d-1]['tv_repaired']+1e-12 for i in range(36) for d in range(1,9)),
        inputs_with_increasing_step=sum(any(lookup[i,d]['tv_repaired']>lookup[i,d-1]['tv_repaired']+1e-12 for d in range(1,9)) for i in range(36)),
        laws_with_negative_mass_gt_1e12=sum(r['negative_mass']>1e-12 for r in combined),
        max_negative_mass=max(r['negative_mass'] for r in combined),
        max_prefix_regression=max(r.get('prefix_regression_error',0) for r in extra),
        max_D2_regression=max(r.get('degree_two_regression_error',0) for r in rows),
        max_retained_failure=max(r['failure_mass'] for r in controls),
        reference_maxima={k:max(float(r[k]) for r in refs) for k in refs[0] if k!='index'},groups=groups,
        scope='36 correlated-input cases, two seeds per ensemble. D8 m8/m12 is post-panel exploration. Enumerated laws, no scalable sampler claim.')
    save(out/'summary.json',summary)
    lines=['# Iteration 17 fixed-degree results','','TV is repaired Meixner vs target; retained includes failure symbol. Each row aggregates 2 ensembles x 2 seeds. Degree 8 for m8/m12 is post-panel exploratory.','','|m|profile|thermal|D2|D6|D8|k2|k4|D8 beats k4|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups: lines.append('|'+ '|'.join([str(g['modes']),g['profile']]+[f'{g[k]:.6g}' for k in ('thermal_median','D2_median','D6_median','D8_median','k2_median','k4_median')]+[str(g['D8_beats_k4'])+'/4'])+'|')
    (out/'summary_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
    for a,(m,p) in zip(axes.flat,((m,p) for m in (8,12) for p in ('lossy','bright','stress'))):
        for e,color in (('haar','tab:blue'),('local_L12','tab:orange')):
            ids=[r['index'] for r in inputs if r['modes']==m and r['profile']==p and r['ensemble']==e]
            a.plot(range(9),[np.median([lookup[i,d]['tv_repaired'] for i in ids]) for d in range(9)],'o-',color=color,label=e)
            a.axhline(np.median([control[i,4]['tv_extended'] for i in ids]),color=color,ls='--',alpha=.7)
        a.set(yscale='log',title=f'm={m}, {p}',xlabel='degree',ylabel='TV'); a.grid(alpha=.2)
    axes[0,0].legend(); fig.suptitle('Meixner repaired laws; dashed retained k4 (two-seed medians)\nD7/D8 are post-panel exploratory; explicit full-law enumeration')
    fig.tight_layout(); fig.savefig(out/'degree_comparison.png',dpi=160); plt.close(fig)
    shutil.copy2(__file__,out/'iteration17_analyze.py')
    print(json.dumps({k:v for k,v in summary.items() if k!='groups'},indent=2))

if __name__=='__main__': main()
