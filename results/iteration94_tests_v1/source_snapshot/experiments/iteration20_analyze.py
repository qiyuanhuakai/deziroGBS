"""Separate band deletion, box/compression, actual construction and resources."""
import json
from pathlib import Path
import shutil
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration14_run import save
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/iteration20_analysis'; out.mkdir(exist_ok=False)
    read=lambda p:json.loads(p.read_text())
    panel=ROOT/'results/iteration20_panel'; scan=ROOT/'results/iteration20_band_scan'
    rows=read(panel/'rows.json'); bands=read(scan/'rows.json'); jobs=read(scan/'plan.json')['jobs']
    assert len(rows)==60 and len(bands)==250
    completed=[r for r in rows if r['status'] in ('accepted','budget_missed')]
    for row in completed:
        row['candidate_state_distance_lower_estimate']=max(0.,row['info']['band_distance']-row['info']['distance_to_band_gaussian'])
    selected=[]
    for i,job in enumerate(jobs):
        table=[r for r in bands if r['index']==i and r['status']=='complete']
        item=dict(job,index=i)
        for target,name in ((.1,'point1'),(.05,'point05')):
            first=next((r for r in table if r['band_distance']<=target),None)
            item['first_width_'+name]=first['width'] if first else None
            if first: item['d6_allocation_at_'+name]=first['allocation_upper_elements']['6']
        item['width3_distance']=next(r['band_distance'] for r in table if r['width']==3); selected.append(item)
    deep=[r for r in completed if r['panel']=='original_unresolved']; quality=[r for r in completed if r['panel']=='quality']
    summary=dict(jobs=len(rows),statuses=dict(Counter(r['status'] for r in rows)),completed_states=len(completed),
        panels={p:dict(Counter(r['status'] for r in rows if r['panel']==p)) for p in sorted({r['panel'] for r in rows})},
        deep_candidate_lower_min=min(r['candidate_state_distance_lower_estimate'] for r in deep),deep_candidate_lower_max=max(r['candidate_state_distance_lower_estimate'] for r in deep),
        deep_candidates_ruled_out_at_point1=sum(r['candidate_state_distance_lower_estimate']>.1 for r in deep),
        all_small_actual_bounds_hold=all(r['actual_state_distance']<=r['construction_trace_budget']+1e-10 and r['actual_state_distance']+1e-10>=r['candidate_state_distance_lower_estimate'] for r in quality),
        max_small_actual_distance=max(r['actual_state_distance'] for r in quality),
        observed_samples=sum(r.get('completed_samples',0) for r in rows),sampling_failures=sum(r.get('sampling_status')=='failed' for r in rows),
        maximum_peak_wset_bytes=max(r.get('process_peak_wset_bytes',0) for r in rows),
        max_covariance_fidelity_error=max(r.get('independent_covariance_fidelity_error',0) for r in bands),
        band_input_summaries=selected,
        completed_rows=[{k:r[k] for k in ('sequence','panel','modes','seed','depth','dimension','width','rank','status','construction_trace_budget','candidate_state_distance_lower_estimate','total_seconds','process_peak_wset_bytes')} for r in completed],
        scope='Pure-core state errors and observed samples, not full finite-precision output TV. Narrow band candidates are not the best band approximation. Preflight element guard is conservative, not a minimal-memory lower bound.')
    save(out/'summary.json',summary)
    lines=['# Iteration 20: direct banded Gaussian MPS','','Literal band deletion only; pure-state distance is not a threshold-output TV lower bound. Allocation is the conservative dense-frontier formula, not a general storage lower bound.','','|m|seed|depth|width3 distance|first width <=0.1|first width <=0.05|d6 elements at first <=0.1|','|---|---|---|---:|---:|---:|---:|']
    for r in selected: lines.append('|'+ '|'.join([str(r[k]) for k in ('modes','seed','depth')]+[f"{r['width3_distance']:.6g}"]+[str(r[k]) for k in ('first_width_point1','first_width_point05','d6_allocation_at_point1')])+'|')
    lines+=['','|job|panel|m|d|width|status|state upper|state lower|seconds|peak MiB|','|---|---|---|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        values=[str(r[k]) for k in ('sequence','panel','modes','dimension','width','status')]
        values += [f"{r.get('construction_trace_budget',float('nan')):.6g}",f"{r.get('candidate_state_distance_lower_estimate',float('nan')):.6g}",f"{r.get('total_seconds',float('nan')):.3f}",f"{r.get('process_peak_wset_bytes',0)/2**20:.1f}"]
        lines.append('|'+ '|'.join(values)+'|')
    (out/'summary_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,m in zip(axes,(32,128)):
        for seed in (150801,150802,200801,200802):
            table=[r for r in bands if r['modes']==m and r['seed']==seed and r['depth']==12 and r['status']=='complete']
            ax.plot([r['width'] for r in table],[max(1e-8,r['band_distance']) for r in table],label=str(seed))
        ax.axhline(.1,color='black',ls='--'); ax.axhline(.05,color='gray',ls=':'); ax.axvspan(0,3,color='gray',alpha=.1)
        ax.set(yscale='log',ylim=(1e-5,1.2),xlabel='retained pairing bandwidth',ylabel='pure Gaussian state distance',title=f'm={m}, depth=12'); ax.grid(alpha=.2)
    axes[0].legend(); fig.suptitle('Literal band deletion: narrow frontier construction loses long-range structure\nShaded widths 0-3; curves are state distances, not threshold-output TV')
    fig.tight_layout(); fig.savefig(out/'band_error.png',dpi=160); plt.close(fig)
    shutil.copy2(__file__,out/'iteration20_analyze.py'); print(json.dumps({k:v for k,v in summary.items() if k not in ('completed_rows','band_input_summaries')},indent=2))

if __name__=='__main__': main()
