"""Summarize actual Gaussian-basis constructions without upgrading TV scope."""
from collections import Counter
import argparse
import json
import math
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration21_worker import save
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--validation',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    read=lambda p:json.loads(p.read_text()); panel=ROOT/'results/iteration21_panel'; rows=read(panel/'rows.json')
    assert len(rows)==len(read(panel/'plan.json')['jobs'])==58
    complete=[r for r in rows if r['status'] in ('accepted','budget_missed')]; small=[r for r in complete if r['panel']=='quality']; large=[r for r in complete if r['panel']!='quality']
    primary=sorted([r for r in complete if r['panel']=='original'],key=lambda r:(r['modes'],r['seed']))
    ref=read(ROOT/'results/iteration21_panel_reference/rows.json'); pilot=read(ROOT/'results/iteration21_pilot/rows.json')
    failed_controls=read(args.validation/'artifact_checks.json')['failed_case_rank_lower_estimates']
    keys=('sequence','modes','seed','ensemble','panel','dimension','rank','status','construction_trace_budget','total_seconds','process_peak_wset_bytes')
    summary=dict(panel_jobs=len(rows),statuses=dict(Counter(r['status'] for r in rows)),panels={p:dict(Counter(r['status'] for r in rows if r['panel']==p)) for p in sorted({r['panel'] for r in rows})},
        primary=[{**{k:r[k] for k in keys},'projection_success':r['info']['projection_success'],'construction_seconds':r['info']['total_seconds'],'schmidt_weight_error':r['max_independent_schmidt_weight_error']} for r in primary],
        large=[{k:r[k] for k in keys} for r in large],
        all_small_bounds_hold=all(r['actual_state_distance']<=r['construction_trace_budget']+1e-10 for r in small),
        max_small_actual_distance=max(r['actual_state_distance'] for r in small),
        max_success_shortcut_distance_error=max(abs(math.sqrt(max(0.,1-r['info']['projection_success']))-r['actual_state_distance']) for r in small),
        max_schmidt_weight_error=max(r['max_independent_schmidt_weight_error'] for r in complete),
        max_reconstruction_residual=max(c['reconstruction_residual'] for r in complete for c in r['info']['cuts']),
        minimum_raw_symplectic_value=min(c['minimum_raw_nu'] for r in complete for c in r['info']['cuts']),
        max_recurrence_amplitude_error=max(r['selected_amplitude_max_error'] for r in ref if r['status']=='complete'),
        max_full_tensor_replay_error=max(r['full_tensor_max_error'] for r in ref if 'full_tensor_max_error' in r),
        observed_panel_samples=sum(r.get('completed_samples',0) for r in rows),observed_pilot_samples=sum(r.get('completed_samples',0) for r in pilot),
        max_peak_wset_bytes=max(r['process_peak_wset_bytes'] for r in rows),
        failed_case_rank_lower_estimates=failed_controls,
        scope='Actual saved MPS with ideal original-projector bounds evaluated in float64. No same-quality speed comparison or complete finite-precision sampler TV certificate.')
    save(out/'summary.json',summary)
    lines=['# Iteration 21: Gaussian-basis MPS conversion','','All saved candidates and failures are included. State bounds are ideal arithmetic estimates; single-run costs are not a speedup benchmark.','','|job|panel|m|seed|d|rank|status|state upper|small actual|construction sec|total sec|peak MiB|','|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        values=[str(r[k]) for k in ('sequence','panel','modes','seed','dimension','rank','status')]
        values.extend(f'{r[k]:.7g}' if k in r else 'unavailable' for k in ('construction_trace_budget','actual_state_distance'))
        values.extend([f"{r.get('info',{}).get('total_seconds',0):.4f}",f"{r.get('total_seconds',0):.4f}",f"{r.get('process_peak_wset_bytes',0)/2**20:.2f}"])
        lines.append('|'+ '|'.join(values)+'|')
    (out/'summary_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(range(len(primary)),[r['construction_trace_budget'] for r in primary],color='#216e99')
    axes[0].axhline(.1,color='black',linestyle='--',label='target 0.1'); axes[0].set_xticks(range(len(primary)),[f"m{r['modes']}\n{r['seed']}" for r in primary])
    axes[0].set(ylabel='ideal pure-state distance upper',title='Original deep cases: constructed, d10 / rank128',ylim=(0,.115)); axes[0].legend()
    for rank in (2,8,32):
        subset=[r for r in small if r['rank']==rank]; axes[1].scatter([r['construction_trace_budget'] for r in subset],[r['actual_state_distance'] for r in subset],label=f'rank {rank}',s=22,alpha=.8)
    maximum=max(r['construction_trace_budget'] for r in small)*1.05; axes[1].plot([0,maximum],[0,maximum],color='gray',linestyle='--')
    axes[1].set(xlabel='ideal state upper',ylabel='independent actual state distance',title='48 small-system checks'); axes[1].legend()
    fig.tight_layout(); fig.savefig(out/'construction_quality.png',dpi=160); plt.close(fig)
    shutil.copy2(__file__,out/'iteration21_analyze.py'); print(json.dumps({k:v for k,v in summary.items() if k not in ('primary','large')},indent=2))


if __name__=='__main__': main()
