"""Keep all unresolved cases, expanded resources and allocation ablations visible."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration14_run import save

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); panel=ROOT/'results/iteration16_pilot'
    rows=json.loads((panel/'rows.json').read_text()); groups=defaultdict(dict); attempts=[]; retries=Counter(); scan_costs=[]
    for r in rows:
        groups[tuple(r[k] for k in ('panel','modes','seed','target'))][r['method']]=r
        path=panel/f"run_{r['sequence']:02d}.attempts.json"
        if path.exists():
            trials=json.loads(path.read_text())
            attempts.extend(dict(t,sequence=r['sequence'],method=r['method']) for t in trials)
            if r['method']!='restart':
                scan_seconds=sum(x['seconds'] for t in trials for layer in t.get('layers',[]) for x in layer['rounding_attempts'])
                attempt_seconds=sum(t['seconds'] for t in trials)
                scan_costs.append({'sequence':r['sequence'],'modes':r['modes'],'method':r['method'],
                    'candidate_copy_and_scan_seconds':scan_seconds,'attempt_seconds':attempt_seconds,
                    'fraction':scan_seconds/attempt_seconds})
                for t in trials:
                    for layer in t.get('layers',[]):
                        retries.update('accepted' if x['accepted'] else 'rejected' for x in layer['rounding_attempts'])
    pairs=[]
    for key,methods in sorted(groups.items()):
        item=dict(zip(('panel','modes','seed','target'),key)); item['methods']=methods
        a,b=methods['restart'],methods['checkpoint_rolling']
        if a['status']==b['status']=='accepted':
            item['restart_over_checkpoint_search_seconds']=a['search_seconds']/b['search_seconds']
            item['restart_over_checkpoint_coefficients']=a['stored_complex_elements']/b['stored_complex_elements']
            if key[0]=='quality': item['checkpoint_actual_distance_not_worse']=b['state_trace_distance']<=a['state_trace_distance']+1e-10
        pairs.append(item)
    summary={'planned':len(rows),'status_counts':dict(Counter(r['status'] for r in rows)),
        'method_status_counts':{m:dict(Counter(r['status'] for r in rows if r['method']==m)) for m in ('restart','checkpoint_rolling','checkpoint_equal')},
        'attempt_status_counts':dict(Counter(t['status'] for t in attempts)),
        'checkpoint_rounding_retry_counts':dict(retries),'attempts_total':len(attempts),
        'small_state_bound_checks':sum(r.get('state_bound_check',False) for r in rows),
        'small_state_bound_failures':sum(not r['state_bound_check'] for r in rows if 'state_bound_check' in r),
        'complete_sampling_runs':sum(r.get('sampling_status')=='complete' for r in rows),
        'complete_sampling_outputs':sum(r.get('completed_samples',0) for r in rows if r.get('sampling_status')=='complete'),
        'checkpoint_scan_costs':scan_costs,'pairs':pairs}
    save(out/'summary.json',summary)
    lines=['# Iteration 16: layer checkpoints and rolling budgets','',
        'One exploratory run per method/input. Larger caps/resources than iteration 15; restart rerun as control. Soft-limit granularity differs, so equal per-attempt seconds is not equal total budget.','',
        '| m | Seed | Target | Method | Status | d | Bound | Search seconds (all attempts) | Final coefficients | Process peak MiB incl. diagnostics | Actual small-state distance |',
        '|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for group in pairs:
        for method,r in group['methods'].items():
            peak=r.get('process_peak_wset_bytes')
            lines.append(f"| {r['modes']} | {r['seed']} | {r['target']} | {method} | {r['status']} | {r.get('dimension','')} | {r.get('construction_trace_budget','')} | {r.get('search_seconds','')} | {r.get('stored_complex_elements','')} | {peak/2**20 if peak else ''} | {r.get('state_trace_distance','')} |")
    lines+=['','## All attempt terminal states','',
        '| Run | Method | d | cap | State | Complete/partial ledger | Error |',
        '|---:|---|---:|---:|---|---:|---|']
    for t in attempts:
        lines.append(f"| {t['sequence']} | {t['method']} | {t['dimension']} | {t.get('cap','layerwise')} | {t['status']} | {t.get('unclipped_construction_trace_budget',t.get('partial_ledger_sum',''))} | {t.get('error','')} |")
    (out/'summary_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    order={'restart':0,'checkpoint_equal':1,'checkpoint_rolling':2}
    hard=sorted((r for r in rows if r['panel']=='unresolved'),key=lambda r:(r['modes'],r['seed'],order[r['method']]))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    labels=[f"{r['modes']} / {r['seed']}\n{r['method'].replace('checkpoint_','cp ')}" for r in hard]
    colors=['#4c799f' if r['method']=='restart' else '#c47e45' if r['method']=='checkpoint_rolling' else '#599f8c' for r in hard]
    axes[0].bar(range(len(hard)),[r.get('search_seconds',0) for r in hard],color=colors)
    axes[0].set_ylabel('Search seconds, all attempts'); axes[0].set_title('All ten large-case searches ended unaccepted')
    axes[0].set_xticks(range(len(hard)),labels,rotation=70,ha='right',fontsize=7)
    spectra=json.loads((ROOT/'results/iteration16_schmidt/rows.json').read_text())
    keys=[(m,s) for m in (32,128) for s in (150801,150802)]
    for j,ensemble in enumerate(('local_L12','haar')):
        vals=[max(r['pure_state_distance_lower_bound'] for r in spectra if r['modes']==m and r['seed']==s and r['ensemble']==ensemble and r['rank']==128) for m,s in keys]
        axes[1].bar([i+(j-.5)*.36 for i in range(4)],vals,width=.36,label=ensemble)
    axes[1].axhline(.1,color='black',ls='--',label='0.1 reference'); axes[1].set_yscale('log')
    axes[1].set_xticks(range(4),[f'{m} / {s}' for m,s in keys],rotation=25,ha='right',fontsize=8)
    axes[1].set_ylabel('Pure-state distance obstruction (three cuts)'); axes[1].set_title('Rank 128: approximate lower bound, not output TV'); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'checkpoint_progress.png',dpi=180); plt.close(fig)
    dest=out/'source_snapshot/experiments/iteration16_analyze.py'; dest.parent.mkdir(parents=True); shutil.copy2(__file__,dest)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('pairs','checkpoint_scan_costs')},indent=2))


if __name__=='__main__': main()
