"""Separate certified-by-ledger construction targets from actual small-state quality."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iteration14_run import save

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    panel=ROOT/'results/iteration15_pilot'; rows=json.loads((panel/'rows.json').read_text())
    groups=defaultdict(dict); trial_counts=Counter(); attempts=[]; unaccepted_reasons=Counter()
    for r in rows:
        groups[tuple(r[k] for k in ('panel','modes','depth','profile','seed','target'))][r['method']]=r
        path=panel/f"run_{r['sequence']:03d}.attempts.json"
        if path.exists():
            trials=json.loads(path.read_text()); trial_counts.update(x['status'] for x in trials)
            attempts.extend(dict(x,sequence=r['sequence'],method=r['method']) for x in trials)
            if r['status']=='search_unaccepted' and trials: unaccepted_reasons[trials[-1]['status']]+=1
    pairs=[]
    for key,group in sorted(groups.items()):
        a=group['adaptive']; f=group['fixed_d8_chi16']; pair=dict(zip(('panel','modes','depth','profile','seed','target'),key))
        pair.update(adaptive_status=a['status'],fixed_status=f['status'],adaptive_sequence=a['sequence'],fixed_sequence=f['sequence'])
        for label,r in [('adaptive',a),('fixed',f)]:
            for field in ('dimension','cap','construction_trace_budget','construction_search_seconds','stored_complex_elements','state_trace_distance','construction_peak_wset_bytes','max_bond_actual'):
                if field in r: pair[label+'_'+field]=r[field]
        if a['status']=='accepted' and f['status']=='accepted':
            pair['fixed_over_adaptive_search_cost']=f['construction_search_seconds']/a['construction_search_seconds']
            pair['fixed_over_adaptive_coefficients']=f['stored_complex_elements']/a['stored_complex_elements']
            if 'state_trace_distance' in a:
                pair['adaptive_actual_distance_not_worse']=a['state_trace_distance']<=f['state_trace_distance']+1e-10
        pairs.append(pair)
    summary={'planned':len(rows),'status_counts':dict(Counter(r['status'] for r in rows)),
        'method_status_counts':{method:dict(Counter(r['status'] for r in rows if r['method']==method)) for method in ('adaptive','fixed_d8_chi16')},
        'trial_status_counts':dict(trial_counts),'attempts_total':len(attempts),
        'unaccepted_last_status_counts':dict(unaccepted_reasons),
        'accepted_only_adaptive':sum(p['adaptive_status']=='accepted' and p['fixed_status']!='accepted' for p in pairs),
        'accepted_only_fixed':sum(p['fixed_status']=='accepted' and p['adaptive_status']!='accepted' for p in pairs),
        'both_accepted':sum(p['fixed_status']==p['adaptive_status']=='accepted' for p in pairs),
        'small_state_bound_checks':sum(r.get('state_bound_check',False) for r in rows),
        'small_state_bound_failures':sum(not r['state_bound_check'] for r in rows if 'state_bound_check' in r),
        'complete_sampling_runs':sum(r.get('sampling_status')=='complete' for r in rows),
        'accepted_sampling_outputs':sum(r.get('completed_samples',0) for r in rows if r['status']=='accepted'),
        'pairs':pairs}
    comparable=[p for p in pairs if 'fixed_over_adaptive_search_cost' in p]
    summary['both_accepted_adaptive_faster_single_run']=sum(p['fixed_over_adaptive_search_cost']>1 for p in comparable)
    summary['both_accepted_adaptive_fewer_coefficients']=sum(p['fixed_over_adaptive_coefficients']>1 for p in comparable)
    small_comparable=[p for p in comparable if 'adaptive_actual_distance_not_worse' in p]
    summary['small_both_accepted_actual_not_worse']=sum(p['adaptive_actual_distance_not_worse'] for p in small_comparable)
    summary['small_both_accepted_actual_compared']=len(small_comparable)
    save(out/'summary.json',summary)
    # Full trial ledgers stay in the run directory; summarize status without dropping failures.
    lines=['# Iteration 15: budget-driven construction pilot','',
        'Single runs, including adaptive retry cost. A common nominal construction target is not equal actual TV. Resource stops are unresolved, not proofs of impossibility.','',
        '| Panel | m | L | Profile | Seed | Target | Adaptive status | d / cap | Bound | Search s | Fixed status | Fixed bound | Fixed s |',
        '|---|---:|---:|---|---:|---:|---|---|---:|---:|---|---:|---:|']
    for x in pairs:
        lines.append(f"| {x['panel']} | {x['modes']} | {x['depth']} | {x['profile']} | {x['seed']} | {x['target']} | {x['adaptive_status']} | {x.get('adaptive_dimension','')} / {x.get('adaptive_cap','')} | {x.get('adaptive_construction_trace_budget','')} | {x.get('adaptive_construction_search_seconds','')} | {x['fixed_status']} | {x.get('fixed_construction_trace_budget','')} | {x.get('fixed_construction_search_seconds','')} |")
    (out/'summary_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    large=[x for x in pairs if x['panel']=='scaling']
    for method,marker in [('fixed','o'),('adaptive','x')]:
        valid=[(i,x) for i,x in enumerate(large) if method+'_construction_trace_budget' in x]
        axes[0].scatter([i for i,x in valid],[x[method+'_construction_trace_budget'] for i,x in valid],label=method,marker=marker)
    axes[0].axhline(.1,color='black',linestyle='--',label='nominal target')
    axes[0].set_yscale('log'); axes[0].set_xlabel('Scaling input (all 18 cases)'); axes[0].set_ylabel('Construction trace budget')
    axes[0].set_title('Missing adaptive points = unaccepted search'); axes[0].legend()
    small=[x for x in pairs if 'adaptive_state_trace_distance' in x]
    axes[1].scatter([x['fixed_state_trace_distance'] for x in small],[x['adaptive_state_trace_distance'] for x in small])
    hi=max([x['fixed_state_trace_distance'] for x in small]+[x['adaptive_state_trace_distance'] for x in small])*1.05
    axes[1].plot([0,hi],[0,hi],'k--'); axes[1].set_xlabel('Fixed actual state distance'); axes[1].set_ylabel('Adaptive actual state distance')
    axes[1].set_title('Small-state quality; below diagonal is better')
    fig.tight_layout(); fig.savefig(out/'adaptive_construction.png',dpi=180); plt.close(fig)
    dest=out/'source_snapshot/experiments/iteration15_analyze.py'; dest.parent.mkdir(parents=True); shutil.copy2(__file__,dest)
    print(json.dumps({k:v for k,v in summary.items() if k!='pairs'},indent=2))


if __name__=='__main__': main()
