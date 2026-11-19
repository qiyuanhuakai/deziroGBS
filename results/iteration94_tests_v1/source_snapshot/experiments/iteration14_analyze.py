"""Keep noise-only costs separate from full sampling and nominal quality."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iteration14_run import save

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((ROOT/'results/iteration14_cost/rows.json').read_text())
    quality=json.loads((ROOT/'results/iteration14_quality/rows.json').read_text())
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in ('panel','modes','depth','samples','seed'))].append(r)
    summaries=[]
    for key,group in sorted(groups.items()):
        record=dict(zip(('panel','modes','depth','samples','seed'),key)); record['methods']={}
        for method in ('dense_full','dense_stream','local_stream'):
            subset=[r for r in group if r['method']==method]
            stats={'planned':len(subset),'complete':sum(r['status']=='complete' for r in subset)}
            if stats['complete']==stats['planned']:
                for metric in ('end_to_end_seconds','circuit_seconds','noise_seconds','peak_wset_bytes','baseline_rss_bytes'):
                    values=[r[metric] for r in subset]; stats[metric+'_median']=statistics.median(values)
                    stats[metric+'_min']=min(values); stats[metric+'_max']=max(values)
                if key[0]=='pipeline':
                    stats['construction_trace_budget']=subset[0]['construction_trace_budget']
                    stats['budget_at_most_point1']=all(r['construction_trace_budget']<=.1 for r in subset)
                    stats['sampling_seconds_median']=statistics.median(r['sampling_seconds'] for r in subset)
            record['methods'][method]=stats
        a,b,c=[record['methods'][k] for k in ('dense_full','dense_stream','local_stream')]
        if all(x['complete']==3 for x in (a,b,c)):
            record.update(dense_full_over_local_seconds=a['end_to_end_seconds_median']/c['end_to_end_seconds_median'],
                dense_stream_over_local_seconds=b['end_to_end_seconds_median']/c['end_to_end_seconds_median'],
                dense_full_over_local_peak=a['peak_wset_bytes_median']/c['peak_wset_bytes_median'],
                dense_stream_over_local_peak=b['peak_wset_bytes_median']/c['peak_wset_bytes_median'])
        if key[0]=='pipeline': record['all_sample_hashes_identical']=len({r.get('sample_sha256') for r in group})==1 and all(r['status']=='complete' for r in group)
        summaries.append(record)
    summary={'planned':len(rows),'complete':sum(r['status']=='complete' for r in rows),'quality_cases':len(quality),
        'quality_complete':sum(r['status']=='complete' for r in quality),'groups':summaries,
        'max_alpha_abs_error':max(r['alpha_max_abs_error'] for r in quality if r['status']=='complete'),
        'max_alpha_relative_l2_error':max(r['alpha_relative_l2_error'] for r in quality if r['status']=='complete'),
        'max_norm_relative_error':max(r['norm_preservation_relative_error'] for r in quality if r['status']=='complete'),
        'max_small_conditional_tv':max(float(r['conditional_ideal_law_tv']) for r in quality if 'conditional_ideal_law_tv' in r)}
    save(out/'summary.json',summary)
    lines=['# Iteration 14: isolated process costs','',
        'Three repetitions per method and circuit. Medians; full pipeline includes identical MPS construction. Peak working set includes imports. No full output-TV certificate.','',
        '| Panel | m | L | N | Seed | Dense full s | Dense streamed s | Local streamed s | Full/local time ratio | Dense/full MiB | Dense/stream MiB | Local/stream MiB | Construction budget |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        methods=[r['methods'][k] for k in ('dense_full','dense_stream','local_stream')]
        if not all(x['complete']==3 for x in methods):
            lines.append(f"| {r['panel']} | {r['modes']} | {r['depth']} | {r['samples']} | {r['seed']} | failed | | | | | | | |"); continue
        times=' | '.join(f"{x['end_to_end_seconds_median']:.6f}" for x in methods)
        mem=' | '.join(f"{x['peak_wset_bytes_median']/2**20:.2f}" for x in methods)
        budget=methods[-1].get('construction_trace_budget')
        lines.append(f"| {r['panel']} | {r['modes']} | {r['depth']} | {r['samples']} | {r['seed']} | {times} | {r['dense_full_over_local_seconds']:.3f} | {mem} | {budget if budget is not None else ''} |")
    (out/'summary_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Fixed example selected by panel design, not by best observed ratio.
    examples=[next(r for r in summaries if r['panel']==panel and r['modes']==m and r['depth']==4 and r['samples']==n and r['seed']==140801)
        for panel,m,n in [('noise',1024,2048),('pipeline',512,128)]]
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for col,r in enumerate(examples):
        labels=['Dense full','Dense batch','Local batch']; methods=[r['methods'][k] for k in ('dense_full','dense_stream','local_stream')]
        axes[0,col].bar(labels,[s['end_to_end_seconds_median'] for s in methods],color=['#547a9e','#58a39a','#cf854b'])
        axes[0,col].set_ylabel('Compute seconds (median of 3)'); axes[0,col].set_title(f"{r['panel']}: m={r['modes']}, N={r['samples']}")
        axes[1,col].bar(labels,[s['peak_wset_bytes_median']/2**20 for s in methods],color=['#547a9e','#58a39a','#cf854b'])
        axes[1,col].set_ylabel('Process peak working set (MiB)'); axes[1,col].set_ylim(bottom=0)
    fig.suptitle('Local propagation: noise savings versus complete sampler cost\nSeed 140801, depth 4; process memory includes imports',fontsize=12)
    fig.tight_layout(); fig.savefig(out/'local_displacement_cost.png',dpi=180); plt.close(fig)
    dest=out/'source_snapshot/experiments/iteration14_analyze.py'; dest.parent.mkdir(parents=True); shutil.copy2(__file__,dest)
    print(json.dumps({k:v for k,v in summary.items() if k!='groups'},indent=2))


if __name__=='__main__': main()
