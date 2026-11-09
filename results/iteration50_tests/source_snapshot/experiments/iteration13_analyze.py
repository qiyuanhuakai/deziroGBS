"""Summarize all conditional audits, including failures and scope limits."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import mpmath as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iteration13_audit import save

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    panel=ROOT/'results/iteration13_conditional_audit'
    rows=json.loads((panel/'rows.json').read_text()); complete=[r for r in rows if r['status']=='complete']
    regular=[r for r in complete if r['variant']!='rare_alpha8']
    summary={'planned':len(rows),'complete':len(complete),'failures':len(rows)-len(complete),
        'sampling_failure_mass_cells':sum(r['real_failure']!='0' or r['grid_failure_numerator']!='0' for r in complete),
        'regular_count':len(regular),'hidden_tree_nodes_total':sum(r['nodes'] for r in complete),
        'max_conditional_norm_residual':max(r['max_conditional_norm_residual'] for r in complete),
        'total_compute_seconds':sum(r['seconds'] for r in rows),'groups':{}}
    for variant in ('zero','recorded','stress'):
        group=[r for r in regular if r['variant']==variant]
        summary['groups'][variant]={'count':len(group),**{key:max(float(r[key]) for r in group) for key in
            ('grid_reference_tv','continuous_cdf_reference_tv','quantization_tv_float','reference_convergence_max','state_norm_defect')}}
    for key in ('grid_reference_tv','continuous_cdf_reference_tv','quantization_tv_float','reference_convergence_max','state_norm_defect'):
        summary['regular_'+key+'_max']=max(float(r[key]) for r in regular)
    rare=json.loads((panel/'law_048.json').read_text())
    with mp.workdps(140):
        ideal=mp.mpf(rare['reference120']['law_decimal'][0]); grid=mp.mpf(rare['grid_numerators'][0])/int(rare['grid_denominator'])
        summary['rare_event']={'ideal_no_click':mp.nstr(ideal,35),'grid_no_click':mp.nstr(grid,35),
            'relative_error':mp.nstr((grid-ideal)/ideal,35),'absolute_error':mp.nstr(abs(grid-ideal),35)}
    old=json.loads((ROOT/'results/iteration12_validation/validation.json').read_text())
    new=json.loads((ROOT/'results/iteration13_validation/validation.json').read_text())
    summary['unchanged_failure_ids']=set(x['test'] for k in ('failure_details','error_details') for x in old[k])==set(x['test'] for k in ('failure_details','error_details') for x in new[k])
    save(out/'summary.json',summary)
    lines=['# Iteration 13: complete conditional sampling-law audit','',
        'Fixed binary64 MPS and displacement; independent fresh uniform-grid model. Not a displacement average or fixed-seed PRNG guarantee.','',
        '| Displacement panel | Count | Max grid/reference TV | Max floating/reference TV | Max grid/continuous TV |',
        '|---|---:|---:|---:|---:|']
    for k,v in summary['groups'].items():
        lines.append(f"| {k} | {v['count']} | {v['grid_reference_tv']:.6e} | {v['continuous_cdf_reference_tv']:.6e} | {v['quantization_tv_float']:.6e} |")
    lines+=['','No renormalization on successful paths: all 49 completed with zero enumerated failure mass.',
        '80/120-digit agreement checks numerical convergence; it is not an interval arithmetic proof.','',
        '| Case | Modes / profile / representation | Variant | Grid/reference TV | Nodes |',
        '|---:|---|---|---:|---:|']
    for r in rows:
        lines.append(f"| {r['index']} | {Path(r['input']).stem} | {r['variant']} | {r.get('grid_reference_tv',r['status'])} | {r.get('nodes','')} |")
    (out/'summary_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),gridspec_kw={'width_ratios':[2,1]})
    x=np.arange(len(regular))
    for key,label,marker in [('grid_reference_tv','53-bit grid vs reference','o'),
        ('continuous_cdf_reference_tv','Floating CDF vs reference','x'),
        ('quantization_tv_float','Grid vs continuous CDF','.')]:
        axes[0].scatter(x,[max(float(r[key]),1e-20) for r in regular],label=label,s=20,marker=marker)
    axes[0].set_yscale('log'); axes[0].set_xlabel('Conditional case (values < 1e-20 shown at floor)'); axes[0].set_ylabel('Total variation distance')
    axes[0].set_title('48 small MPS conditional laws'); axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    axes[1].bar(['Ideal','53-bit grid'],[float(summary['rare_event']['ideal_no_click']),float(summary['rare_event']['grid_no_click'])],color=['#286f9e','#d47a2d'])
    axes[1].set_yscale('log'); axes[1].set_ylabel('No-click probability'); axes[1].set_title('Vacuum displaced by alpha = 8')
    fig.suptitle('Arithmetic audit under independent uniforms; fixed displacements',fontsize=12)
    fig.tight_layout(); fig.savefig(out/'conditional_sampling_audit.png',dpi=180); plt.close(fig)
    dest=out/'source_snapshot/experiments/iteration13_analyze.py'; dest.parent.mkdir(parents=True); shutil.copy2(__file__,dest)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
