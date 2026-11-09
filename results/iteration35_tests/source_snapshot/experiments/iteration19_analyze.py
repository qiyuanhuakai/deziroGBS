"""Separate existence, rank obstruction and explicit construction evidence."""
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
    out=ROOT/'results/iteration19_analysis'; out.mkdir(exist_ok=False); panel=ROOT/'results/iteration19_panel'
    read=lambda p:json.loads(p.read_text())
    inputs=read(panel/'inputs.json'); rows=read(panel/'rows.json'); built=read(panel/'constructions.json')
    assert len(inputs)==20 and len(rows)==700 and len(built)==72 and all(r['status']=='complete' for r in rows+built)
    selected=[r for r in rows if r['modes']>=32 and r['dimension']==10 and r['rank']==128]
    pairs=[]
    for r in built:
        if r['method']!='local_layerwise': continue
        direct=next(x for x in built if x['method']=='direct_box_tt' and all(x[k]==r[k] for k in ('index','dimension','rank')))
        pairs.append(dict(index=r['index'],dimension=r['dimension'],rank=r['rank'],direct_distance=direct['state_trace_distance'],layerwise_distance=r['state_trace_distance'],
            direct_seconds=direct['total_seconds'],layerwise_seconds=r['total_seconds'],direct_stored=direct['stored_complex_elements'],layerwise_stored=r['stored_complex_elements']))
    summaries=[]
    for item in inputs[:12]:
        table=[r for r in rows if r['index']==item['index'] and r['dimension']==10]
        minimum=next((r['rank'] for r in table if r['target_met']),None)
        summaries.append(dict(index=item['index'],modes=item['modes'],seed=item['seed'],ensemble=item['ensemble'],min_tested_rank_d10=minimum,
            upper_d10_r128=next(r['existence_trace_upper_estimate'] for r in table if r['rank']==128),
            lower_r128=next(r['max_cut_rank_distance_lower_estimate'] for r in table if r['rank']==128),
            lower_r512=next(r['max_cut_rank_distance_lower_estimate'] for r in table if r['rank']==512)))
    refs=read(ROOT/'results/iteration19_reference/rows.json')
    summary=dict(inputs=20,cut_spectra=sum(r['cut_count'] for r in inputs),feasibility_rows=len(rows),constructed_small_states=len(built),
        original_failed_local_d10_r128_below_target=sum(r['target_met'] for r in selected if r['ensemble']=='local_L12' and r['seed'] in (150801,150802)),
        new_local_d10_r128_below_target=sum(r['target_met'] for r in selected if r['ensemble']=='local_L12' and r['seed'] in (190801,190802)),
        all_small_actual_distances_bounded=all(r['state_trace_distance']<=r['bound']+1e-10 for r in built),
        paired_constructions=len(pairs),direct_distance_better=sum(r['direct_distance']<r['layerwise_distance']-1e-12 for r in pairs),
        direct_total_time_lower=sum(r['direct_seconds']<r['layerwise_seconds'] for r in pairs),
        symplectic_max_error=max(r['independent_symplectic_max_error'] for r in inputs),
        marginal_reference_maxima={k:max(float(r[k]) for r in refs) for k in ('max_abs_error','max_relative_error','precision_convergence','max_chernoff_remainder')},
        large_input_summaries=summaries,paired_rows=pairs,
        scope='Existence estimates are not actual large MPS construction, interval certificates or output-TV lower bounds. Dense small TT-SVD and layerwise timings are single runs with box construction charged to direct method.')
    save(out/'summary.json',summary)
    lines=['# Iteration 19 all-cut representation diagnostics','','Ideal bounds evaluated in float64. d10/r128 upper is an existence bound; no large MPS is constructed. Lower bounds constrain the chosen pure core, not threshold-output TV.','','|m|seed|ensemble|min tested rank at d10, upper <=0.1|d10/r128 upper|rank128 lower|rank512 lower|','|---|---|---|---:|---:|---:|---:|']
    for r in summaries: lines.append('|'+ '|'.join([str(r[k]) for k in ('modes','seed','ensemble','min_tested_rank_d10')]+[f'{r[k]:.6g}' for k in ('upper_d10_r128','lower_r128','lower_r512')])+'|')
    (out/'summary_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,m in zip(axes,(32,128)):
        for e,col in (('local_L12','tab:blue'),('haar','tab:red')):
            for seed,style in ((150801,'-'),(150802,'--')):
                table=[r for r in rows if r['modes']==m and r['ensemble']==e and r['seed']==seed and r['dimension']==10]
                ax.plot([r['rank'] for r in table],[r['existence_trace_upper_estimate'] for r in table],style,color=col,label=f'{e} {seed} upper')
                ax.plot([r['rank'] for r in table],[r['max_cut_rank_distance_lower_estimate'] for r in table],':',color=col,alpha=.65)
        ax.axhline(.1,color='black',lw=1); ax.set(xscale='log',yscale='log',xlabel='bond cap',ylabel='pure-state distance bound',title=f'm={m}, d=10'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7); fig.suptitle('All cuts: existence upper and dotted rank lower estimates\nOriginal failed local inputs and matched Haar controls; no large MPS constructed')
    fig.tight_layout(); fig.savefig(out/'existence_and_obstruction.png',dpi=160); plt.close(fig)
    shutil.copy2(__file__,out/'iteration19_analyze.py'); print(json.dumps({k:v for k,v in summary.items() if k not in ('paired_rows',)},indent=2))

if __name__=='__main__': main()
