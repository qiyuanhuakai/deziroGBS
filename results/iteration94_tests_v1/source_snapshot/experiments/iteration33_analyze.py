import json,shutil
from fractions import Fraction as F
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration22_external import digest,save
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'results/iteration33_analysis';out.mkdir(exist_ok=False);source=Path(__file__);sha=digest(source);shutil.copy2(source,out/source.name);panel=ROOT/'results/iteration33_panel';rows=json.loads((panel/'rows.json').read_text());assert len(rows)==38 and all(r['status']=='completed' for r in rows)
    forest=rows[:24];ranges=[]
    for k in (0,2,4):
        group=[r for r in forest if r['budget']==k];ratios=[r['metrics']['coupling_upper']/r['metrics']['extended_tv'] for r in group]
        ranges.append(dict(budget=k,minimum_ratio=min(ratios),maximum_ratio=max(ratios),maximum_tv=max(r['metrics']['extended_tv'] for r in group),maximum_coupling_upper=max(r['metrics']['coupling_upper'] for r in group),maximum_absolute_improvement_over_smaller_hybrid=max(min(r['metrics']['hybrid_target_upper'],r['metrics']['hybrid_proxy_upper'])-r['metrics']['coupling_upper'] for r in group)))
    partial=[]
    for budget in (0,8,32,128,255):
        bounds=[next(p['coupling_upper'] for p in r['partials'] if p['budget']==budget and p['uniform_bits'] is None) for r in forest];partial.append(dict(queries=budget,minimum_upper=min(bounds),maximum_upper=max(bounds)))
    native=[]
    for r in rows[24:]:
        data=json.loads((panel/f"case_{r['sequence']:02d}.audit.json").read_text());grid=data['grid53'];native.append(dict(case=r['sequence'],panel=r['panel'],modes=r['modes'],exponent=r.get('exponent'),precision=r['precision'],ideal_tv=r['metrics']['extended_tv'],grid53_tv=float(F(grid['extended_tv']['rational'])),ideal_failure_mass=r['metrics']['failure_mass_seen'],grid53_failure_mass=float(F(grid['failure_mass_seen']['rational'])),failure_nodes=r['failure_nodes'],queries=r['queries'],identical_query_order_replay=r['replay_table_identical']))
    summary=dict(analyzer_sha256=sha,forest_ranges=ranges,forest_partial_ranges=partial,native_cases=native,maximum_target_mass_error=max(abs(r['target_original_mass_error']) for r in forest),maximum_target_normalization_l1_change=max(r['target_normalization_l1_change'] for r in forest),maximum_grid_law_tv=max(r['grid_law_tv'] for r in rows),formal_seconds=sum(r['seconds'] for r in rows),scope='Exact rational bounds for supplied finite laws, including failure. Reused Gaussian target tables are explicitly normalized floating inputs; thermal targets are analytic. No 144-mode Gaussian TV certificate. Grid53 is an abstract independent-uniform grid model, not a deterministic PRNG proof.')
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for kind,marker in (('chain','o'),('static_mi_tree','s')):
        group=[r for r in forest if r['forest']==kind];axes[0,0].scatter([r['budget']+(-.05 if kind=='chain' else .05) for r in group],[r['metrics']['coupling_upper']/r['metrics']['extended_tv'] for r in group],marker=marker,label=kind)
    axes[0,0].set(title='Complete audit: coupling upper / actual TV',xlabel='Retained clicks',ylabel='Ratio');axes[0,0].legend(fontsize=8)
    axes[0,1].fill_between([r['queries'] for r in partial],[r['minimum_upper'] for r in partial],[r['maximum_upper'] for r in partial],alpha=.3,label='Range over 24 forest laws');axes[0,1].plot([r['queries'] for r in partial],[r['maximum_upper'] for r in partial],'o-');axes[0,1].set(title='Incomplete work retains unresolved mass',xlabel='Audited conditional queries',ylabel='Valid partial upper bound',ylim=(-.02,1.04));axes[0,1].legend(fontsize=8)
    for exponent,ax in ((8,axes[1,0]),(40,axes[1,1])):
        group=[r for r in native if r['panel']=='thermal' and r['modes']==8 and r['exponent']==exponent];labels=[r['precision'] for r in group]
        ax.semilogy(labels,[r['ideal_tv'] for r in group],'o-',label='Ideal Bernoulli extended TV');ax.semilogy(labels,[r['grid53_tv'] for r in group],'s--',label='Abstract grid53 extended TV')
        nonzero=[(j,r['ideal_failure_mass']) for j,r in enumerate(group) if r['ideal_failure_mass']];ax.scatter([j for j,_ in nonzero],[v for _,v in nonzero],marker='x',s=65,label='Explicit failure mass');ax.set(title=f'Analytic thermal target: 8 modes, excess 2^-{exponent}',ylabel='Probability / extended TV');ax.legend(fontsize=8)
    fig.suptitle('Iteration 33: failure-aware finite-law audits and their limits');fig.savefig(out/'coupling_quality.png',dpi=140);plt.close(fig);summary['unchanged_source']=digest(source)==sha;save(out/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
