import json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration23_support import digest,save
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'results/iteration32_analysis';out.mkdir(exist_ok=False);source=Path(__file__);sha=digest(source);shutil.copy2(source,out/source.name);rows=json.loads((ROOT/'results/iteration32_panel/rows.json').read_text());assert len(rows)==48 and all(r['status']=='completed' for r in rows)
    mps=rows[:24];forest=rows[24:];families=list(dict.fromkeys(r['family'] for r in mps));witnesses=[]
    for family in families:
        group=[r for r in mps if r['family']==family];best=max(group,key=lambda r:r['distances']['tv']);witnesses.append(dict(family=family,start=best['start'],four_mode_tv=best['distances']['tv'],four_mode_only_cauchy_upper=best['distances']['hellinger_cauchy_upper_for_this_block']))
    forest_summary=[dict(budget=budget,forest=kind,minimum_tv=min(r['distances']['tv'] for r in forest if r['budget']==budget and r['forest']==kind),maximum_tv=max(r['distances']['tv'] for r in forest if r['budget']==budget and r['forest']==kind)) for budget in (0,2,4) for kind in ('chain','static_mi_tree')]
    summary=dict(analyzer_sha256=sha,mps_witnesses=witnesses,forest_ranges=forest_summary,maximum_full_prefix_check_error=max(c['absolute_error'] for r in forest for c in r['full_prefix_checks']),full_prefix_checks=sum(len(r['full_prefix_checks']) for r in forest),scope='MPS four-mode TVs lower-bound the specified full mixed-state output distance. Cauchy upper bounds apply only to each displayed block. Forest eight-mode laws concern the explicitly reordered measurement prefixes and do not certify future numerical success or full-output accuracy.')
    fig,axes=plt.subplots(1,3,figsize=(17,4.8),layout='constrained');labels=['New 32','New 64','New 128','New 200','Author PCO','Author haf']
    for start in (0,18,54,140):axes[0].plot(labels,[next(r['distances']['tv'] for r in mps if r['family']==family and r['start']==start) for family in families],'o-',label=f'Modes {start}..{start+3}')
    axes[0].set_title('Finite MPS: deterministic four-mode TV');axes[0].set_ylabel('Full-output TV lower diagnostic');axes[0].tick_params(axis='x',rotation=30);axes[0].legend(fontsize=8)
    for kind in ('chain','static_mi_tree'):axes[1].semilogy([0,2,4],[next(r['maximum_tv'] for r in forest_summary if r['budget']==budget and r['forest']==kind) for budget in (0,2,4)],'o-',label=kind)
    axes[1].set_xlabel('Retained-click budget');axes[1].set_ylabel('Maximum among four stated orders');axes[1].set_title('Forest: exact eight-mode prefix TV');axes[1].legend(fontsize=8)
    pco=[r for r in mps if r['family']=='author_pco'];axes[2].plot([r['start'] for r in pco],[r['distances']['tv'] for r in pco],'o',label='Computed four-mode TV');axes[2].plot([r['start'] for r in pco],[r['distances']['hellinger_cauchy_upper_for_this_block'] for r in pco],'s',label='Upper bound for THIS BLOCK only');axes[2].set_xlabel('Four-mode block starting index (zero based)');axes[2].set_title('Author PCO: local upper vs local TV');axes[2].legend(fontsize=8);fig.suptitle('Iteration 32: deterministic output quality, with explicit marginal and ordering scope');fig.savefig(out/'block_quality.png',dpi=140);plt.close(fig)
    summary['unchanged_source']=digest(source)==sha;save(out/'summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
