"""Descriptive selector comparisons; local certificates and global tests stay separate."""
import json,sys,shutil,csv
from pathlib import Path
from fractions import Fraction as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iteration22_external import digest,save
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration37_analysis';out.mkdir(exist_ok=False);sha=digest(__file__);shutil.copy2(__file__,out/Path(__file__).name);read=lambda p:decode(json.loads(Path(p).read_text()));local=ROOT/'results/iteration37_local';full=ROOT/'results/iteration37_full';rows=read(local/'rows.json');fr=read(full/'rows.json');target=read(ROOT/'results/iteration36_recovery/case_11.audit.json')['moments'];transfer=read(ROOT/'results/iteration35_panel/case_07.audit.json')['transfers']['-40']['all_measurement_TV_transfer_upper'];groups=[];ls=[];comparisons=[]
    for r in rows:
        a=read(local/f"case_{r['sequence']:02d}.audit.json")['audits'][1];ls.append(dict(start=r['start'],selection=r['selection'],forest=r['forest'],tv_lower=float(a['tv_lower']),tv_upper=float(a['tv_upper']),interval_width=float(a['tv_upper']-a['tv_lower']),failure_mass=float(a['seen_failure_mass']),seconds=r['seconds']))
    for r in fr:
        rec=read(full/f"case_{r['sequence']:02d}.audit.json");best=max(rec['events'],key=lambda e:e['full_extended_TV_statistical_lower']);groups.append(dict(selection=r['selection'],attempts=64,completed=r['completed_attempts'],failed=r['failed_attempts'],sample_mean_completed=r['mean_completed_clicks'],best_predeclared_threshold=best['threshold'],successes=best['successes'],q_lower=float(best['sampler_probability_lower']),target_upper=float(best['target_probability_upper']),statistical_tv_lower=float(best['full_extended_TV_statistical_lower']),shifted_target_statistical_tv_lower=float(max(F(0),best['full_extended_TV_statistical_lower']-transfer)),selector_seconds=rec['sampler_diagnostics']['selector_seconds'],seconds=r['seconds']))
    for start in (0,54,134):
        for forest in ('chain','static_mi_tree'):
            subset=[r for r in rows if r['start']==start and r['forest']==forest];tables={r['selection']:read(local/f"case_{r['sequence']:02d}.first.json") for r in subset}
            for other in ('recent','pairwise','click_influence'):
                a,b=tables['conditional_mi'],tables[other];comparisons.append(dict(start=start,forest=forest,first='conditional_mi',second=other,selected_set_disagreements=sum(set(x.get('retained',()))!=set(y.get('retained',())) for x,y in zip(a,b)),q_hex_disagreements=sum(x['q_hex']!=y['q_hex'] for x,y in zip(a,b)),prefixes=1023))
    for filename,data in (('local_complete_laws.csv',ls),('full_confirmation.csv',groups),('selector_disagreements.csv',comparisons)):
        with (out/filename).open('w',encoding='utf-8',newline='') as file:w=csv.DictWriter(file,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    selectors=('recent','pairwise','conditional_mi','click_influence');labels=('Recent','Pairwise MI','Conditional MI','Click influence');colors=('#aa5b50','#bb9037','#286d96','#5c8560');fig,axes=plt.subplots(1,3,figsize=(14,4.6))
    for ax,forest in zip(axes[:2],('chain','static_mi_tree')):
        for k,sel in enumerate(selectors):
            points=[r for r in ls if r['forest']==forest and r['selection']==sel];ax.bar(np.arange(3)+(k-1.5)*.2,[p['tv_upper'] for p in points],.2,color=colors[k],label=labels[k])
        ax.set_yscale('log');ax.set_xticks(range(3),['0-9','54-63','134-143']);ax.set(title=forest.replace('_',' '),xlabel='Ten-mode block',ylabel='Complete stopped-law TV');ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);axes[2].bar(range(4),[g['sample_mean_completed'] for g in groups],color=colors);axes[2].axhline(float(target['mean_interval'][0]),color='black',linestyle='--',label='Exact physical target mean');axes[2].set_xticks(range(4),['Recent','Pairwise\nMI','Conditional\nMI','Click\ninfluence'],fontsize=8);axes[2].set(title='144-mode confirmation',ylabel='Mean clicks in 64 completed outputs',ylim=(0,100));axes[2].legend(fontsize=8);axes[2].text(.02,.96,'Descriptive means are not a TV accuracy certificate.',transform=axes[2].transAxes,va='top',fontsize=7.5);fig.suptitle('Iteration 37: fixed four-click retention budget');fig.tight_layout();fig.savefig(out/'selector_comparison.png',dpi=180);plt.close(fig)
    summary=dict(analyzer_sha256=sha,local_cases=24,full_cases=4,local=ls,full_sampling=groups,selector_disagreements=comparisons,maximum_local_interval_width=max(r['interval_width'] for r in ls),full_attempts=256,full_completed=sum(r['completed'] for r in groups),full_failed=sum(r['failed'] for r in groups),formal_native_calls=sum(r['native_calls'] for r in rows+fr),formal_conditional_attempts=sum(r['conditional_attempts'] for r in rows+fr),reverse_mismatches=sum(r['reverse_mismatches'] for r in rows),familywise_alpha=.01,statistical_iid_premise=True,full_law_upper_bound=False,target_mean=float(target['mean_interval'][0]),target_variance=float(target['variance_interval'][0]),target_transfer_upper=float(transfer),timing_scope='nonisolated; panels and validation work may overlap')
    assert digest(__file__)==sha;save(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('local','selector_disagreements')},indent=2))
if __name__=='__main__':main()
