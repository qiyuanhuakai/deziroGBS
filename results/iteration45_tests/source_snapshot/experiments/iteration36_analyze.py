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
    out=ROOT/'results/iteration36_analysis';out.mkdir(exist_ok=False);source=Path(__file__);sha=digest(source);shutil.copy2(source,out/source.name);p=ROOT/'results/iteration36_panel';r=ROOT/'results/iteration36_recovery';read=lambda n:decode(json.loads(Path(n).read_text()));rows=read(p/'rows.json')[:6]+read(r/'rows.json');main=read(r/'case_11.audit.json')['moments'];transfer=read(ROOT/'results/iteration35_panel/case_07.audit.json')['transfers']['-40']['all_measurement_TV_transfer_upper'];local=[];groups=[]
    for i in range(7,11):
        rec=read(r/f'case_{i:02d}.audit.json');first=rows[i]['start'];distance=rec['shifted_target_TV_interval']
        for a in rec['audits']:
            if a['uniform_bits']!=53:continue
            old=read(ROOT/f"results/iteration35_panel/case_{a['source_case']:02d}.audit.json")['audits'][-1];local.append(dict(start=first,budget=a['budget'],forest=a['forest'],direct_tv_lower=float(a['tv_lower']),direct_tv_upper=float(a['tv_upper']),direct_interval_width=float(a['tv_upper']-a['tv_lower']),old_transferred_tv_upper=float(min(F(1),old['tv_upper']+transfer)),physical_local_target_tv_lower=float(distance[0]),physical_local_target_tv_upper=float(distance[1])))
    for i in range(12,16):
        rec=read(r/f'case_{i:02d}.audit.json');best=max(rec['events'],key=lambda e:e['full_extended_TV_statistical_lower']);group=dict(sequence=i,budget=rows[i]['budget'],forest=rows[i]['forest'],attempts=64,completed=rows[i]['completed_attempts'],failures=rows[i]['failed_attempts'],sample_mean_completed=rows[i]['mean_completed_clicks'],best_predeclared_event=best['event'],successes=best['successes'],q_lower=float(best['sampler_probability_lower']),target_upper=float(best['target_probability_upper']),statistical_tv_lower=float(best['full_extended_TV_statistical_lower']),shifted_target_statistical_tv_lower=float(max(F(0),best['full_extended_TV_statistical_lower']-transfer)),seconds=rows[i]['seconds']);groups.append(group)
    for name,data in (('direct_local_comparison.csv',local),('full_sampling_event_tests.csv',groups)):
        with (out/name).open('w',encoding='utf-8',newline='') as file:writer=csv.DictWriter(file,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7));chosen=[a for a in local if a['budget']==4 and a['forest']=='static_mi_tree'];x=np.arange(4);axes[0].bar(x-.18,[a['old_transferred_tv_upper'] for a in chosen],.36,label='Round 35: global transfer');axes[0].bar(x+.18,[a['direct_tv_upper'] for a in chosen],.36,label='Round 36: direct exact target');axes[0].set_yscale('log');axes[0].set_xticks(x,[str(a['start']) for a in chosen]);axes[0].set(xlabel='First mode of eight-mode block',ylabel='Stopped eight-step TV upper bound',title='Tighter local certificates');axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.25)
    axes[1].bar(x,[g['statistical_tv_lower'] for g in groups],color=['#b33939','#d77932','#ad5656','#4b7c9c']);axes[1].set_xticks(x,[f"budget {g['budget']}\n{g['forest']}" for g in groups],fontsize=8);axes[1].set(ylabel='Full-output TV statistical lower bound',title='Global discrepancy despite accurate early prefixes',ylim=(0,max(.4,max(g['statistical_tv_lower'] for g in groups)*1.25)));axes[1].grid(axis='y',alpha=.25);axes[1].text(.02,.97,'99% simultaneous coverage under iid uniform-grid attempts\nZero means not detected, not accurate.',transform=axes[1].transAxes,va='top',fontsize=8);fig.suptitle('Iteration 36: exact local targets and predeclared full-output tests',fontsize=13);fig.tight_layout();fig.savefig(out/'local_and_global.png',dpi=180);plt.close(fig)
    summary=dict(analyzer_sha256=sha,effective_jobs=len(rows),original_completed=6,recovery_completed=10,preserved_primary_failures=1,preserved_dependent_stops=9,physical_model_mean_interval=[float(x) for x in main['mean_interval']],physical_model_mean_interval_width=float(main['mean_interval'][1]-main['mean_interval'][0]),physical_model_variance_interval=[float(x) for x in main['variance_interval']],physical_model_variance_interval_width=float(main['variance_interval'][1]-main['variance_interval'][0]),direct_local_max_interval_width=max(a['direct_interval_width'] for a in local),local_target_distance_range=[min(a['physical_local_target_tv_lower'] for a in local),max(a['physical_local_target_tv_upper'] for a in local)],local=local,full_sampling=groups,new_native_calls=sum(a.get('native_calls',0) for a in rows),new_attempts=256,new_completed=sum(g['completed'] for g in groups),new_failed=sum(g['failures'] for g in groups),predeclared_events=32,familywise_alpha=.01,conditional_independence_assumption=True,full_law_upper_bound=False,effective_job_seconds_sum=sum(a['seconds'] for a in rows))
    assert digest(source)==sha;save(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('local','full_sampling')},indent=2));print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
