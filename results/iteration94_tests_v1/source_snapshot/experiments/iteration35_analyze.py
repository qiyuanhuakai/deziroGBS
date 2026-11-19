import json,sys,shutil,csv
from pathlib import Path
from fractions import Fraction as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration22_external import digest,save
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration35_analysis';out.mkdir(exist_ok=False);source=Path(__file__);sha=digest(source);shutil.copy2(source,out/source.name);panel=ROOT/'results/iteration35_panel';rows=json.loads((panel/'rows.json').read_text());read=lambda n:decode(json.loads((panel/n).read_text()));model=read('case_07.audit.json');prefix=[r for r in rows if r['panel']=='prefix'];sampling=[r for r in rows if r['panel']=='sampling'];transfer=model['transfers']['-40']['all_measurement_TV_transfer_upper'];table=[]
    for r in prefix:
        audit=read(f"case_{r['sequence']:02d}.audit.json")['audits'][-1];lo=audit['tv_lower'];hi=audit['tv_upper'];table.append(dict(start=r['start'],budget=r['budget'],forest=r['forest'],tv_lower=float(lo),tv_upper=float(hi),interval_width=float(hi-lo),target_transfer_upper=float(transfer),exact_BF_target_TV_lower=float(max(F(0),lo-transfer)),exact_BF_target_TV_upper=float(min(F(1),hi+transfer)),native_calls=r['native_calls'],seconds=r['seconds']))
    with (out/'prefix_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7));colors=plt.cm.tab10.colors
    for j,start in enumerate((0,18,54,136)):
        for forest,style in (('chain','--'),('static_mi_tree','-')):
            group=[r for r in table if r['start']==start and r['forest']==forest];axes[0].semilogy([r['budget'] for r in group],[r['tv_upper'] for r in group],style+'o',color=colors[j],label=f'{start}: {forest}')
    axes[0].set(xlabel='Retained-click budget',ylabel='Certified TV (53-bit uniform grid)',title='Actual 144-mode sampler, stopped after 8 modes');axes[0].set_xticks([0,2,4]);axes[0].grid(alpha=.25);axes[0].legend(fontsize=7,ncol=2)
    group=[r for r in table if r['budget']==4 and r['forest']=='static_mi_tree'];x=list(range(4));axes[1].bar([v-.18 for v in x],[r['tv_upper'] for r in group],width=.36,label='To explicitly shifted target');axes[1].bar([v+.18 for v in x],[r['exact_BF_target_TV_upper'] for r in group],width=.36,label='To exact B,F target (triangle bound)');axes[1].set_yscale('log');axes[1].set_xticks(x,[str(r['start']) for r in group]);axes[1].set(xlabel='First mode of 8-mode block',ylabel='TV upper bound',title='Target-transfer bound can dominate local error');axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.25);fig.suptitle('Iteration 35: certified physical targets; no full 144-step TV upper bound',fontsize=12);fig.tight_layout();fig.savefig(out/'physical_targets.png',dpi=180);plt.close(fig)
    summary=dict(analyzer_sha256=sha,formal_jobs=len(rows),completed_jobs=sum(r['status']=='completed' for r in rows),main_vacuum_interval_decimal=[format(float(x),'.17g') for x in model['vacuum_interval']],main_vacuum_interval_width=float(model['vacuum_interval'][1]-model['vacuum_interval'][0]),main_core_residual=float(model['core']['diagnostics']['exact_residual_upper']),original_model_operator_error_upper=float(model['original_comparison']['operator_difference_upper']),target_transfer_upper=float(transfer),physicality=[{k:r[k] for k in ('sequence','modes','bits','family','physicality_status')}|({'shift_exponent':r['shift_exponent']} if 'shift_exponent' in r else {}) for r in rows if r['panel']=='uncertainty'],prefix=table,total_native_calls=sum(r.get('native_calls',0) for r in rows),duplicate_prefix_checks=sum(r.get('duplicate_checks',0) for r in rows),sampling_attempts=32,sampling_completed=sum(r['completed_attempts'] for r in sampling),sampling_failed=sum(r['failed_attempts'] for r in sampling),effective_job_seconds_sum=sum(r['seconds'] for r in rows),scope='Eight-step stopped laws and physical-target transfer only; 32 full attempts are operational coverage, no full sampler accuracy estimate')
    assert digest(source)==sha;save(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('prefix','physicality')},indent=2))
if __name__=='__main__':main()
