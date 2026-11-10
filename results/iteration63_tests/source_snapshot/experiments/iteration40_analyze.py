"""Render equal-attempt confirmation and the distinct deterministic path witness."""
import sys,json,shutil,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration22_external import digest,save
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration40_analysis';out.mkdir(exist_ok=False);sha=digest(__file__);shutil.copy2(__file__,out/Path(__file__).name);read=lambda p:decode(json.loads(Path(p).read_text()));groups=[];deps={}
    for panel in ('frugal','recent','conditional_mi'):
        folder=ROOT/f'results/iteration40_{panel}';s=read(folder/'summary.json');assert s['completed']==s['planned']==(3 if panel=='frugal' else 1)
        for p in folder.glob('*.json'):deps[p.relative_to(ROOT).as_posix()]=digest(p)
        bounds=read(folder/'target_bounds.json')
        for i,row in enumerate(read(folder/'rows.json')):
            a=read(folder/f'case_{i:02d}.audit.json');x=a['intervals']['X'];y=a['intervals']['Y'];label='Frugal K'+str(row['cutoff']) if panel=='frugal' else 'Forest '+panel;groups.append(dict(label=label,method=row['method'],samples=256,completed=row['completed_attempts'],failed=row['failed_attempts'],mean_clicks=row['sample_mean_mapped_clicks'],mean_square=row['sample_mean_mapped_square'],mean_lower=float(x['lower']*144),mean_upper=float(x['upper']*144),square_lower=float(y['lower']*5184),square_upper=float(y['upper']*5184),old_product_square_lower=float(y['product_lower_diagnostic']*5184),mean_TV_lower=float(a['observable_TV_lower']['X']),square_TV_lower=float(a['observable_TV_lower']['Y']),event_TV_lower=float(max(e['TV_lower'] for e in a['events'])),full_TV_lower=float(a['full_TV_statistical_lower']),shifted_TV_lower=float(a['shifted_target_TV_statistical_lower']),saturated_steps=row.get('saturated'),nonfinite_steps=row.get('nonfinite'),wall_seconds=row['seconds'],official_sampling_seconds=row.get('timings',{}).get('official_sampling'),audit_seconds=row.get('timings',{}).get('audit_replay'),native_calls=row.get('native_calls')))
    proofdir=ROOT/'results/iteration40_saturation_validation';sv=read(proofdir/'validation.json');assert sv['passed']==sv['count'];physical=read(proofdir/'physical_coefficients.json');raw=read(ROOT/'results/iteration39_global/case_01.stdout.json');selection=read(ROOT/'results/iteration40_saturation/selection.json');table=[]
    for row in physical['result']['records']:
        j=row['mode'];lo,hi=row['threshold'];native=float.fromhex(raw['audit_paths'][selection['sample']][j][1]);table.append(dict(mode_zero_based=j,native_threshold=native,physical_recurrence_lower=float(lo),physical_recurrence_upper=float(hi),physical_interval_width=float(hi-lo),difference_midpoint_from_native=float((lo+hi)/2)-native))
    for p in [proofdir/'validation.json',proofdir/'physical_coefficients.json',ROOT/'results/iteration40_saturation/selection.json',ROOT/'results/iteration39_global/case_01.stdout.json']:deps[p.relative_to(ROOT).as_posix()]=digest(p)
    for name,data in (('confirmation.csv',groups),('physical_saturation_path.csv',table)):
        with (out/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    fig,axes=plt.subplots(2,2,figsize=(13.5,9.0));x=np.arange(5);labels=['Frugal\nK1','Frugal\nK2','Frugal\nK4','Forest\nrecent','Forest\nconditional MI'];target_mean=float(bounds['targets']['X'][0]*144);target_square=float(bounds['targets']['Y'][0]*5184)
    for ax,center,lower,upper,target,title,ylabel in ((axes[0,0],'mean_clicks','mean_lower','mean_upper',target_mean,'256 attempts per configuration: mean','E[N]'),(axes[0,1],'mean_square','square_lower','square_upper',target_square,'Hybrid one-sided mean tests','E[(N - 72)^2]')):
        y=np.array([r[center] for r in groups]);lo=np.array([r[lower] for r in groups]);hi=np.array([r[upper] for r in groups]);ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='o',capsize=4,color='#277898');ax.axhline(target,linestyle='--',color='#ba5c40',label='Certified physical target');ax.set(title=title,ylabel=ylabel);ax.set_xticks(x,labels,fontsize=8);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    axes[1,0].bar(x,[r['full_TV_lower'] for r in groups],color=['#277898']*3+['#b37b28','#678d59']);axes[1,0].set_xticks(x,labels,fontsize=8);axes[1,0].set(title='Simultaneous statistical TV lower bounds',ylabel='Full-output TV lower bound',ylim=(0,max(r['full_TV_lower'] for r in groups)*1.25));axes[1,0].text(.5,.92,'Zero = not detected; no TV upper bound',transform=axes[1,0].transAxes,ha='center',fontsize=8);axes[1,0].grid(axis='y',alpha=.2)
    shown=table[-25:];axes[1,1].plot([r['mode_zero_based']+1 for r in shown],[r['physical_recurrence_lower'] for r in shown],color='#277898',label='Exact physical-coefficient recurrence');axes[1,1].scatter([r['mode_zero_based']+1 for r in shown],[r['native_threshold'] for r in shown],facecolors='none',edgecolors='#b37b28',s=28,label='Official binary64 path');axes[1,1].axhline(1,color='#ba5c40',linestyle='--');axes[1,1].scatter([107],[table[-1]['physical_recurrence_lower']],marker='*',s=120,color='#ba5c40',zorder=5);axes[1,1].set(title='Separate exploratory path certificate',xlabel='Mode number (one-based)',ylabel='Zero-branch mass / previous mass');axes[1,1].legend(fontsize=7,loc='lower left');axes[1,1].grid(axis='y',alpha=.2)
    fig.suptitle('Iteration 40: common-budget confirmation and a physical saturation witness',fontsize=14);fig.subplots_adjust(top=.91,bottom=.12,wspace=.28,hspace=.40);fig.text(.5,.035,'Confirmation: 57 predeclared tails, 99% simultaneous coverage under iid attempts.\nPath certificate: exact physical cumulants still yield a ratio above one; this is not a TV error bound.',ha='center',fontsize=9);fig.savefig(out/'confirmation_and_physical_saturation.png',dpi=180);plt.close(fig)
    result=dict(analyzer_sha256=sha,dependencies=deps,groups=groups,attempts=1280,completed=sum(r['completed'] for r in groups),failed=sum(r['failed'] for r in groups),target_mean=target_mean,target_square=target_square,confidence_tails=57,familywise_alpha=.01,physical_saturation=sv['cases'],saturation_physical_determinants=sv['physical_determinants'],scope='Common attempt budget, not equal compute cost; no full TV upper bound; deterministic path witness is separate from statistical family');assert digest(__file__)==sha;save(out/'summary.json',result);print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
