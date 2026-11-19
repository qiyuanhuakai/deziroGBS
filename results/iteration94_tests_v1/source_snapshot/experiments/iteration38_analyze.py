"""Separate global confidence evidence from same-input small-vector diagnostics."""
import json,sys,shutil,csv,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration22_external import digest,save
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration38_analysis';out.mkdir(exist_ok=False);sha=digest(__file__);shutil.copy2(__file__,out/Path(__file__).name);read=lambda p:decode(json.loads(Path(p).read_text()));full=ROOT/'results/iteration38_confirmation';external=ROOT/'results/iteration38_external';rows=read(full/'rows.json');er=read(external/'rows.json');bounds=read(full/'target_bounds.json');groups=[];frugal=[];forest=[]
    for i,r in enumerate(rows):
        a=read(full/f'case_{i:02d}.audit.json');m=a['mean_interval'];groups.append(dict(selection=r['selection'],attempts=96,completed=r['completed_attempts'],failed=r['failed_attempts'],sample_mean_mapped_clicks=r['sample_mean_mapped_clicks'],mean_clicks_lower=float(m['lower']*144),mean_clicks_upper=float(m['upper']*144),mean_TV_statistical_lower=float(a['mean_TV_statistical_lower']),tail_TV_statistical_lower=float(max(e['full_TV_statistical_lower'] for e in a['events'])),full_TV_statistical_lower=float(a['full_TV_statistical_lower']),shifted_target_TV_statistical_lower=float(a['shifted_target_TV_statistical_lower']),seconds=r['seconds']))
    for r in er[1:]:
        if r['panel']=='frugal':frugal.append({k:r[k] for k in ('start','cutoff','samples','half_L1_export','normalization_defect','saturated_steps','nonfinite_steps','seconds')})
        else:forest.append({k:r[k] for k in ('start','selection','forest','tv_ideal','tv_grid','native_calls','max_context_probability_difference','context_structure_mismatches','q_hex_context_mismatches','seconds')})
    for name,data in (('full_mean_and_tail_confirmation.csv',groups),('frugal_exported_vector_diagnostics.csv',frugal),('same_input_forests.csv',forest)):
        with (out/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.2));x=np.arange(4);means=np.array([r['sample_mean_mapped_clicks'] for r in groups]);lower=np.array([r['mean_clicks_lower'] for r in groups]);upper=np.array([r['mean_clicks_upper'] for r in groups]);axes[0].errorbar(x,means,yerr=[means-lower,upper-means],fmt='o',capsize=5,color='#286d96');target=float(bounds['target_X_interval'][0]*144);axes[0].axhline(target,color='black',linestyle='--',label='Exact physical target mean');axes[0].set_xticks(x,['Recent','Pairwise MI','Conditional MI','Click influence'],fontsize=8);axes[0].set(title='144-mode confirmation: 96 attempts each',ylabel='Mapped click count (failure = 0)',ylim=(math.floor(min(target,min(lower))-2),math.ceil(max(upper)+2)));axes[0].legend(fontsize=8,loc='upper right');axes[0].grid(axis='y',alpha=.2)
    for start,color in zip((0,54,134),('#286d96','#b97b24','#5c8560')):
        f=[r for r in frugal if r['start']==start];baseline=next(r for r in forest if r['start']==start and r['selection']=='conditional_mi' and r['forest']=='static_mi_tree');axes[1].plot([r['cutoff'] for r in f],[r['half_L1_export'] for r in f],'-o',color=color,label=f'{start}-{start+9}');axes[1].scatter([6],[baseline['tv_ideal']],marker='D',s=45,color=color)
    axes[1].set_yscale('log');axes[1].set_xticks(range(1,7),['1','2','3','4','5','Forest\nbudget 4']);axes[1].set(title='Same ten-mode input comparisons',xlabel='Frugal cumulant cutoff / fixed forest',ylabel='Half-L1 diagnostic / stopped-law TV',ylim=(1e-6,.1));axes[1].legend(title='Ten-mode block',fontsize=8,title_fontsize=8);axes[1].grid(axis='y',alpha=.2);fig.suptitle('Iteration 38: stronger global witnesses and official FrugalGBS baseline',y=.99);fig.subplots_adjust(top=.86,bottom=.21,wspace=.30);fig.text(.5,.035,'Left: simultaneous iid confidence allocation. Right: floating Frugal trace vs certified forest law; no global speed or accuracy ranking.',ha='center',fontsize=8.5);fig.savefig(out/'global_witness_and_external.png',dpi=180);plt.close(fig)
    result=dict(analyzer_sha256=sha,formal_confirmation_cases=4,external_jobs=40,official_calibration_fixtures=15,full_attempts=384,full_completed=sum(r['completed'] for r in groups),full_failed=sum(r['failed'] for r in groups),full_native_calls=sum(r['native_calls'] for r in rows),small_native_calls=sum(r['native_calls'] for r in forest),mean_tails=8,mean_alpha_per_tail=.001,event_tails=32,event_alpha_per_tail=1/16000,familywise_alpha=.01,statistical_iid_premise=True,target_mean=target,full=groups,frugal=frugal,forest=forest,max_context_probability_difference=max(r['max_context_probability_difference'] for r in forest),context_q_hex_disagreements=sum(r['q_hex_context_mismatches'] for r in forest),Frugal_distance_scope='half-L1 to floating exported continuous-threshold vector; not a strict finite-RNG sampler certificate',full_TV_upper_bound=False,timing_scope='nonisolated; no cross-method runtime ranking');assert digest(__file__)==sha;save(out/'summary.json',result);print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
