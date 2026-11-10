"""Separate exact local laws from simultaneous global discrepancy witnesses."""
import sys,json,shutil,csv,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration22_external import digest,save
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration39_analysis';out.mkdir(exist_ok=False);sha=digest(__file__);shutil.copy2(__file__,out/Path(__file__).name);read=lambda p:decode(json.loads(Path(p).read_text()));full=ROOT/'results/iteration39_global';local=ROOT/'results/iteration39_local';bounds=read(full/'target_bounds.json');groups=[];small=[];deps={}
    for folder,n in ((full,4),(local,15)):
        summary=read(folder/'summary.json');assert summary['completed']==summary['planned']==n
        for p in folder.glob('*.json'):deps[p.relative_to(ROOT).as_posix()]=digest(p)
    for i,row in enumerate(read(full/'rows.json')):
        a=read(full/f'case_{i:02d}.audit.json');r=dict(cutoff=i+1,samples=1024,mean_clicks=row['mean_clicks'],mean_centered_square=row['mean_centered_square'],mean_clicks_lower=float(144*a['intervals']['X']['lower']),mean_clicks_upper=float(144*a['intervals']['X']['upper']),centered_square_lower=float(5929*a['intervals']['Y']['lower']),centered_square_upper=float(5929*a['intervals']['Y']['upper']),mean_TV_lower=float(a['observable_TV_lower']['X']),square_TV_lower=float(a['observable_TV_lower']['Y']),tail_TV_lower=float(max(e['TV_lower'] for e in a['events'])),full_TV_lower=float(a['full_TV_statistical_lower']),shifted_TV_lower=float(a['shifted_target_TV_statistical_lower']),setup_seconds=sum(row['timings'][k] for k in ('input','layout','cumulants','target')),official_sampling_seconds=row['timings']['official_sampling'],audit_seconds=row['timings']['audit_replay'],wall_seconds=row['seconds'],nonfinite=row['nonfinite'],saturated=row['saturated']);raw=read(full/f'case_{i:02d}.stdout.json');bad=[(sample,j,float.fromhex(a[1])) for sample,path in enumerate(raw['audit_paths']) for j,a in enumerate(path) if float.fromhex(a[1])<0 or float.fromhex(a[1])>1];r.update(saturated_samples=len({sample for sample,j,q in bad}),saturated_threshold_min=min((q for sample,j,q in bad),default=None),saturated_threshold_max=max((q for sample,j,q in bad),default=None),first_saturated_mode=min((j for sample,j,q in bad),default=None));assert len(bad)==r['saturated'];groups.append(r)
    for i,row in enumerate(read(local/'rows.json')):
        a=read(local/f'case_{i:02d}.audit.json');small.append(dict(start=row['start'],cutoff=row['cutoff'],TV_uint64_lower=float(a['TV_intervals']['uint64'][0]),TV_uint64_upper=float(a['TV_intervals']['uint64'][1]),TV_continuous_lower=float(a['TV_intervals']['continuous'][0]),TV_continuous_upper=float(a['TV_intervals']['continuous'][1]),TV_interval_width=float(a['TV_intervals']['uint64'][1]-a['TV_intervals']['uint64'][0]),uniform_model_distance=float(a['uniform_model_distance']),legacy_half_L1=float(a['legacy_half_L1']),reachable_nonfinite_prefixes=len(a['laws']['uint64']['reachable_nonfinite_prefixes']),reachable_saturated_prefixes=len(a['laws']['uint64']['reachable_saturated_prefixes'])))
    for name,data in (('global_witnesses.csv',groups),('complete_local_laws.csv',small)):
        with (out/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    fig,axes=plt.subplots(2,2,figsize=(12.8,8.8));x=np.arange(1,5);target_mean=float(bounds['X'][0]*144);target_square=float(bounds['Y'][0]*5929)
    for ax,mean,lo,hi,target,title,ylabel in ((axes[0,0],'mean_clicks','mean_clicks_lower','mean_clicks_upper',target_mean,'144 modes: click-count mean','E[N]'),(axes[0,1],'mean_centered_square','centered_square_lower','centered_square_upper',target_square,'144 modes: centered second moment','E[(N - 67)^2]')):
        y=np.array([r[mean] for r in groups]);lower=np.array([r[lo] for r in groups]);upper=np.array([r[hi] for r in groups]);ax.errorbar(x,y,yerr=[y-lower,upper-y],fmt='o',capsize=5,color='#24779a');ax.axhline(target,color='#b45b40',linestyle='--',label='Certified physical target');ax.set(title=title,xlabel='Official Frugal cumulant cutoff',ylabel=ylabel,xticks=x);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    for start,color in zip((0,54,134),('#24779a','#bc8424','#678c55')):
        data=[r for r in small if r['start']==start];axes[1,0].plot([r['cutoff'] for r in data],[r['TV_uint64_lower'] for r in data],'-o',color=color,label=f'Modes {start}-{start+9}')
    axes[1,0].set(title='Complete ten-mode finite-word laws',xlabel='Official Frugal cumulant cutoff',ylabel='TV to certified physical marginal',yscale='log',xticks=range(1,6));axes[1,0].grid(axis='y',alpha=.2);axes[1,0].legend(fontsize=8)
    for offset,key,label,color in ((-.25,'setup_seconds','Setup','#97b5bc'),(0,'official_sampling_seconds','1024 official samples','#24779a'),(.25,'audit_seconds','Separate replay','#bc8424')):axes[1,1].bar(x+offset,[r[key] for r in groups],width=.24,label=label,color=color)
    axes[1,1].set(title='CPU costs, shown separately',xlabel='Official Frugal cumulant cutoff',ylabel='Seconds (log scale; nonisolated)',yscale='log',xticks=x);axes[1,1].legend(fontsize=8);axes[1,1].grid(axis='y',alpha=.2)
    fig.suptitle('Iteration 39: official FrugalGBS at 144 modes and exact local laws',fontsize=14);fig.subplots_adjust(top=.91,bottom=.13,wspace=.28,hspace=.40);fig.text(.5,.035,'Top: 1024 outputs per cutoff; 99% simultaneous iid witness family, not TV upper bounds.\nBottom left: independent uint64-word model. Timings do not establish a cross-method ranking.',ha='center',fontsize=9);fig.savefig(out/'official_frugal_laws_and_witnesses.png',dpi=180);plt.close(fig)
    result=dict(analyzer_sha256=sha,dependencies=deps,formal_global_configurations=4,global_samples=4096,local_configurations=15,local_samples=3855,global_results=groups,local_results=small,target_mean=target_mean,target_centered_square=target_square,confidence_family=dict(mean_tails=16,event_tails=28,alpha=.01,iid_premise=True),max_local_TV_interval_width=max(r['TV_interval_width'] for r in small),max_uniform_model_distance=max(r['uniform_model_distance'] for r in small),full_TV_upper_bound=False,scope='Exact rational local laws conditional on exported binary64 thresholds and independent uint64 words; global bounds require iid; fixed-seed replay does not certify iid');assert digest(__file__)==sha;save(out/'summary.json',result);print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
