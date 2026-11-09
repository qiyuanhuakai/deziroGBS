"""Separate failure of universal accuracy improvement from numerical accuracy."""
import json,csv,shutil,math
from pathlib import Path
from fractions import Fraction as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'results/iteration44_analysis';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('iteration44_analyze','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    rows=read('results/iteration44_recovery_v2/comparisons.json');mechanism=read('results/iteration44_mechanism_validation/comparisons.json');cases=[read(f'results/iteration44_panel/case_{i:02d}.json') for i in range(24)];native={(i,k):read(f'results/iteration44_native/case_{i:02d}_k{k}.audit.json') for i in range(24) for k in (2,4)};runtimes=read('results/iteration44_panel/rows.json');order_results=[];infinity_count=0;reachable_infinity_count=0
    for i in range(0,24,2):
        n=cases[i]['job']['modes']
        def aligned(values,case):
            result=[F(0)]*len(values);order=cases[case]['job']['order']
            for mask,p in enumerate(values):result[(mask&~((1<<n)-1))|sum(1<<order[j] for j in range(n) if mask>>j&1)]=p
            return result
        pair=dict(cases=(i,i+1))
        for name in ('completion','tree','K2','K4'):
            def law(case):return cases[case]['projection']['law'] if name=='completion' else cases[case]['tree']['law'] if name=='tree' else native[case,int(name[1:])]['laws']['uint64']['distribution']
            pair[name+'_aligned_TV']=sum(abs(a-b) for a,b in zip(aligned(law(i),i),aligned(law(i+1),i+1)))/2
        order_results.append(pair)
    for i in range(24):
        for k in (2,4):
            raw=read(f'results/iteration44_native/case_{i:02d}_k{k}.stdout.json');law=native[i,k]['laws']['uint64']['distribution']
            for node in raw['prefix_tree']:
                if math.isinf(float.fromhex(node['threshold_hex'])):
                    infinity_count+=1;prefix=node['prefix'];mask=sum(1<<j for j,bit in enumerate(prefix) if bit=='1');mass=sum(law[a] for a in range(mask,1024,1<<len(prefix)));reachable_infinity_count+=mass>0
    first=cases[0];R=first['projection']['law'];target=first['target']['target_intervals'];event=[x for x,(p,(a,b)) in enumerate(zip(R,target)) if p>b];Q=sum(R[x] for x in event);P=(sum(target[x][0] for x in event),sum(target[x][1] for x in event));eps=first['bridge']['TV_to_true_maxent_upper'];event_record=dict(case=0,masks=event,completion_probability=Q,physical_probability_interval=P,true_maxent_event_gap_lower=max(F(0),Q-eps-P[1]),true_maxent_physical_TV_interval=first['true_maxent_physical_TV_interval'],same_moment_K2_feasible_upper=mechanism[0]['true_feasible_upper']);assert event_record['true_maxent_event_gap_lower']>event_record['same_moment_K2_feasible_upper'];save(out/'event_witness.json',encode(event_record));save(out/'order_comparisons.json',encode(order_results))
    with (out/'comparisons.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['case','modes','strength','transmission','order','completion_TV_lower','K2_TV_lower','K4_TV_lower','tree_TV_lower','TV_to_true_maxent_upper','true_moment_bridge_upper'])
        for r,c in zip(rows,cases):j=c['job'];writer.writerow([r['case'],r['modes'],j['strength'],str(j['transmission']),j['order_label'],float(r['completion_TV'][0]),float(r['K2_TV'][0]),float(r['K4_TV'][0]),float(r['tree_TV'][0]),float(r['maxent_distance_upper']),float(r['true_moment_bridge_upper'])])
    ratios=[r['completion_TV'][0]/r['K2_TV'][1] for r in rows];summary=dict(cases=24,distinct_inputs_up_to_order=12,completion_worse_K2=sum(r['completion_worse_K2'] for r in rows),completion_worse_K4=sum(r['completion_worse_K4'] for r in rows),true_maxent_worse_K2=sum(r['true_maxent_worse_K2'] for r in rows),completion_better_tree=sum(r['completion_TV'][1]<r['tree_TV'][0] for r in rows),physical_error_ratio_to_K2_range=[float(min(ratios)),float(max(ratios))],maxent_distance_bound_range=[float(min(r['maxent_distance_upper'] for r in rows)),float(max(r['maxent_distance_upper'] for r in rows))],true_moment_bridge_upper_range=[float(min(r['true_moment_bridge_upper'] for r in rows)),float(max(r['true_moment_bridge_upper'] for r in rows))],same_moment_counterexamples=12,case0_excess_lower=float(mechanism[0]['true_maxent_excess_lower']),maximum_completion_order_TV=float(max(r['completion_aligned_TV'] for r in order_results)),maximum_tree_order_TV=float(max(r['tree_aligned_TV'] for r in order_results)),maximum_K2_order_TV=float(max(r['K2_aligned_TV'] for r in order_results)),maximum_K4_order_TV=float(max(r['K4_aligned_TV'] for r in order_results)),infinite_threshold_nodes=infinity_count,positive_finite_word_prefix_mass_infinite_nodes=reachable_infinity_count,optimizer_success=sum(r['optimizer']['success'] for r in runtimes),max_final_gradient=max(r['optimizer']['gradient_max'] for r in runtimes),producer_seconds=sum(r['seconds'] for r in runtimes),source_hashes=hashes,dependencies=deps)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained');ax=axes[0,0]
    for n,color in ((4,'#147d92'),(8,'#ab5c1b')):
        rs=[r for r in rows if r['modes']==n];ax.scatter([float(r['K2_TV'][0]) for r in rs],[float(r['completion_TV'][0]) for r in rs],color=color,label=f'{n} active modes',s=40,alpha=.8)
    ax.plot([0,.2],[0,.2],'--',color='#999999');ax.set(xlabel='Official K2 finite-word physical TV',ylabel='Moment-completion physical TV',title='A | All 24 nonclassical cases are worse than K2',xlim=(0,.2),ylim=(0,.2));ax.legend(frameon=False,loc='upper left')
    ax=axes[0,1];selected=[rows[i] for i in (22,20,18)];trans=[.25,.75,1.]
    for key,label,marker in (('completion_TV','Moment completion','o'),('K2_TV','Official K2','s'),('K4_TV','Official K4','^'),('tree_TV','Selected tree','d')):ax.semilogy(trans,[float(r[key][0]) for r in selected],marker=marker,label=label)
    ax.set(xlabel='Transmission',ylabel='Physical-target TV',title='B | Strong eight-mode input, natural order',xticks=trans,ylim=(1e-4,.5));ax.legend(frameon=False,fontsize=9,loc='lower right');ax.grid(axis='y',alpha=.2)
    ax=axes[1,0];x=np.arange(24);ax.semilogy(x,[float(r['completion_TV'][0]) for r in rows],'o',ms=4,label='Physical-target TV',color='#147d92');ax.semilogy(x,[float(r['maxent_distance_upper']) for r in rows],'s',ms=4,label='Upper bound to true-moment max entropy',color='#ab5c1b');ax.set(xlabel='Frozen case index',ylabel='Total variation',title='C | Optimization accuracy does not imply target accuracy',ylim=(1e-8,2));ax.legend(frameon=False,loc='upper right',fontsize=9);ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];x=np.arange(2)
    for offset,key,label,color in ((-.24,'matching','Same-moment K2','#147d92'),(0,'completion','Moment completion','#ab5c1b'),(.24,'tree','Selected tree','#78818c')):
        values=[float(mechanism[i]['matching_TV'][0]) if key=='matching' else float(rows[i]['completion_TV' if key=='completion' else 'tree_TV'][0]) for i in (0,6)];ax.bar(x+offset,values,.23,label=label,color=color)
    ax.set(xticks=x,xticklabels=['Moderate four-mode','Strong four-mode'],ylabel='Physical-target TV',title='D | No loss: different high-order choices',ylim=(0,.24));ax.legend(frameon=False,fontsize=9,loc='upper left');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Round44: certified nonclassical counterexamples to universal maximum-entropy improvement',fontsize=14);fig.savefig(out/'nonclassical_completion.png',dpi=180);plt.close(fig);assert all(digest(ROOT/n)==s for n,s in hashes.items());save(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('source_hashes','dependencies')}))
if __name__=='__main__':main()
