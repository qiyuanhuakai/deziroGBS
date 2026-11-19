"""Same-target accuracy and mode-order comparisons; resource budgets differ."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def aligned_reverse(law,n):
    size=1<<n;out=[F(0)]*len(law)
    for mask,p in enumerate(law):
        if len(law)==size+1 and mask==size:out[mask]=p;continue
        rev=sum(((mask>>j)&1)<<(n-1-j) for j in range(n));out[(mask&~(size-1))|rev]=p
    return out

def main():
    out=ROOT/'results/iteration46_analysis';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('iteration46_analyze','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    validations={name:read(f'results/iteration46_{name}/validation.json') for name in ('input_validation','mps_validation','native_validation','hierarchy_validation','displacement_validation')};assert all(v['count']==v['passed'] for v in validations.values());mps=[read(f'results/iteration46_mps/case_{i:02d}.json') for i in range(32)];hierarchy=read('results/iteration46_hierarchy_validation/rows.json');native={(i,k):read(f'results/iteration46_native/case_{i:02d}_k{k}.audit.json') for i in range(8) for k in (2,3,4)};hier={};rows=[]
    for r in hierarchy:hier[r['input'],r['order']]=r
    for i in range(8):
        source=read(f'results/iteration46_inputs/case_{i:02d}.json');job=source['job'];selected=mps[i*4+3];values={'MPS_max_budget':selected['sampler_physical_TV_interval'],'maxent_degree2':hier[i,2]['true_maxent_TV'],'maxent_degree3':hier[i,3]['true_maxent_TV']}
        for k in (2,3,4):values[f'Frugal_K{k}']=native[i,k]['physical_TV_intervals']['uint64']
        if job['modes']==4:values['maxent_full_information']=hier[i,4]['true_maxent_TV']
        comparisons=[]
        for a in values:
            for b in values:
                if a!=b and values[a][1]<values[b][0]:comparisons.append(dict(better=a,worse=b,certified_gap=values[b][0]-values[a][1]))
        rows.append(dict(case=i,job=job,values=values,strict_comparisons=comparisons));print('same target',i,{k:float(v[0]) for k,v in values.items()},flush=True)
    orders=[]
    for i in range(0,8,2):
        n=rows[i]['job']['modes']
        for slot in range(4):
            a=mps[i*4+slot]['sampling']['word_law'];b=aligned_reverse(mps[(i+1)*4+slot]['sampling']['word_law'],n);orders.append(dict(case_pair=[i,i+1],method='MPS',configuration=mps[i*4+slot]['job'],aligned_TV=sum(abs(x-y) for x,y in zip(a,b))/2))
        for k in (2,3,4):
            a=native[i,k]['laws']['uint64']['distribution'];b=aligned_reverse(native[i+1,k]['laws']['uint64']['distribution'],n);orders.append(dict(case_pair=[i,i+1],method=f'Frugal_K{k}',aligned_TV=sum(abs(x-y) for x,y in zip(a,b))/2))
        for k in ((2,3,4) if n==4 else (2,3)):
            a=read(f"results/iteration46_hierarchy/case_{hier[i,k]['case']:02d}.json")['projection']['law'];b=aligned_reverse(read(f"results/iteration46_hierarchy/case_{hier[i+1,k]['case']:02d}.json")['projection']['law'],n);gap=sum(abs(x-y) for x,y in zip(a,b))/2;upper=hier[i,k]['distance_to_true_maxent_upper']+hier[i+1,k]['distance_to_true_maxent_upper'];assert gap<=upper;orders.append(dict(case_pair=[i,i+1],method=f'maxent_degree{k}',aligned_TV=gap,permutation_equivariance_certificate_upper=upper))
    displacement=read('results/iteration46_displacement_validation/comparisons.json');improvements=read('results/iteration46_hierarchy_validation/comparisons.json');summary=dict(shared_inputs=8,mps_configurations=32,native_configurations=24,hierarchy_configurations=20,displacement_controls=24,scientific_checks=sum(v['count'] for v in validations.values()),degree3_improves_degree2=sum(r['degree3_certified_better'] for r in improvements),mps_max_budget_better_K2=sum(r['values']['MPS_max_budget'][1]<r['values']['Frugal_K2'][0] for r in rows),mps_max_budget_better_K4=sum(r['values']['MPS_max_budget'][1]<r['values']['Frugal_K4'][0] for r in rows),degree3_worse_K2=sum(r['values']['maxent_degree3'][0]>r['values']['Frugal_K2'][1] for r in rows),native_K3_worse_K2=sum(r['values']['Frugal_K3'][0]>r['values']['Frugal_K2'][1] for r in rows),max_order_TV={method:max(float(r['aligned_TV']) for r in orders if r['method']==method) for method in sorted(set(r['method'] for r in orders))},displacement_TV_upper=max(float(r['TV'][1]) for r in displacement),displacement_exact_max_width=max(r['maximum_target_width'] for r in displacement),source_hashes=hashes,dependencies=deps)
    save(out/'comparisons.json',encode(rows));save(out/'order_comparisons.json',encode(orders));plt.rcParams.update({'font.size':11});fig,axes=plt.subplots(1,2,figsize=(13,4.7),constrained_layout=True)
    for name,color,marker in [('MPS_max_budget','#37629c','o'),('Frugal_K2','#ae6650','s'),('Frugal_K4','#358572','^'),('maxent_degree3','#86589a','x')]:axes[0].semilogy(range(8),[float(r['values'][name][0]) for r in rows],marker+'-',color=color,label=name.replace('_',' '))
    axes[0].axvline(3.5,color='gray',linestyle=':');axes[0].set(title='Same physical target; different resource budgets',xlabel='Input index (0–3: four modes; 4–7: eight)',ylabel='Full-law total variation');axes[0].legend(fontsize=8)
    for state,color in [(0,'#37629c'),(1,'#ae6650'),(2,'#358572'),(3,'#86589a')]:
        r=[x for x in displacement if x['job']['state']==state];axes[1].plot(range(6),[float(x['TV'][1])*1e16 for x in r],'o-',color=color,label=['Vacuum','Single photon','Two-mode Bell','Two-mode complex'][state])
    axes[1].set(title='Displaced finite-input measurement audit',xlabel='Fixed displacement setting',ylabel='Certified TV upper bound (× 1e−16)');axes[1].legend(fontsize=8)
    for a in axes:a.grid(alpha=.2)
    fig.suptitle('Round 46: matched targets and nonzero displacement; no throughput ranking',fontsize=13);fig.savefig(out/'matched_targets.png',dpi=180);plt.close(fig);assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());summary.update(unchanged_sources=True,unchanged_dependencies=True);save(out/'summary.json',encode(summary));print(json.dumps({k:v for k,v in summary.items() if k not in ('source_hashes','dependencies','displacement_exact_max_width')},indent=2))
if __name__=='__main__':main()
