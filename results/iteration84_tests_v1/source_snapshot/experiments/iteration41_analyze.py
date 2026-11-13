"""Summarize certified complete laws and produce a scope-labeled research figure."""
import csv,json,shutil
from pathlib import Path
from fractions import Fraction as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'results/iteration41_analysis';out.mkdir(exist_ok=False);source='experiments/iteration41_analyze.py';shutil.copy2(ROOT/source,out/Path(source).name);deps={}
    def read(name):
        deps[name]=digest(ROOT/name);return decode(json.loads((ROOT/name).read_text()))
    for name in ('iteration41_validation','iteration41_symbolic_validation'):
        v=read(f'results/{name}/validation.json');assert v['passed']==v['count']
    rows=read('results/iteration41_panel/rows.json');records=[read(f'results/iteration41_panel/case_{i:02d}.audit.json') for i in range(44)];assert all(x['status']=='completed' for x in rows)
    validation=read('results/iteration41_validation/rows.json');assert next(x for x in validation if x['sequence']==21)['clamp_activations_over_all_terminal_evaluations']==0
    four=records[8];eight=records[21];event=lambda law:sum((mass for mask,mass in enumerate(law) if (mask&15).bit_count()==3),F(0))
    event_probability=dict(target=event(four['target']),real_reference=event(four['controls']['sampler']),native_continuous=event(four['native']['continuous']),native_uint64=event(four['native']['uint64']))
    assert event_probability['target']==F(512,3315) and event_probability['real_reference']==0
    event_lower=event_probability['target']-event_probability['native_uint64'];assert event_lower>F(1544,10000)
    assert eight['approximation_half_L1']==F(32768,31640625)
    order=[]
    def align(law,permutation):
        result=[F(0)]*len(law);n=len(permutation)
        for mask,mass in enumerate(law):
            dest=(mask>>n)<<n
            for j,k in enumerate(permutation):dest|=((mask>>j)&1)<<k
            result[dest]=mass
        return result
    for i in (30,32,34,36):
        a,b=records[i],records[i+1];pa,pb=a['job']['order'],b['job']['order'];assert align(a['target'],pa)==align(b['target'],pb)
        distance=lambda x,y:sum((abs(u-v) for u,v in zip(x,y)),F(0))/2
        order.append(dict(first=i,second=i+1,active=a['job']['active'],intensity=a['job']['intensity'],native_TV_between_orders=distance(align(a['native']['uint64'],pa),align(b['native']['uint64'],pb)),real_TV_between_orders=distance(align(a['controls']['sampler'],pa),align(b['controls']['sampler'],pb)),first_target_TV=a['TV']['uint64'],second_target_TV=b['TV']['uint64']))
    mechanism=dict(four_mode_event=event_probability,four_mode_event_TV_lower=event_lower,four_mode_native_full_TV=four['TV']['uint64'],four_mode_real_TV=four['real_sampler_TV'],eight_mode_factorization_TV=eight['approximation_half_L1'],eight_mode_real_target_TV=eight['real_sampler_TV'],order_comparisons=order,max_native_to_real_TV=max(r.get('native_to_physical_real_sampler_TV',F(0)) for r in records),max_native_vacuum_padding_leak=max(r['vacuum_padding_leak']['uint64'] for r in records));save(out/'mechanisms.json',encode(mechanism))
    with (out/'panel.csv').open('w',newline='') as f:
        fields=['sequence','active','intensity','family','cutoff','TV_uint64','TV_continuous','real_sampler_TV','approximation_half_L1','native_to_real_TV','saturated_reachable'];writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,2,figsize=(13.5,9));blue='#1864ab';orange='#e67700';green='#2b8a3e'
    ax=axes[0,0];x=list(range(5));count=lambda law:[sum(float(v) for m,v in enumerate(law) if (m&15).bit_count()==k) for k in x];target=count(four['target']);native=count(four['native']['uint64'])
    ax.bar([k-.18 for k in x],target,.36,label='Physical target',color=blue);ax.bar([k+.18 for k in x],native,.36,label='Official K2',color=orange);ax.set(xlabel='Clicks among 4 active modes',ylabel='Probability',title='Four-mode witness: a = 4');ax.set_xticks(x);ax.legend();ax.annotate('Target P(N=3) = 0.15445\nOfficial finite-word P < 1e-18',xy=(3,.15445),xytext=(.3,.48),arrowprops=dict(arrowstyle='->'),fontsize=9)
    ax=axes[0,1]
    for a,color in (('1/4',blue),('4',orange)):
        for cutoff,style in ((2,'-o'),(4,'--s')):
            chosen=[r for r in rows if r['family']=='equal' and r['active'] in (4,8,10) and r['intensity']==a and r['cutoff']==cutoff];ax.plot([r['active'] for r in chosen],[r['TV_uint64'] for r in chosen],style,color=color,label=f'a={a}, K={cutoff}')
    ax.set(xlabel='Active modes (10 modes including vacuum padding)',ylabel='Full output TV',title='Higher cutoff need not improve TV');ax.legend(fontsize=8);ax.set_ylim(-.02,.86)
    ax=axes[1,0];chosen=[r for r in rows if r['family']=='equal' and r['intensity']=='1/4' and r['cutoff']==2];ax.bar([r['active'] for r in chosen],[r['approximation_half_L1'] for r in chosen],color=green);ax.set(xlabel='Active modes',ylabel='Half L1: truncation vs unclamped recurrence',title='Additional marginal factorization error');ax.set_xticks([2,4,6,7,8,10]);ax.text(.04,.82,'Symbolic equality through 7 modes\n8 modes: 16 missing perfect matchings\nAt 8, a=1/4: both laws nonnegative; no clamps',transform=ax.transAxes,fontsize=9)
    ax=axes[1,1];selected=order[-1];ax.bar(['Original order','Reversed order'],[float(selected['first_target_TV']),float(selected['second_target_TV'])],color=[blue,orange]);ax.set(ylabel='Full output TV to the same physical target',title='Weighted 8-mode input, a = 4');ax.text(.04,.85,f'Between-order TV = {float(selected["native_TV_between_orders"]):.6f}\nLabels aligned before comparison',transform=ax.transAxes,fontsize=9);ax.set_ylim(0,.14)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('GBS iteration 41: separating truncation, factorization and clipping',fontsize=15);fig.text(.5,.015,'Exact full laws on an easily sampled classical Gaussian family. No 144-mode bound, speed ranking, or Jiuzhang simulation claim.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(out/'ablation_and_output_error.png',dpi=180);plt.close(fig)
    assert event_probability['native_uint64']<F(1,10**18)
    summary=dict(cases=44,completed=44,source_hashes={source:digest(ROOT/source)},dependencies=deps,four_event={k:float(v) for k,v in event_probability.items()},four_event_TV_lower=float(event_lower),four_full_TV=float(four['TV']['uint64']),eight_factorization_TV=float(eight['approximation_half_L1']),order_comparisons=[{k:float(v) if isinstance(v,F) else v for k,v in r.items()} for r in order],max_native_to_real_TV=float(mechanism['max_native_to_real_TV']),max_vacuum_padding_leak=float(mechanism['max_native_vacuum_padding_leak']));save(out/'summary.json',summary);print(json.dumps(summary,default=str),flush=True)
if __name__=='__main__':main()
