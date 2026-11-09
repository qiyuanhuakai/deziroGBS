"""Analyze actual144-prefix bounds and exact parity-repair feasibility."""
import json,csv,shutil
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
    out=ROOT/'results/iteration42_analysis';out.mkdir(exist_ok=False);source='experiments/iteration42_analyze.py';shutil.copy2(ROOT/source,out/Path(source).name);deps={}
    def read(name):deps[name]=digest(ROOT/name);return decode(json.loads((ROOT/name).read_text()))
    for folder in ('iteration42_prefix_recovery','iteration42_repair_recovery'):
        r=read(f'results/{folder}/validation.json');assert r['count']==r['passed']
    prefix=read('results/iteration42_prefix/rows.json');global_rows=[r for r in prefix if r['modes']==144];depths=read('results/iteration42_prefix_recovery/depth_bounds.json');old=read('results/iteration42_repair/rows.json');fresh=read('results/iteration42_fresh/rows.json')
    dual=read('results/iteration42_repair_recovery/nonextendability.json');e,o=dual['even_atom'],dual['odd_atom'];terms=dual['Walsh_terms'];gap=-dual['negative_expectation'];weight=sum((abs(t['coefficient']) for t in terms if t['subset_mask']),F(0));assert weight==F(127,128) and gap>0
    for atom in range(256):
        v=sum((t['coefficient']*(-1)**(t['subset_mask']&atom).bit_count() for t in terms),F(0));assert v==int(atom==e)+int(atom==o)
    forced=gap/(2*weight);assert forced==64*gap/127
    minima=dual['proper_marginal_minima'];assert len(minima)==254 and min(r['minimum'] for r in minima)>=0
    bound=dict(dual_negative_gap=gap,nonconstant_coefficient_absolute_sum=weight,any_probability_law_max_proper_marginal_TV_lower=forced,pointwise_indicator_checks=256,proper_marginals=254,scope='At least one proper marginal of any nonnegative completion must change by this TV; proper reference marginals are all valid, but jointly incompatible');save(out/'dual_bound.json',encode(bound))
    counts={}
    for name,rows in (('published_exploration',old),('new_panel',fresh)):
        counts[name]=dict(cases=len(rows),feasible=sum(r['feasible'] for r in rows),infeasible=sum(not r['feasible'] for r in rows),improved=sum(r['feasible'] and r['corrected_TV']<r['base_TV'] for r in rows),worsened=sum(r['feasible'] and r['corrected_TV']>r['base_TV'] for r in rows))
    exact=[]
    for k in range(1,5):
        r=read(f'results/iteration42_prefix/global_{k}.audit.json');exact.append(dict(cutoff=k,full144_BF_TV_lower=r['full144_TV_lower'],full144_shifted_TV_lower=r['full144_shifted_TV_lower'],prefix_TV_width=r['prefix_TV_intervals']['uint64'][1]-r['prefix_TV_intervals']['uint64'][0],event_TV_lower=r['event']['TV_lower']))
    save(out/'global_bounds.json',encode(exact))
    for filename,rows,fields in (('global_bounds.csv',global_rows,['cutoff','prefix_TV_lower','shifted_lower','event_lower','seconds']),('repair_fresh.csv',fresh,['case','family','intensity','feasible','base_TV','corrected_TV','oracle_TV','negative_proper_order'])):
        with (out/filename).open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,2,figsize=(13.5,9));colors=['#1864ab','#e67700','#2b8a3e','#862e9c']
    ax=axes[0,0];x=[r['cutoff'] for r in global_rows];ax.bar([k-.17 for k in x],[r['prefix_TV_lower'] for r in global_rows],.34,label='Exact B,F target',color=colors[0]);ax.bar([k+.17 for k in x],[r['shifted_lower'] for r in global_rows],.34,label='Shifted physical target',color=colors[1]);ax.set_yscale('log');ax.set_xticks(x);ax.set(xlabel='Official cumulant cutoff K',ylabel='Certified full144-output TV lower bound',title='Actual144-mode program: first10 bits');ax.legend(fontsize=9)
    ax=axes[0,1]
    for k,color in zip(range(1,5),colors):
        chosen=[r for r in depths if r['cutoff']==k];ax.plot([r['depth'] for r in chosen],[float(r['shifted_lower']) if r['shifted_lower']>0 else float('nan') for r in chosen],'-o',markersize=3,color=color,label=f'K={k}')
    ax.set_yscale('log');ax.set(xlabel='Observed prefix length',ylabel='Full144-output TV lower bound (shifted)',title='Information from longer prefixes');ax.legend(fontsize=9)
    ax=axes[1,0];selected=[r for r in fresh if r['feasible']];x=list(range(len(selected)));ax.bar(x,[r['corrected_TV']-r['base_TV'] for r in selected],color=[colors[1] if r['corrected_TV']>r['base_TV'] else colors[2] for r in selected]);ax.set_yscale('symlog',linthresh=1e-9);ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,[str(r['case']) for r in selected]);ax.set(xlabel='New feasible reference case',ylabel='TV after correction minus TV before',title='Completing missing terms is not monotone');ax.text(.03,.06,'15 new cases: 5 feasible, 10 infeasible\n2 improve slightly; 3 worsen (order pairs included)',transform=ax.transAxes,fontsize=9)
    ax=axes[1,1];r=read('results/iteration42_fresh/case_01.json');interval=r['correction']['interval'];ax.scatter([float(interval['upper']),float(interval['lower'])],[0,0],s=75,color=[colors[1],colors[0]]);ax.annotate('Required upper bound U',xy=(float(interval['upper']),0),xytext=(-.009,.6),arrowprops=dict(arrowstyle='->'),fontsize=9);ax.annotate('Required lower bound L',xy=(float(interval['lower']),0),xytext=(-.004,-.6),arrowprops=dict(arrowstyle='->'),fontsize=9);ax.set(xlim=(-.011,.001),ylim=(-1,1),xlabel='Parity correction delta',title='All proper marginals valid; no joint completion');ax.set_yticks([]);ax.text(.03,.86,f'U < L: empty feasible interval\nSome proper marginal must change by TV >= {float(forced):.6f}',transform=ax.transAxes,fontsize=9)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('GBS iteration42: global lower bounds and limits of local repair',fontsize=15);fig.text(.5,.012,'Prefix lower bounds are not full-law upper bounds. Repair panels use exact small-table references; oracle optimization knows the full target.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(out/'prefix_bounds_and_repair.png',dpi=180);plt.close(fig)
    summary=dict(source_hashes={source:digest(ROOT/source)},dependencies=deps,global_bounds=[{k:float(v) if isinstance(v,F) else v for k,v in r.items()} for r in exact],repair_counts=counts,dual_gap=float(gap),forced_proper_marginal_TV_lower=float(forced),recovered_verifier_issues=['Serialized absolute path representation mismatch','Tuple/list representation mismatch in marginal certificate'])
    save(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='dependencies'}),flush=True)
if __name__=='__main__':main()
