"""Aggregate fixed-mixture holdout results, finite displacement and actual144 bounds."""
import sys,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration47_analysis';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('iteration47_analyze','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    validations={name:read(f'results/iteration47_{name}/validation.json') for name in ('input_validation','native_validation','prefix_validation','displacement_validation','mixture_validation')};assert all(r['count']==r['passed'] for r in validations.values());mixture=read('results/iteration47_mixture_validation/rows.json');bounds=read('results/iteration47_prefix_validation/bounds.json');displaced=read('results/iteration47_displacement_validation/comparisons.json');holdout=[r for r in mixture if r['job']['held_out']];summary=dict(scientific_checks=sum(r['count'] for r in validations.values()),mixtures=48,holdout_mixtures=len(holdout),holdout_material_better_both=sum(r['material_better_both'] for r in holdout),holdout_material_worse_best=sum(r['material_worse_best'] for r in holdout),historical_material_better_both=sum(r['material_better_both'] for r in mixture if not r['job']['held_out']),historical_material_worse_best=sum(r['material_worse_best'] for r in mixture if not r['job']['held_out']),prefix_depth=13,prefix_all_strictly_strengthened=all(r['strictly_strengthened'] for r in bounds),full144_TV_lower=[float(r['full144_TV_lower']) for r in bounds],full144_shifted_TV_lower=[float(r['full144_shifted_TV_lower']) for r in bounds],prefix_gain_factors=[float(r['full144_TV_lower']/r['old_twelve_mode_lower']) for r in bounds],displacement_new_controls=24,dimension2_regressions=24,displacement_max_TV_upper=max(float(r['TV'][1]) for r in displaced),displacement_max_target_width=max(r['maximum_target_width'] for r in displaced),source_hashes=hashes,dependencies=deps)
    plt.rcParams.update({'font.size':10});fig,axes=plt.subplots(1,3,figsize=(16,4.7),constrained_layout=True);xs=list(range(1,5));axes[0].semilogy(xs,[float(r['old_twelve_mode_lower']) for r in bounds],'s--',color='#8b6c56',label='12-bit prefix');axes[0].semilogy(xs,[float(r['full144_TV_lower']) for r in bounds],'o-',color='#345e99',label='13-bit prefix');axes[0].set(title='Actual 144-mode sampler',xlabel='Frugal cutoff K',ylabel='Certified full-law TV lower bound');axes[0].set_xticks(xs);axes[0].legend()
    selected=[r for r in holdout if r['job']['first']>=4 and r['job']['cutoff']>=3];labels=[f"{'moderate' if r['job']['first']==4 else 'strong'}\nK{r['job']['cutoff']}" for r in selected]
    for name,color,marker in [('forward_TV','#9b6a56','s'),('reverse_TV','#7f7698','^'),('mixture_TV','#358172','o')]:axes[1].plot(range(4),[float(r[name][0]) for r in selected],marker+'-',color=color,label=name.replace('_TV',''))
    axes[1].set(title='Held-out eight-mode fixed fair mixture',ylabel='Full-law TV');axes[1].set_xticks(range(4),labels);axes[1].legend(fontsize=8)
    for state,color in enumerate(('#345e99','#a7664d','#358172','#80619b','#9a8852','#4d8c99')):
        rows=[r for r in displaced if r['job']['state']==state];axes[2].plot(range(4),[float(r['TV'][1])*1e16 for r in rows],'o-',label=['Fock 2','Fock 3','NOON','Complex d4','Paired d6','Three-mode d3'][state],color=color)
    axes[2].set(title='Higher-occupation displacement certificate',xlabel='Fixed displacement setting',ylabel='TV upper bound (× 1e−16)');axes[2].legend(fontsize=7)
    for a in axes:a.grid(alpha=.2)
    fig.suptitle('Round 47: strict error evidence, fixed mixture limits, and finite-input controls',fontsize=13);fig.savefig(out/'three_directions.png',dpi=180);plt.close(fig);assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());summary.update(unchanged_sources=True,unchanged_dependencies=True);save(out/'summary.json',encode(summary));print(json.dumps({k:v for k,v in summary.items() if k not in ('source_hashes','dependencies','displacement_max_target_width')},indent=2))
if __name__=='__main__':main()
