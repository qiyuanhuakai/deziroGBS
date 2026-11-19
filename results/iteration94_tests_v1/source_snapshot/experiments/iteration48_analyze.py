"""Aggregate full-input certificates and fixed-mixture error geometry."""
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
    out=ROOT/'results/iteration48_analysis';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('iteration48_analyze','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    validations={name:read(f'results/iteration48_{name}/validation.json') for name in ('gaussian_validation','mixture_validation','controls')};assert all(r['count']==r['passed'] for r in validations.values());gaussian=read('results/iteration48_gaussian_validation/comparisons.json');mixture=read('results/iteration48_mixture_validation/rows.json');selected=[r for r in mixture if r['job']['held_out'] and r['job']['first']>=4 and r['job']['cutoff']>=3];full=[r for r in gaussian if r['job']['bond']==r['job']['dimension']];summary=dict(scientific_checks=sum(r['count'] for r in validations.values()),Gaussian_cases=len(gaussian),Gaussian_constructions=30,all_failure_zero=all(r['failure']==0 for r in gaussian),arithmetic_TV_max_upper=max(r['arithmetic_TV'][1] for r in gaussian),Gaussian_TV_min_lower=min(r['Gaussian_TV'][0] for r in gaussian),Gaussian_TV_max_upper=max(r['Gaussian_TV'][1] for r in gaussian),full_budget_compression_trace_max=max(r['compression_trace'][1] for r in full),max_finite_target_width=max(r['maximum_target_width'] for r in gaussian),all_full_TV_bounded=all(r['Gaussian_TV'][1]<=r['total_TV_upper'] for r in gaussian),mixture_reanalyses=len(mixture),selected_mixture_geometry=selected,source_hashes=hashes,dependencies=deps)
    plt.rcParams.update({'font.size':10});fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True);xs=[4,6,8]
    for bond,color in [(2,'#9f6354'),('full','#316e91')]:
        rows=[r for r in gaussian if r['job']['target']==3 and r['job']['displacement']==1 and (r['job']['bond']==2 if bond==2 else r['job']['bond']==r['job']['dimension'])];label='bond 2' if bond==2 else 'bond d'
        axes[0].semilogy(xs,[float(r['Gaussian_TV'][1]) for r in rows],'o-',color=color,label=label+' actual TV');axes[0].semilogy(xs,[float(r['total_TV_upper']) for r in rows],'s--',color=color,label=label+' certified upper')
    axes[0].set(title='Strong paired Gaussian, fixed displacement',xlabel='Local Fock dimension d',ylabel='Full Gaussian output TV');axes[0].set_xticks(xs);axes[0].legend(fontsize=8)
    for s,color in enumerate(('#345e99','#a7664d','#358172','#80619b','#9a8852','#4d8c99')):
        rows=[r for r in full if r['job']['target']==s and r['job']['displacement']==1];axes[1].semilogy(xs,[float(r['Gaussian_TV'][1]) for r in rows],'o-',color=color,label=['single moderate','single strong','paired moderate','paired strong','complex moderate','complex strong'][s])
    axes[1].set(title='Complete bond budget, six Gaussian inputs',xlabel='Local Fock dimension d',ylabel='Direct full-target TV upper');axes[1].set_xticks(xs);axes[1].legend(fontsize=8)
    xx=list(range(4));axes[2].bar([x-.18 for x in xx],[float(r['cancellation'][0])*1e3 for r in selected],width=.36,color='#358172',label='cancellation G');axes[2].bar([x+.18 for x in xx],[float(r['branch_imbalance'][0])*1e3 for r in selected],width=.36,color='#a7664d',label='branch gap J');axes[2].set_xticks(xx,['moderate\nK3','moderate\nK4','strong\nK3','strong\nK4']);axes[2].set(title='Fixed mixture improves best branch iff G > J',ylabel='TV contribution (x 1e-3)');axes[2].legend(fontsize=8)
    for a in axes:a.grid(alpha=.2,axis='y')
    fig.suptitle('Round 48: infinite-input certificates expose budget limits; mixture gains need sufficient cancellation',fontsize=13);fig.savefig(out/'gaussian_bridge.png',dpi=180);plt.close(fig);assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());summary.update(unchanged_sources=True,unchanged_dependencies=True);save(out/'summary.json',encode(summary));print(json.dumps({k:float(v) if hasattr(v,'denominator') and not isinstance(v,(int,bool)) else v for k,v in summary.items() if k not in ('source_hashes','dependencies','selected_mixture_geometry')},indent=2))
if __name__=='__main__':main()
