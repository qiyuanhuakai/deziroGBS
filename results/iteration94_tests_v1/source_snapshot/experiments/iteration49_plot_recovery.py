"""Aggregate incremental144-prefix evidence and continuous-noise error controls."""
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
    out=ROOT/'results/iteration49_plot_recovery';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('iteration49_plot_recovery','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    validations={name:read(f'results/iteration49_{name}/validation.json') for name in ('prefix_validation','noise_validation','integrated_recovery')};assert all(r['count']==r['passed'] for r in validations.values());bounds=read('results/iteration49_prefix_validation/bounds.json');noise=read('results/iteration49_noise_validation/rows.json');integrated=read('results/iteration49_integrated_recovery/rows.json');target=read('results/iteration49_target/summary.json');nonmonotone=sum(noise[k+2]['TV'][0]>noise[k+1]['TV'][1] for k in range(0,36,3));summary=dict(scientific_checks=sum(r['count'] for r in validations.values()),full144_TV_lower=[float(r['full144_TV_lower']) for r in bounds],full144_shifted_TV_lower=[float(r['full144_shifted_TV_lower']) for r in bounds],prefix_gain_factors=[float(r['full144_TV_lower']/r['old_thirteen_mode_lower']) for r in bounds],all_prefix_bounds_stronger=all(r['strictly_strengthened'] for r in bounds),new_subsets=8192,reused_subsets=8191,recomputed_old_controls=13,target_timings=target['timings'],noise_cases=36,conditional_components=4068,noise_min_TV_lower=min(r['TV'][0] for r in noise),noise_max_TV_upper=max(r['TV'][1] for r in noise),noise_max_arithmetic=max(r['arithmetic_upper'] for r in noise),noise_max_selector=max(r['selector_upper'] for r in noise),noise_all_failure_zero=all(r['failure']==0 for r in noise),fine_grid_worse_cases=nonmonotone,grid_comparison_groups=12,initial_integral_verifier_failure_preserved=True,source_hashes=hashes,dependencies=deps)
    plt.rcParams.update({'font.size':10});fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True);xs=list(range(1,5));axes[0].semilogy(xs,[float(r['old_thirteen_mode_lower']) for r in bounds],'s--',color='#9f6954',label='13-bit prefix');axes[0].semilogy(xs,[float(r['full144_TV_lower']) for r in bounds],'o-',color='#316e91',label='14-bit prefix');axes[0].set(title='Actual144-mode full-law lower bounds',xlabel='Frugal cutoff K',ylabel='Certified TV lower bound');axes[0].set_xticks(xs);axes[0].legend(fontsize=9)
    chosen=noise[:3];x=[16,64,256];axes[1].loglog(x,[float(r['TV'][1]) for r in chosen],'o-',color='#358172',label='actual output TV');axes[1].loglog(x,[float(r['total_TV_upper']) for r in chosen],'s--',color='#9f6954',label='composed upper bound');axes[1].set(title='Vacuum input + real noise, sigma=1/4',xlabel='Interior displacement bins',ylabel='TV to continuous mixed Gaussian');axes[1].set_xticks(x,[str(v) for v in x]);axes[1].legend(fontsize=9)
    chosen=integrated[33:36];axes[2].plot(x,[float(r['actual_TV'][1]) for r in chosen],'o-',color='#316e91',label='discrete program TV');axes[2].axhline(float(chosen[0]['representation_bias'][0]),color='#9f6954',linestyle='--',label='exact finite-state continuous bias');axes[2].set_xscale('log',base=4);axes[2].set_xticks(x,[str(v) for v in x]);axes[2].set(title='Squeezed input: lambda=3/4, d=8, sigma=1/2',xlabel='Interior displacement bins',ylabel='Full-target TV / vacuum bias');axes[2].legend(fontsize=8)
    axes[1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for a in axes:a.grid(alpha=.2)
    fig.suptitle('Round 49: reuse preserves certificates; finer noise grids can remove helpful error cancellation',fontsize=13);fig.savefig(out/'incremental_and_noise.png',dpi=180);plt.close(fig);assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());summary.update(unchanged_sources=True,unchanged_dependencies=True);save(out/'summary.json',encode(summary));print(json.dumps({k:float(v) if hasattr(v,'denominator') and not isinstance(v,(int,bool)) else v for k,v in summary.items() if k not in ('source_hashes','dependencies')},indent=2))
if __name__=='__main__':main()
