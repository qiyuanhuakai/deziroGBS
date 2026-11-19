"""Show correlation, complete-target bias and representation improvements."""
import sys,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,NullLocator
from iteration22_external import digest,save
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration54_analysis';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('iteration54_analysis','iteration22_external','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    rows=read('results/iteration54_comparisons/rows.json');c=read('results/iteration54_comparisons/correlation.json');summary=read('results/iteration54_comparisons/summary.json');v=read('results/iteration54_comparisons/validation.json');assert v['count']==v['passed'];fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');blue='#19647e';orange='#e08e45';green='#538c55'
    a=axes[0,0];vals=[float(c['single_mode_vacuum_intervals'][0][0]),float(c['common_joint_vacuum'][0]),float(c['opposite_joint_vacuum'][0])];a.bar(range(3),vals,color=[green,blue,orange]);a.set_xticks(range(3),['Each single mode\n(both directions)','Joint vacuum\ncommon direction','Joint vacuum\nopposite direction']);a.set_ylim(0,1);a.set_ylabel('Exact Gaussian target probability');a.set_title('A. Identical marginals; different joint laws')
    for i,x in enumerate(vals):a.text(i,x+.02,f'{x:.6f}',ha='center')
    a.text(.5,.92,f"Full-law TV between directions: {float(c['full_law_TV'][0]):.6f}",transform=a.transAxes,ha='center',fontsize=10)
    a=axes[0,1];rr=[rows[i] for i in (23,24,25)];cells=[r['job']['cells'] for r in rr];a.loglog(cells,[float(r['actual_TV'][0]) for r in rr],'o-',color=orange,label='Program vs full Gaussian target');a.loglog(cells,[float(r['finite_continuous_TV'][0]) for r in rr],'o-',color=blue,label='Program vs finite continuous reference');a.axhline(float(rr[0]['reference_TV'][0]),color=green,linestyle='--',label='Reference vs full target');a.set_title('B. Complex heldout input, local dimension 6');a.set_xlabel('Normal-grid cells');a.set_ylabel('Complete two-mode TV');a.set_ylim(1e-6,2e-2);a.legend(fontsize=8,loc='upper right')
    a=axes[1,0];families=['common','opposite','quadrature','strong','complex_holdout'];x=np.arange(5)
    for offset,d,color in [(-.18,4,orange),(.18,6,blue)]:
        r=[next(r for r in rows if r['job']['family']==f and r['job']['dimension']==d and r['job']['cells']==64) for f in families];a.bar(x+offset,[float(v['actual_TV'][0]) for v in r],width=.34,color=color,label=f'Local dimension {d}')
    a.set_yscale('log');a.set_ylim(1e-4,.2);a.set_xticks(x,['Common','Opposite','Quadrature','Strong','Complex\nheldout']);a.set_ylabel('Complete-target TV at 64 cells');a.set_title('C. Larger finite input improves these families');a.legend(fontsize=9,loc='upper left')
    a=axes[1,1];x=np.arange(2);a.bar(x-.18,[summary['physical_refinements']['improved'],summary['finite_refinements']['improved']],width=.34,label='Improved',color=blue);a.bar(x+.18,[summary['physical_refinements']['worsened'],summary['finite_refinements']['worsened']],width=.34,label='Worsened',color=orange);a.set_xticks(x,['Full Gaussian target','Finite continuous reference']);a.set_ylim(0,18);a.set_ylabel('Count among 14 grid refinements');a.set_title('D. Grid convergence does not remove input bias');a.legend(fontsize=9,loc='upper left')
    axes[0,1].set_xscale('log',base=4);axes[0,1].set_xticks(cells);axes[0,1].xaxis.set_major_formatter(ScalarFormatter());axes[0,1].xaxis.set_minor_locator(NullLocator())
    for a in axes.flat:a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    fig.suptitle('Round 54 | Certified correlated and complex displacement on entangled inputs',fontsize=14);fig.savefig(out/'rank_one_noise.png',dpi=160);plt.close(fig);assert all(digest(ROOT/n)==s for n,s in hashes.items());assert all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps,scope='Two-mode rank-one noise only; no large-scale speed comparison.'))
if __name__=='__main__':main()
