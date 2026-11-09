"""Show ideal convergence, actual bias and finite-word limits separately."""
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
    out=ROOT/'results/iteration51_analysis';out.mkdir(exist_ok=False);names=['experiments/iteration51_analysis.py','experiments/iteration22_external.py','experiments/verify_iteration34.py'];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    rows=read('results/iteration51_comparisons/rows.json');paired=read('results/iteration51_comparisons/paired.json');limit=read('results/iteration51_controls/ideal_limit.json')['sequence'];v=read('results/iteration51_comparisons/validation.json');assert v['count']==v['passed']
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');blue='#19647e';orange='#e08e45';green='#538c55'
    a=axes[0,0];seq=limit[:17];a.semilogy([r['k'] for r in seq],[float(r['ideal_TV_upper']) for r in seq],color=blue,label='Ideal coupling envelope');a.semilogy([r['k'] for r in seq],[float(r['word_grid_count_bound']) for r in seq],color=orange,label='53-bit counting bound only');a.axvspan(2,5,color=green,alpha=.12,label='Executed k range');a.set_title('A. Ideal limit and a separate precision cost');a.set_xlabel('k, with cells=4^k and radius=k+1');a.set_ylabel('TV bound');a.legend(fontsize=8)
    a=axes[0,1];r=[paired[i] for i in (0,1,2)];cells=[x['job']['cells'] for x in r]
    a.loglog(cells,[float(x['old_TV'][0]) for x in r],'o-',color=orange,label='Fixed radius 4');a.loglog(cells,[float(x['new_TV'][0]) for x in r],'o-',color=blue,label='Growing radius 3,4,5');a.set_title('B. Same cells need not improve actual TV');a.set_ylabel('Full Gaussian target TV');a.set_xlabel('Cells (vacuum, sigma=0.25, d4)');a.legend(fontsize=9)
    a=axes[1,0];rr=[rows[i] for i in (45,46,47,49)];cells=[x['job']['cells'] for x in rr]
    for field,label,color in [('actual_TV','Full Gaussian target TV',orange),('finite_continuous_TV','Finite continuous reference TV',blue)]:a.loglog(cells,[float(x[field][0]) for x in rr],'o-',label=label,color=color)
    a.set_title('C. Heldout pairing=0.625, sigma=0.375, d8');a.set_xlabel('Cells');a.set_ylabel('TV');a.legend(fontsize=9)
    a=axes[1,1];a.plot(cells,[float(x['actual_TV'][0]) for x in rr],'o-',color=orange,label='Full target error');a.axhline(float(rr[0]['representation_bias'][0]),color=green,linestyle='--',label='Finite input bias (signed vacuum)');a.set_xscale('log',base=4);a.set_title('D. Refinement approaches the input bias');a.set_xlabel('Cells (same heldout input)');a.set_ylabel('TV / positive vacuum bias');a.ticklabel_format(axis='y',style='plain',useOffset=False);a.legend(fontsize=8)
    for a,cells in ((axes[0,1],[16,64,256]),(axes[1,0],[16,64,256,1024]),(axes[1,1],[16,64,256,1024])):a.set_xscale('log',base=4);a.set_xticks(cells);a.xaxis.set_major_formatter(ScalarFormatter());a.xaxis.set_minor_locator(NullLocator())
    for a in axes.flat:a.grid(alpha=.2);a.set_axisbelow(True)
    fig.suptitle('Round 51 | Growing radius removes an ideal tail floor; state bias remains',fontsize=14);fig.savefig(out/'growing_radius.png',dpi=160);plt.close(fig)
    assert all(digest(ROOT/n)==s for n,s in hashes.items());assert all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps,scope='Panel A beyond k5 is analytic only; orange bound excludes probability rounding and state error. Others use certified complete one-mode laws.'))
if __name__=='__main__':main()
