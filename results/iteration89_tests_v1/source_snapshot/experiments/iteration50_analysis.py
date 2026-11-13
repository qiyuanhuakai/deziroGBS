"""Readable plots of validated precision changes, without timing claims."""
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
    out=ROOT/'results/iteration50_analysis';out.mkdir(exist_ok=False);names=['experiments/iteration50_analysis.py','experiments/iteration22_external.py','experiments/verify_iteration34.py'];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    m=read('results/iteration50_comparisons/mps.json');noise=read('results/iteration50_comparisons/noise.json');validation=read('results/iteration50_comparisons/validation.json');assert validation['count']==validation['passed']
    fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained');colors=['#19647e','#e08e45']
    for c,label,color in zip((4,6),('46 moderate / forward','46 strong / forward'),colors):
        r=[x for x in m if x['job']['source_round']==46 and x['job']['case']==c];ax[0,0].plot(range(4),[float(x['TV'][0]) for x in r],'o-',label=label,color=color)
    ax[0,0].set_xticks(range(4),['d3 / b8','d3 / b16','d4 / b8','d4 / b16']);ax[0,0].set_title('A. Larger local and bond budgets');ax[0,0].set_ylabel('Full output TV');ax[0,0].legend(fontsize=9)
    r=[x for x in m if x['job']['dimension']==4 and x['job']['rank']==16];x=np.arange(8);w=.36
    ax[0,1].bar(x-w/2,[float(a['TV'][0]) for a in r],w,label='MPS d4 / b16',color=colors[0]);ax[0,1].bar(x+w/2,[float(a['native']['4']['native_TV'][0]) for a in r],w,label='Official Frugal K4',color=colors[1]);ax[0,1].set_xticks(x,[f"{a['job']['source_round']}/{a['job']['case']}" for a in r]);ax[0,1].set_title('B. Same targets; unequal resource budgets');ax[0,1].set_xlabel('Input round / case (forward, reverse pairs)');ax[0,1].set_ylabel('Full output TV');ax[0,1].legend(fontsize=9)
    for axis,indices,title in [(ax[1,0],[0,1,2],'C. Vacuum, noise sigma=0.25, d=4'),(ax[1,1],[33,34,35],'D. Pairing=0.75, noise sigma=0.5, d=8')]:
        rr=[noise[i] for i in indices];cells=[r['job']['cells'] for r in rr]
        for field,label,color in [('old_TV','Tails mapped to zero',colors[1]),('new_TV','Tails clipped to boundary',colors[0])]:axis.plot(cells,[float(r[field][0]) for r in rr],'o-',label=label,color=color)
        axis.set_xscale('log',base=4);axis.set_xticks(cells);axis.xaxis.set_major_formatter(ScalarFormatter());axis.xaxis.set_minor_locator(NullLocator());axis.set_xlabel('Interior grid cells');axis.set_ylabel('Full output TV');axis.set_title(title);axis.legend(fontsize=9)
    ax[1,0].set_yscale('log');ax[1,1].ticklabel_format(axis='y',style='plain',useOffset=False)
    for a in ax.flat:a.grid(alpha=.2,axis='y');a.set_axisbelow(True)
    fig.suptitle('Round 50 | Certified small-system accuracy, not a speed benchmark',fontsize=15)
    fig.savefig(out/'budget_and_boundary.png',dpi=160);plt.close(fig)
    assert all(digest(ROOT/n)==s for n,s in hashes.items());assert all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps,scope='Displays interval lower endpoints; widths negligible at plot resolution; reused inputs'))
if __name__=='__main__':main()
