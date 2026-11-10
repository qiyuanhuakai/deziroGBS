"""Visualize certified depth gains and sparse event concentration."""
import sys,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration22_external import digest,save
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration56_analysis';out.mkdir(exist_ok=False)
    names=['experiments/iteration56_analysis.py','experiments/iteration22_external.py','experiments/verify_iteration34.py'];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    path='results/iteration56_comparisons/rows.json';rows=decode(json.loads((ROOT/path).read_text()));deps={path:digest(ROOT/path)}
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained');x=np.arange(4);ax=axs[0,0]
    ax.bar(x-.18,[float(r['old_lower']) for r in rows],.36,label='15-bit bound',color='#c7793c');ax.bar(x+.18,[float(r['new_lower']) for r in rows],.36,label='16-bit bound',color='#216b83');ax.set_yscale('log');ax.set_xticks(x,[f'K{r["cutoff"]}' for r in rows]);ax.set_ylabel('Full 144-mode TV lower bound');ax.legend();ax.set_title('A. Certified lower bounds, same sampler')
    ax=axs[0,1];ys=[float(r['percent_gain']) for r in rows];ax.bar(x,ys,color='#216b83');ax.set_xticks(x,[f'K{r["cutoff"]}' for r in rows]);ax.set_ylabel('Increase over 15-bit bound (%)');ax.set_ylim(0,max(ys)*1.25)
    for i,y in enumerate(ys):ax.text(i,y+max(ys)*.025,f'{y:.2f}%',ha='center')
    ax.set_title('B. Gain from one additional observed bit')
    for ax,side,title in ((axs[1,0],'program_excess','C. Events where sampler exceeds target'),(axs[1,1],'target_excess','D. Events where target exceeds sampler')):
        for r in rows:
            ev=[e for e in r['sparse'] if e['side']==side];xs=[e['atoms'] for e in ev];ys=[float(e['fraction_of_full_prefix_lower'])*100 for e in ev];ax.plot(xs,ys,'o-',label=f'K{r["cutoff"]}')
        ax.set_xscale('log');ax.set_xlabel('Number of 16-bit patterns in cylinder event');ax.set_ylabel('Event lower bound / full-prefix lower (%)');ax.set_ylim(0,105);ax.grid(alpha=.2);ax.legend();ax.set_title(title)
    fig.suptitle('Round 56 | Actual 144-mode data: 16-bit bounds and event witnesses',fontsize=17)
    fig.savefig(out/'prefix16_events.png',dpi=160);plt.close(fig)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items())
    save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps))


if __name__=='__main__':main()
