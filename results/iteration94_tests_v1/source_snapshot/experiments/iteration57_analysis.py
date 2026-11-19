"""Show how weak the full-law upper bound is and diagnose rare-event bias."""
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
    out=ROOT/'results/iteration57_analysis';out.mkdir(exist_ok=False);names=['experiments/iteration57_analysis.py','experiments/iteration22_external.py','experiments/verify_iteration34.py'];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    rows=[r for r in read('results/iteration57_comparisons/rows.json') if r['modes']==144];rare=read('results/iteration57_rare_events/rows.json');physical=read('results/iteration57_target/target.json');fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained');x=np.arange(4);labels=[f'K{k}' for k in range(1,5)]
    ax=axs[0,0];ax.bar(x,[float(r['models']['uint64']['gap_from_one']) for r in rows],color='#216b83');ax.axhline(float(physical['transfer_upper']),color='#bf6535',linestyle='--',label='Existing target-transfer bound');ax.set_yscale('log');ax.set_xticks(x,labels);ax.set_ylabel('Gap delta (full-law upper = 1 - delta)');ax.set_ylim(1e-42,1e-4);ax.legend(loc='upper left',fontsize=9);ax.set_title('A. Strict upper bounds remain almost 1')
    ax=axs[0,1];ys=[float(r['target_over_program_event_interval'][0]) for r in rare];ax.bar(x,ys,color='#bf6535');ax.set_yscale('log');ax.set_xticks(x,labels);ax.set_ylabel('Target mass / sampler mass in five outputs');ax.set_title('B. Rare event is strongly underrepresented')
    ax=axs[1,0];ys=[float(r['conditional_TV_interval'][0]) for r in rare];ax.bar(x,ys,color='#216b83');ax.set_xticks(x,labels);ax.set_ylim(0,max(ys)*1.2);ax.set_ylabel('Conditional TV within the five-output event');ax.set_title('C. Conditional shape also differs')
    for i,y in enumerate(ys):ax.text(i,y+.007,f'{y:.4f}',ha='center')
    ax=axs[1,1];atoms=rare[-1]['atom_ratios'];xx=np.arange(5);ax.bar(xx-.18,[float(r['target_interval'][0]) for r in atoms],.36,label='Physical target',color='#bf6535');ax.bar(xx+.18,[float(r['program_probability']) for r in atoms],.36,label='K4 program',color='#216b83');ax.set_yscale('log');ax.set_xticks(xx,['Vacuum','Click 0','Click 47','Click 95','Click 143']);ax.set_ylabel('Probability of complete 144-bit output');ax.legend();ax.set_title('D. All five outputs were fixed in advance')
    fig.suptitle('Round 57 | Full-output overlap: rigorous upper bounds, limited usefulness',fontsize=17);fig.savefig(out/'complete_overlap.png',dpi=160);plt.close(fig);assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps))


if __name__=='__main__':main()
