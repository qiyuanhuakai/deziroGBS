"""Projection references, nonmonotone output error and observed audit costs."""
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
    out=ROOT/'results/iteration52_analysis';out.mkdir(exist_ok=False);names=['experiments/iteration52_analysis.py','experiments/iteration22_external.py','experiments/verify_iteration34.py'];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    rows=read('results/iteration52_comparisons/rows.json');timings=read('results/iteration52_comparisons/timings.json');v=read('results/iteration52_comparisons/validation.json');assert v['count']==v['passed'];table={(r['run_round'],r['job']['source_round'],r['job']['case'],r['job']['dimension'],r['job']['rank']):r for r in rows};keys=[(46,4),(46,6),(47,4),(47,6)];labels=['46 moderate','46 strong','47 moderate','47 strong'];blue='#19647e';orange='#e08e45';green='#538c55';gray='#7f7890'
    fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained');x=np.arange(4);width=.2;low=[table[50,it,c,4,16] for it,c in keys];high=[table[52,it,c,4,32] for it,c in keys]
    for off,vals,label,color in [(-1.5,[float(r['TV'][0]) for r in low],'MPS d4 / b16',orange),(-.5,[float(r['TV'][0]) for r in high],'MPS d4 / b32',blue),(.5,[float(r['ideal_box_TV'][0]) for r in high],'Ideal local d4 projection',green),(1.5,[float(r['native']['4'][0]) for r in high],'Official Frugal K4',gray)]:ax[0,0].bar(x+off*width,vals,width,label=label,color=color)
    ax[0,0].set_xticks(x,labels);ax[0,0].set_title('A. Same physical target, different approximations');ax[0,0].set_ylabel('Full output TV');ax[0,0].legend(fontsize=8)
    rr=[table[50,46,4,3,16],table[52,46,4,3,32]];ax[0,1].plot([16,32],[float(r['TV'][0]) for r in rr],'o-',color=blue,label='Actual MPS word law');ax[0,1].axhline(float(rr[0]['ideal_box_TV'][0]),color=green,linestyle='--',label='Ideal d3 projection');ax[0,1].set_xticks([16,32]);ax[0,1].set_title('B. More bond budget can raise physical TV');ax[0,1].set_xlabel('Bond budget (46 moderate, forward, d3)');ax[0,1].set_ylabel('Full output TV');ax[0,1].ticklabel_format(axis='y',style='plain',useOffset=False);ax[0,1].legend(fontsize=8)
    ax[1,0].bar(x-.18,[float(r['state_vs_box_TV']) for r in low],.36,label='Bond16',color=orange);ax[1,0].bar(x+.18,[float(r['state_vs_box_TV']) for r in high],.36,label='Bond32',color=blue);ax[1,0].set_yscale('log');ax[1,0].set_xticks(x,labels);ax[1,0].set_title('C. Stored MPS approaches ideal local projection');ax[1,0].set_ylabel('TV between the two finite output laws');ax[1,0].legend(fontsize=9)
    chosen=[next(r for r in timings if r['job']==dict(source_round=it,case=c,dimension=4,rank=b)) for it,c in keys for b in (16,32)];bottom=np.zeros(8)
    phases=[('construction','Construction',green),('exact_state','Exact state audit',orange),('full_tree','Complete program tree',blue),('stored_result','Errors / compression / result write',gray)]
    for key,label,color in phases:
        vals=np.array([r['phase_seconds'][key] for r in chosen]);ax[1,1].bar(range(8),vals,bottom=bottom,label=label,color=color);bottom+=vals
    rest=np.array([r['phase_seconds']['tensor_storage']+r['phase_seconds']['public_replay'] for r in chosen]);ax[1,1].bar(range(8),rest,bottom=bottom,label='Tensor I/O + public replay',color='#b9bdc5');ax[1,1].set_xticks(range(8),[f'{it}/{c}\nb{b}' for it,c in keys for b in (16,32)]);ax[1,1].set_title('D. Single-run cost observations; concurrent work');ax[1,1].set_ylabel('Seconds (producer only)');ax[1,1].legend(fontsize=7,loc='upper left')
    for a in ax.flat:a.grid(alpha=.2,axis='y');a.set_axisbelow(True)
    fig.suptitle('Round 52 | Local projection is a reference, not a universal TV lower bound',fontsize=14);fig.savefig(out/'projection_and_budget.png',dpi=160);plt.close(fig)
    assert all(digest(ROOT/n)==s for n,s in hashes.items());assert all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps,scope='All accuracy panels use certified exact laws; timing observations are not isolated or matched-quality benchmarks.'))
if __name__=='__main__':main()
