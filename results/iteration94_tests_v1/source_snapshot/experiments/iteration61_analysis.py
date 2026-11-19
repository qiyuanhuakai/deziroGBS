"""Plot verified fixed-set order effects without inferring full144 error rankings."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
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
    out=ROOT/'results/iteration61_analysis';out.mkdir(exist_ok=False);names=['experiments/iteration61_analysis.py','experiments/iteration22_external.py','experiments/verify_iteration34.py'];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    validation=read('results/iteration61_order_validation/validation.json');assert validation['count']==validation['passed'];groups=read('results/iteration61_order_validation/comparisons.json');main=[g for g in groups if g['modes']==144];local=next(g for g in groups if g['modes']==10 and g['cutoff']==4);colors=['#94a3b8','#2563eb','#f59e0b','#10b981'];labels=['Original','Reverse','Even / odd','Uniform mixture'];plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.labelsize':11})
    fig,axs=plt.subplots(2,2,figsize=(14.5,10),layout='constrained');fig.suptitle('Iteration 61 | Fixed physical modes, different sampling orders',fontsize=18)
    for j,(name,color) in enumerate(zip(labels,colors)):
        axs[0,0].plot(range(1,5),[float(g['variants'][j]['TV_interval'][0]) for g in main],marker='o',label=name,color=color)
    axs[0,0].set(yscale='log',xticks=range(1,5),xlabel='Truncation order K',ylabel='16-bit marginal TV',title='A  Actual144 sampler, same 16-mode target');axs[0,0].legend(fontsize=9);axs[0,0].grid(alpha=.2)
    x=np.arange(3)
    for j in range(1,4):
        values=[100*(float(g['variants'][j]['TV_interval'][0]/g['variants'][0]['TV_interval'][0])-1) for g in main[1:]];axs[0,1].bar(x+(j-2)*.24,values,.24,label=labels[j],color=colors[j])
    axs[0,1].axhline(0,color='#64748b',linewidth=1);axs[0,1].set(xticks=x,xticklabels=['K2','K3','K4'],ylabel='TV change from original (%)',title='B  Relative change: negative means smaller');axs[0,1].legend(fontsize=9)
    for kind,color,name in zip(('reverse','even_odd'),colors[1:3],labels[1:3]):axs[1,0].plot(range(1,5),[max(float(g['program_TV_from_identity'][kind]),1e-20) for g in main],marker='o',label=name,color=color)
    axs[1,0].set(yscale='log',xticks=range(1,5),xlabel='Truncation order K',ylabel='Exact TV between implemented prefix laws',title='C  Order changes the implemented distribution');axs[1,0].grid(alpha=.2);axs[1,0].legend(fontsize=9)
    vals=[100*(float(v['TV_interval'][0]/local['variants'][0]['TV_interval'][0])-1) for v in local['variants']];axs[1,1].bar(range(4),vals,color=colors);axs[1,1].axhline(0,color='#64748b',linewidth=1);axs[1,1].set(xticks=range(4),xticklabels=labels,ylabel='TV change from original (%)',title='D  Complete ten-mode control, K4')
    fig.supxlabel('144-mode panels certify a marginal and a full-law lower bound, not a full-law error ranking.\nMixture uses ideal 1/3 order selection; one previously studied input, no new holdout.',fontsize=10)
    fig.savefig(out/'order_effects.png',dpi=180);fig.savefig(out/'order_effects.pdf');plt.close(fig)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps,panels=4,unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
