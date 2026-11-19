"""Verified finite-selector error and fixed-window mixture comparisons."""
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
    out=ROOT/'results/iteration62_analysis';out.mkdir(exist_ok=False);names=['experiments/iteration62_analysis.py','experiments/iteration22_external.py','experiments/verify_iteration34.py'];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    v=read('results/iteration62_mixture_validation/validation.json');assert v['count']==v['passed'];groups=read('results/iteration62_mixture_validation/comparisons.json');new=[g for g in groups if g['new_native']];labels=[f'{g["start"]}..{g["start"]+g["depth"]-1}\nK{g["cutoff"]}' for g in groups];newlabels=labels[2:];colors=['#94a3b8','#2563eb','#f59e0b','#10b981'];plt.rcParams.update({'font.size':10,'axes.titlesize':13,'axes.labelsize':11})
    fig,axs=plt.subplots(2,2,figsize=(14.5,10),layout='constrained');fig.suptitle('Iteration 62 | Finite-word mixture on fixed physical windows',fontsize=18)
    x=np.arange(6)
    for j,c,name,offset in ((0,colors[0],'Original order',-.18),(4,colors[3],'Finite-word mixture',.18)):axs[0,0].bar(x+offset,[float(g['variants'][j]['TV_interval'][0]) for g in groups],.36,color=c,label=name)
    axs[0,0].set(yscale='log',xticks=x,xticklabels=labels,ylabel='Prefix TV to physical target',title='A  16-bit replay anchors and new 10-bit windows');axs[0,0].axvline(1.5,color='#64748b',linestyle=':');axs[0,0].legend(fontsize=9)
    x=np.arange(4)
    for j,idx,name,c in ((0,1,'Reverse',colors[1]),(1,2,'Even / odd',colors[2]),(2,4,'Finite mixture',colors[3])):
        values=[100*(float(g['variants'][idx]['TV_interval'][0]/g['variants'][0]['TV_interval'][0])-1) for g in new];axs[0,1].bar(x+(j-1)*.25,values,.25,label=name,color=c)
    axs[0,1].axhline(0,color='#64748b',linewidth=1);axs[0,1].set(xticks=x,xticklabels=newlabels,ylabel='TV change from original (%)',title='B  New windows: negative means smaller');axs[0,1].legend(fontsize=9)
    axs[1,0].bar(np.arange(6),[float(g['finite_to_ideal_TV']) for g in groups],color=colors[1]);axs[1,0].axhline(float(groups[0]['selector_bias_bound']),color=colors[2],linestyle='--',label='Universal selector bound');axs[1,0].set(yscale='log',xticks=np.arange(6),xticklabels=labels,ylabel='TV from ideal equal-weight mixture',title='C  Exact finite-word selection error');axs[1,0].legend(fontsize=9)
    bottom=np.zeros(4)
    for j,c,name in zip(range(3),colors[:3],('Original','Reverse','Even / odd')):
        counts=np.array([g['public_component_counts'][j] for g in new]);axs[1,1].bar(np.arange(4),counts,bottom=bottom,color=c,label=name)
        for i,n in enumerate(counts):axs[1,1].text(i,bottom[i]+n/2,str(n),ha='center',va='center')
        bottom+=counts
    assert all(bottom==17);axs[1,1].set(xticks=np.arange(4),xticklabels=newlabels,ylim=(0,22),ylabel='Actual native samples requested',title='D  17 outputs per group, one component each');axs[1,1].legend(ncol=3,fontsize=9,loc='upper center')
    fig.supxlabel('All windows belong to the same previously studied 144-mode input; they are not new physical holdouts.\nPrefix TV provides a full-law lower bound, not a ranking of full144 errors. Preprocessing remains separate per order.',fontsize=10);fig.savefig(out/'finite_mixture.png',dpi=180);fig.savefig(out/'finite_mixture.pdf');plt.close(fig)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps,panels=4,unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
