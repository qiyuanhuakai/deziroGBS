"""Scientific comparison of measured errors and distinct resource budgets."""
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
    out=ROOT/'results/iteration60_analysis';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('iteration60_analysis','iteration22_external','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    panels=read('results/iteration60_comparisons/panels.json');read('results/iteration60_comparisons/validation.json');plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained');x=np.arange(3);w=.25;labels=['real pair','complex rotated','mixed directions'];actions=[('baseline','#94a3b8','d4 / coarse'),('grid','#2563eb','d4 / selected axis fine'),('input','#f59e0b','d6 / coarse')]
    for ax,key,title in [(axs[0,0],'actual_TV','A  Actual complete-program error'),(axs[0,1],'total_upper','B  Complete-program guarantees')]:
        for shift,(action,col,lab) in zip((-1,0,1),actions):ax.bar(x+shift*w,[float(p['variants'][action][key][1] if key=='actual_TV' else p['variants'][action][key]) for p in panels],w,color=col,label=lab)
        ax.set(title=title,ylabel='TV' if key=='actual_TV' else 'Certified TV upper bound',xticks=x,xticklabels=labels);ax.legend(fontsize=8,loc='upper left');ax.set_ylim(0,ax.get_ylim()[1]*1.35)
    ax=axs[1,0];values=np.array([[float(p['variants']['baseline']['actual_TV'][0]/p['variants'][a]['actual_TV'][1]) for p in panels] for a in ('grid','input')]);im=ax.imshow(values,cmap='YlGnBu',aspect='auto',vmin=min(1,float(values.min())),vmax=max(2,float(values.max())));ax.set(title='C  Actual error improvement over the old baseline',xticks=x,xticklabels=labels,yticks=[0,1],yticklabels=['Selected-axis refinement','Finite input d4 to d6'])
    for i in range(2):
        for j in range(3):ax.text(j,i,f'{values[i,j]:.2f}x',ha='center',va='center',color='white' if values[i,j]>(values.max()+1)/2 else '#172554',fontsize=15)
    fig.colorbar(im,ax=ax,shrink=.8,label='Baseline TV / new TV (larger is better)')
    ax=axs[1,1];metrics=['nodes','fock_box_states','trace_nodes'];metriclabels=['Product nodes','Fock-box states','Measurement-tree nodes'];p=panels[0]['variants']
    for shift,action,col,label in [(-.5,'grid','#2563eb','Selected-axis refinement'),(.5,'input','#f59e0b','Finite input d4 to d6')]:
        ratios=[p[action][k]/p['baseline'][k] for k in metrics];ax.bar(x+shift*.32,ratios,.32,color=col,label=label)
        for j,val in enumerate(ratios):ax.text(j+shift*.32,val+.06,f'{val:.2f}x',ha='center',fontsize=9)
    assert all(all(pan['variants'][a][k]/pan['variants']['baseline'][k]==p[a][k]/p['baseline'][k] for k in metrics for a in ('grid','input')) for pan in panels)
    ax.axhline(1,color='#64748b',lw=1,ls='--');ax.set(title='D  Different costs: no equal-budget speed claim',ylabel='Resource count / old baseline',xticks=x,xticklabels=metriclabels,ylim=(0,4.3));ax.legend(fontsize=8);fig.suptitle('Iteration 60 | Use measured error sources to compare budget choices',fontsize=15);fig.supxlabel('Three previously studied two-mode inputs; axis selection frozen before new program evaluation',fontsize=11);fig.savefig(out/'budget_choices.png',dpi=180);fig.savefig(out/'budget_choices.pdf');plt.close(fig);assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps,figures=['budget_choices.png','budget_choices.pdf']));print('budget plot complete')
if __name__=='__main__':main()
