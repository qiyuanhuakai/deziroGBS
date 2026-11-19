"""Rank4 complete-program error, dimension and guarantee comparisons."""
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
    out=ROOT/'results/iteration59_analysis';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('iteration59_analysis','iteration22_external','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    rows=read('results/iteration59_comparisons/error_decomposition.json');pairs=read('results/iteration59_comparisons/dimension_pairs.json');read('results/iteration59_comparisons/validation.json');plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained');labels=['real pair','complex rotated','mixed holdout'];x=np.arange(3);w=.24;d4=rows[1::2]
    ax=axs[0,0]
    for shift,dim,color in [(-.5,3,'#94a3b8'),(.5,4,'#2563eb')]:ax.bar(x+shift*.32,[float(rows[2*i+(dim==4)]['actual_TV'][1]) for i in range(3)],.32,label=f'local dimension {dim}',color=color)
    ax.set(title='A  Actual complete-program error',ylabel='TV to the full Gaussian target',xticks=x,xticklabels=labels);ax.legend(fontsize=9)
    ax=axs[0,1]
    for shift,key,col,lab in [(-1,'program_vs_finite','#f59e0b','Program to finite continuous'),(0,'finite_vs_Gaussian','#a855f7','Finite continuous to Gaussian'),(1,'actual_TV','#2563eb','Program to Gaussian')]:ax.bar(x+shift*w,[float(r[key][1]) for r in d4],w,color=col,label=lab)
    ax.set(title='B  Error sources at local dimension 4',ylabel='Total variation',xticks=x,xticklabels=labels);ax.legend(fontsize=8,loc='upper left');ax.set_ylim(0,.020)
    ax=axs[1,0]
    for shift,key,col,lab in [(-1,'old_total_upper','#94a3b8','Old complete guarantee'),(0,'total_upper','#2563eb','New complete guarantee'),(1,'actual_TV','#10b981','Actual complete-program TV')]:ax.bar(x+shift*w,[float(r[key][1] if key=='actual_TV' else r[key]) for r in d4],w,color=col,label=lab)
    ax.set_yscale('log');ax.set(title='C  Guarantees remain conservative (dimension 4)',ylabel='Total variation / upper bound',xticks=x,xticklabels=labels,ylim=(.001,2));ax.legend(fontsize=8)
    ax=axs[1,1]
    for shift,key,col,lab in [(-.5,'ideal_grid_TV','#10b981','Ideal Gaussian grid: actual TV'),(.5,'coupling_upper','#2563eb','Joint coupling guarantee')]:ax.bar(x+shift*.32,[float(r[key][1] if key=='ideal_grid_TV' else r[key]) for r in d4],.32,color=col,label=lab)
    ax.set_yscale('log');ax.set(title='D  Isolate the ideal noise channel',ylabel='Total variation / upper bound',xticks=x,xticklabels=labels,ylim=(.001,.5));ax.legend(fontsize=8)
    fig.suptitle('Iteration 59 | Full rank4 noise on two physical modes',fontsize=16);fig.supxlabel('Each case uses all 1,296 coarse-grid nodes; rank4 refers to four real noise directions',fontsize=11);fig.savefig(out/'rank4_programs.png',dpi=180);fig.savefig(out/'rank4_programs.pdf');plt.close(fig);assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps,figures=['rank4_programs.png','rank4_programs.pdf']));print('plot complete')
if __name__=='__main__':main()
