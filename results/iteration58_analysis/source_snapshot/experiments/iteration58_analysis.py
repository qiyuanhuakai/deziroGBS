"""Scientific plots of guarantee improvements and candidate costs."""
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
    out=ROOT/'results/iteration58_analysis';out.mkdir(exist_ok=False);deps={};names=['experiments/iteration58_analysis.py','experiments/iteration22_external.py','experiments/verify_iteration34.py'];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    rows=read('results/iteration58_coupling/retrospective_rows.json');rank4=read('results/iteration58_coupling/rank4_rows.json');resources=read('results/iteration58_comparisons/resources.json');read('results/iteration58_comparisons/validation.json')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained');colors=['#1d4ed8','#c2410c','#047857'];labels=['pair','rotated','complex (old holdout)']
    ax=axs[0,0]
    for start,col,lab in zip((4,12,20),colors,labels):
        g=rows[start:start+4];x=np.arange(4);ax.plot(x,[float(r['bound']['old_axis_upper']) for r in g],'--o',color=col,alpha=.55,label=lab+' / old');ax.plot(x,[float(r['bound']['best_upper']) for r in g],'-o',color=col,label=lab+' / best')
    ax.set(title='A  Two-drive ideal-channel guarantee',ylabel='Certified TV upper bound',xticks=np.arange(4),xticklabels=['4 / 4','16 / 4','4 / 16','16 / 16'],xlabel='Interior cells by axis');ax.legend(fontsize=8,ncol=2)
    ax=axs[0,1];g=[rows[i] for i in (7,15,23)];x=np.arange(3);width=.25
    for shift,key,col,lab in [(-1,'old_total_upper','#94a3b8','Old guarantee'),(0,'new_total_upper','#2563eb','New guarantee')]:ax.bar(x+shift*width,[float(r[key]) for r in g],width,color=col,label=lab)
    ax.bar(x+width,[float(r['actual_TV'][1]) for r in g],width,color='#10b981',label='Actual TV (unchanged)');ax.set(title='B  Complete program: d=6, both axes fine',ylabel='Total variation',xticks=x,xticklabels=['pair','rotated','complex']);ax.legend(fontsize=9)
    ax=axs[1,0]
    for fam,col in zip(('real_pair','complex_rotated'),colors):
        g=[r for r in rank4 if r['family']==fam];ns=sorted(set(r['nodes'] for r in g))
        for key,style,label in [('old_axis_upper','--o','old'),('best_upper','-o','best')]:ax.plot(ns,[min(float(r['bound'][key]) for r in g if r['nodes']<=n) for n in ns],style,color=col,label=f'{fam} / {label}')
    ax.set_xscale('log',base=3);ax.set(title='C  Rank4: ideal-channel bounds only',xlabel='Candidate product nodes (not runtime)',ylabel='Best guarantee within node budget');ax.set_xticks(ns,[f'{n:,}' for n in ns],rotation=15);ax.legend(fontsize=8)
    ax=axs[1,1];groups=resources['rank4'];x=np.arange(2);w=.25
    for shift,key,col,label in [(-1,'old_fine_nodes','#94a3b8','Old all-fine reference'),(0,'exact_l1_min_nodes','#f59e0b','Exact L1 minimum'),(1,'new_min_nodes','#2563eb','Combined minimum')]:ax.bar(x+shift*w,[r[key] for r in groups],w,label=label,color=col)
    ax.set_yscale('log');ax.set(title='D  Rank4: meet the old all-fine guarantee',ylabel='Minimum nodes among 16 candidates',xticks=x,xticklabels=['real_pair','complex_rotated']);ax.legend(fontsize=8,loc='upper right');ax.set_ylim(700,900000)
    for j in range(2):ax.text(j+w,1800,'1,296\n81x fewer',ha='center',fontsize=9,color='#1d4ed8')
    fig.suptitle('Iteration 58 | Tighter error guarantees, unchanged sampling algorithms',fontsize=16);fig.savefig(out/'joint_coupling.png',dpi=180);fig.savefig(out/'joint_coupling.pdf');plt.close(fig);assert all(digest(ROOT/n)==s for n,s in hashes.items());assert all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps,figures=['joint_coupling.png','joint_coupling.pdf']));print('plots complete')

if __name__=='__main__':main()
