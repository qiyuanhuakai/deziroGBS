"""Visual comparison of all frozen noise resource variants and prediction outcomes."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from iteration65_support import ROOT,snapshot,digest,save
from verify_iteration34 import decode

def main():
    out=ROOT/'results/iteration65_analysis';hashes=snapshot(out);deps={}
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    v=read('results/iteration65_comparisons/validation.json');assert v['count']==v['passed'];panels=read('results/iteration65_comparisons/panels.json');actions=['baseline','grid','input','joint'];colors=['#94a3b8','#2563eb','#f59e0b','#10b981'];labels=['Old real','Old complex','Old mixed','New real','New phase / mixed'];x=np.arange(5)
    fig,axs=plt.subplots(2,2,figsize=(15,10),layout='constrained');fig.suptitle('Iteration 65 | Joint rank-4 noise budget and new physical inputs',fontsize=18)
    for j,(action,color) in enumerate(zip(actions,colors)):
        axs[0,0].bar(x+(j-1.5)*.2,[float(p['variants'][action]['actual_TV'][0]) for p in panels],.2,color=color,label=action)
        axs[0,1].bar(x+(j-1.5)*.2,[float(p['variants'][action]['total_upper']) for p in panels],.2,color=color,label=action)
    axs[0,0].set(yscale='log',ylabel='Complete two-mode threshold TV',title='A  Actual finite-word program error');axs[0,0].legend(ncol=2,fontsize=9)
    axs[0,1].set(ylabel='Certified full-program TV upper bound',title='B  Input + noise + selector + arithmetic budget');axs[0,1].legend(ncol=2,fontsize=9)
    for j,(action,color) in enumerate(zip(['grid','input','joint'],colors[1:])):axs[1,0].bar(x+(j-1)*.25,[float(p['variants']['baseline']['actual_TV'][0]/p['variants'][action]['actual_TV'][1]) for p in panels],.25,color=color,label=action)
    axs[1,0].axhline(1,color='#64748b',linestyle='--');axs[1,0].set(ylabel='Baseline TV / new TV',title='C  Improvement factors; costs differ');axs[1,0].legend(fontsize=9)
    axs[1,1].axis('off');table_rows=[]
    for label,p in zip(labels,panels):table_rows.append([label,str(p['selected_axis']),p['predicted_winner'],p['actual_winner'],'yes' if p['joint_beats_both'] else 'no'])
    table=axs[1,1].table(cellText=table_rows,colLabels=['Input','Axis','Proxy choice','Actual choice','Joint best?'],loc='center',cellLoc='center',colWidths=[.25,.1,.22,.22,.21]);table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2.1);axs[1,1].set_title('D  Frozen grid-vs-input prediction, all outcomes')
    for ax in (axs[0,0],axs[0,1],axs[1,0]):ax.set_xticks(x,labels,rotation=15);ax.axvline(2.5,color='#64748b',linestyle=':');ax.grid(axis='y',alpha=.15)
    fig.supxlabel('Two physical modes with four independent real drives. Three old-input joint runs + eight new-holdout runs; nine prior cases reused by hashes.\nCoarse/fine grids: 1,296/3,888 nodes. Local dimensions: 4/6. Proxy chooses from state+coupling bounds before full-program execution.',fontsize=10)
    fig.savefig(out/'joint_budget.png',dpi=180);fig.savefig(out/'joint_budget.pdf');plt.close(fig)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
