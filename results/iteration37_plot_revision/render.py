"""Presentation-only correction: preserve the original figure, separate its labels."""
import json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
root=Path(__file__).resolve().parents[2];out=Path(__file__).parent;data=root/'results/iteration37_analysis/summary.json';s=json.loads(data.read_text());selectors=('recent','pairwise','conditional_mi','click_influence');labels=('Recent','Pairwise MI','Conditional MI','Click influence');colors=('#aa5b50','#bb9037','#286d96','#5c8560');fig,axes=plt.subplots(1,3,figsize=(14,4.8))
for ax,forest in zip(axes[:2],('chain','static_mi_tree')):
    for k,sel in enumerate(selectors):
        points=[r for r in s['local'] if r['forest']==forest and r['selection']==sel];ax.bar(np.arange(3)+(k-1.5)*.2,[p['tv_upper'] for p in points],.2,color=colors[k])
    ax.set_yscale('log');ax.set_ylim(2.5e-6,3.5e-5);ax.set_xticks(range(3),['0-9','54-63','134-143']);ax.set(title=forest.replace('_',' '),xlabel='Ten-mode block',ylabel='Complete stopped-law TV');ax.grid(axis='y',alpha=.2)
axes[2].bar(range(4),[g['sample_mean_completed'] for g in s['full_sampling']],color=colors);axes[2].axhline(s['target_mean'],color='black',linestyle='--');axes[2].set_xticks(range(4),['Recent','Pairwise\nMI','Conditional\nMI','Click\ninfluence'],fontsize=8);axes[2].set(title='144-mode confirmation',ylabel='Mean clicks in 64 completed outputs',ylim=(0,100));fig.suptitle('Iteration 37: fixed four-click retention budget',y=.99);fig.legend(handles=[*[Patch(color=c,label=l) for c,l in zip(colors,labels)],Line2D([0],[0],color='black',linestyle='--',label='Exact target mean (right panel)')],loc='upper center',bbox_to_anchor=(.5,.94),ncol=5,fontsize=9,frameon=False);fig.subplots_adjust(top=.78,bottom=.20,wspace=.34);fig.text(.5,.025,'Local panels show exact stopped-law comparisons. Full-output sample means do not certify global TV accuracy.',ha='center',fontsize=10);target=out/'selector_comparison.png';assert not target.exists();fig.savefig(target,dpi=180);plt.close(fig)
record=dict(scope='Presentation-only revision; common local y-axis scale and shared legend remove overlap. Original analysis image preserved.',data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),image_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
with (out/'render.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
