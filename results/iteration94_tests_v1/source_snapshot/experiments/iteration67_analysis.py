"""Compare certified expected resources at two matched error requirements."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iteration67_support import ROOT,read,save,digest,snapshot

def main():
    out=ROOT/'results/iteration67_analysis';hashes=snapshot(out);deps={}
    def load(n):deps[n]=digest(ROOT/n);return read(ROOT/n)
    valid=load('results/iteration67_samples_validation/validation.json');assert valid['count']==valid['passed'];plans=load('results/iteration67_selection/plans.json');fig,axs=plt.subplots(2,2,figsize=(15,10),layout='constrained');x=np.arange(5);labels=['Triangle 3','Path 4','Cycle 4','New star 4','New diamond 4'];fig.suptitle('Iteration 67 | Exact expected cost at matched conditional-TV limits',fontsize=17)
    for row in range(2):
        selected=plans[row::2]
        for col,metric in enumerate(('expected_words','expected_steps')):
            ax=axs[row,col]
            for ri,(route,color,label) in enumerate((('joint','#2563eb','Select alpha + time'),('fixed_alpha','#f59e0b','Fixed alpha = 4n^2'))):
                values=[float(p[route][metric]) if p[route] else np.nan for p in selected];ax.bar(x+(ri-.5)*.35,values,.35,color=color,label=label)
                for j,p in enumerate(selected):
                    if p[route] is None:ax.text(j+(ri-.5)*.35,.05,'N/A',rotation=90,ha='center',transform=ax.get_xaxis_transform(),fontsize=8)
            ax.set(title=f'TV <= {float(selected[0]["epsilon"]):g}',ylabel='Expected 64-bit words / output' if col==0 else 'Expected transitions / output');ax.set_xticks(x,labels,rotation=12);ax.legend(fontsize=9);ax.grid(axis='y',alpha=.15);ax.axvline(2.5,color='#64748b',linestyle=':')
    fig.supxlabel('20 kernels; 140 finite laws; 10 error-constrained choices. Selection minimizes expected words, then transitions, alpha and time.\nIndependent restarts after fixed-time attempts; collision-free conditional graph target only. Exact small-state audits; no wall-clock or large-graph claim.',fontsize=10);fig.savefig(out/'matching_cost.png',dpi=180);fig.savefig(out/'matching_cost.pdf');plt.close(fig);assert all(digest(ROOT/n)==h for n,h in hashes.items()) and all(digest(ROOT/n)==h for n,h in deps.items());save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
