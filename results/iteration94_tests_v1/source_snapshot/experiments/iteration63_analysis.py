"""Visualize audited exact graph comparisons without fitting or selection."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iteration63_common import ROOT,read,save,digest,snapshot

def main():
    out=ROOT/'results/iteration63_analysis';hashes=snapshot(out);deps={}
    def load(n):deps[n]=digest(ROOT/n);return read(ROOT/n)
    valid=load('results/iteration63_graph_validation/validation.json');assert valid['passed']==valid['count']
    rows=load('results/iteration63_graph_validation/comparisons.json');labels=['I1','E4','P6','C6','Star6','K6','2 triangles','Bowtie5','K3,3','K8'];colors=['#2563eb','#f59e0b','#10b981'];x=np.arange(10)
    plt.rcParams.update({'font.size':10});fig,axs=plt.subplots(2,2,figsize=(15,10),layout='constrained');fig.suptitle('Iteration 63 | Exact graph-law construction and finite-word audit',fontsize=18)
    for ci,label,color in zip(range(3),('c = 1/4','c = 1','c = 2'),colors):
        group=rows[ci::3]
        axs[0,0].plot(x,[float(r['universal_mass']) for r in group],'o-',color=color,label=label)
        axs[0,1].bar(x+(ci-1)*.25,[float(r['alpha_ratio']) for r in group],.25,color=color,label=label)
        positive=[(i,float(r['natural_direct_TV'])) for i,r in enumerate(group) if r['natural_direct_TV']]
        if positive:axs[1,0].scatter([i+(ci-1)*.15 for i,v in positive],[v for i,v in positive],color=color,label=label)
    axs[0,0].axhline(.5,color='#64748b',linestyle='--',label='Guarantee > 1/2');axs[0,0].set(ylim=(.45,1.02),ylabel='Perfect-matching stationary probability',title='A  Uniform edge multiplier 4 n^2');axs[0,0].legend(fontsize=8)
    axs[0,1].set(yscale='log',ylabel='4 n^2 / smallest successful integer',title='B  Guarantee looseness, not measured speedup');axs[0,1].legend(fontsize=8)
    axs[1,0].set(yscale='log',ylabel='TV between two finite 64-bit laws',title='C  Natural matching order vs direct subset CDF');axs[1,0].legend(fontsize=8)
    physical=rows[::3];axs[1,1].bar(x,[float(r['bridge']['collision_free_probability'][0]) for r in physical],color=colors[0]);axs[1,1].set(ylim=(0,1.06),ylabel='Collision-free PNR event probability',title='D  Valid pure Gaussian bridge, c = 1/4')
    for ax in axs.flat:ax.set_xticks(x,labels,rotation=30,ha='right');ax.grid(axis='y',alpha=.15)
    fig.supxlabel('10 fixed graphs, 30 rational-parameter cases, at most 8 graph vertices. Exact exponential tables; no Markov-chain runtime result.\nGrouped matching CDF equals direct subset CDF in every case. The graph law is conditional on collision-free PNR output.',fontsize=10)
    fig.savefig(out/'graph_construction.png',dpi=180);fig.savefig(out/'graph_construction.pdf');plt.close(fig)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items())
    save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True))

if __name__=='__main__':main()
