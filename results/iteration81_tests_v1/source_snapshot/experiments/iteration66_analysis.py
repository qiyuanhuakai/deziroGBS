"""Show all fixed-displacement errors and actual stored-state certificates."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,json
from iteration66_support import ROOT,snapshot,save,digest
from verify_iteration34 import decode

def main():
    out=ROOT/'results/iteration66_analysis';hashes=snapshot(out);deps={}
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    v=read('results/iteration66_comparisons/validation.json');assert v['count']==v['passed'];panels=read('results/iteration66_comparisons/panels.json');actions=['baseline','rank','dimension','joint'];colors=['#94a3b8','#2563eb','#f59e0b','#10b981'];labels=['Pair2','Phase2','Chain3','Dense phase3','New mixed3'];x=np.arange(5);fig,axs=plt.subplots(2,2,figsize=(15,10),layout='constrained');fig.suptitle('Iteration 66 | Actual Gaussian-SVD tensors to complete finite-word programs',fontsize=17)
    for z,ax in enumerate(axs.flat[:3]):
        selected=[p for p in panels if p['displacement']==z]
        for j,(a,c) in enumerate(zip(actions,colors)):ax.bar(x+(j-1.5)*.2,[float(p['variants'][a]['actual_TV'][1]) for p in selected],.2,color=c,label=a)
        ax.set(yscale='log',ylabel='Complete threshold + failure TV',title=['A  Zero displacement','B  Fixed displacement 1','C  Fixed displacement 2'][z]);ax.legend(ncol=2,fontsize=9)
    ax=axs[1,1];selected=[p for p in panels if p['displacement']==0]
    for j,(a,c) in enumerate(zip(actions,colors)):ax.bar(x+(j-1.5)*.2,[float(p['variants'][a]['state_trace'][1]) for p in selected],.2,color=c,label=a)
    ax.set(yscale='log',ylabel='Certified stored-state trace distance',title='D  Infinite rational Gaussian versus actual binary64 tensors');ax.legend(ncol=2,fontsize=9)
    for ax in axs.flat:ax.set_xticks(x,labels,rotation=12);ax.axvline(3.5,color='#64748b',linestyle=':');ax.grid(axis='y',alpha=.15)
    fig.supxlabel('5 physical inputs,20 Gaussian-SVD constructions,60 complete programs. Four inputs follow exploratory pilots; the final input is new.\n(d,rank):baseline(4,2),rank(4,6),dimension(6,2),joint(6,6). Rank is an occupation cap,not necessarily the actual MPS bond. Costs differ.',fontsize=10);fig.savefig(out/'gsvd_program.png',dpi=180);fig.savefig(out/'gsvd_program.pdf');plt.close(fig);assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
