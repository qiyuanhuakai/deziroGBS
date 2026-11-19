"""Plot finite-time and first-hit error, retaining quantization and failure scope."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iteration64_common import ROOT,read,save,digest,snapshot

def main():
    out=ROOT/'results/iteration64_analysis';hashes=snapshot(out);deps={}
    def load(n):deps[n]=digest(ROOT/n);return read(ROOT/n)
    v=load('results/iteration64_chain_validation/validation.json');assert v['count']==v['passed'];rows=load('results/iteration64_chain_validation/comparisons.json')
    fig,axs=plt.subplots(2,2,figsize=(15,10),layout='constrained');fig.suptitle('Iteration 64 | Actual matching chains, finite words and stopping laws',fontsize=18);colors=['#2563eb','#f59e0b','#10b981','#a855f7'];selected=[r for r in rows if r['graph'] in ('path_3','triangle_3') and r['c']==1]
    for color,r in zip(colors,selected):
        points=[p for p in r['variants'][0]['series'] if p['conditional_TV'] is not None and p['conditional_TV']>0]
        axs[0,0].plot([p['step'] for p in points],[float(p['conditional_TV']) for p in points],'o-',color=color,label=f'{r["graph"]}, alpha={r["alpha"]}')
    axs[0,0].set(xscale='log',yscale='log',xlabel='Fixed chain steps, empty start',ylabel='Conditional projected TV to graph target',title='A  Exact-word chain: c=1, nontrivial graph panel');axs[0,0].legend(fontsize=8)
    active=[r for r in rows if r['graph'] not in ('isolated_1','empty_2')];labels=[f'{r["graph"]}\nc={r["c"]}, a={r["alpha"]}' for r in active];x=np.arange(len(active))
    axs[0,1].bar(x-.18,[float(r['variants'][0]['first_hit_TV']) for r in active],.36,color=colors[1],label='First perfect hit, eventually');axs[0,1].bar(x+.18,[float(r['variants'][0]['series'][-1]['conditional_TV']) for r in active],.36,color=colors[0],label='Fixed 256 steps, then condition');axs[0,1].set(ylabel='Projected TV to graph target',title='B  Different stopping rules produce different laws');axs[0,1].legend(fontsize=8)
    for j,color in [(1,colors[1]),(2,colors[0])]:
        points=[(i,float(r['variants'][j]['stationary_TV'])) for i,r in enumerate(active) if r['variants'][j]['stationary_TV']>0]
        axs[1,0].scatter([i for i,y in points],[y for i,y in points],color=color,label='Naive '+('8-bit' if j==1 else '64-bit'))
    axs[1,0].set(yscale='log',ylabel='Full stationary TV from intended weighted law',title='C  Quantized selection / acceptance can bias equilibrium');axs[1,0].legend(fontsize=8)
    axs[1,1].bar(x-.18,[r['samples'][0]['attempts'] for r in active],.36,color=colors[2],label='Exact rejection words');axs[1,1].bar(x+.18,[r['samples'][1]['attempts'] for r in active],.36,color=colors[3],label='Naive 64-bit words');axs[1,1].set(ylabel='Observed 64-step attempts for 7 requested outputs',title='D  Public trajectories, not a throughput benchmark');axs[1,1].legend(fontsize=8)
    for ax in (axs[0,1],axs[1,0],axs[1,1]):ax.set_xticks(x,labels,rotation=75,ha='right',fontsize=7)
    for ax in axs.flat:ax.grid(axis='y',alpha=.15)
    fig.supxlabel('5 fixed graphs x 3 c values x 2 edge multipliers; at most 32 matching states. Exact finite-time laws, not large-system mixing certificates.\nPublic output uses independent restarted 64-step attempts, capped at 64; any failure remains explicit. First-hit is an audited comparator, not the sampler.',fontsize=10)
    fig.savefig(out/'matching_chain.png',dpi=180);fig.savefig(out/'matching_chain.pdf');plt.close(fig)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items())
    save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True))

if __name__=='__main__':main()
