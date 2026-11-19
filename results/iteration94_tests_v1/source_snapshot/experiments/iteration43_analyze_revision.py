"""Summarize certified physical errors separately from maximum-entropy error."""
import json,shutil,csv
from pathlib import Path
from fractions import Fraction as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'results/iteration43_analysis_revision';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('iteration43_analyze_revision','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    rows=read('results/iteration43_completion_recovery/comparisons.json');bounds=read('results/iteration43_prefix_validation/bounds.json');depths=read('results/iteration43_prefix_validation/depth_bounds.json');raw=read('results/iteration43_completion/rows.json');cases=[read(f'results/iteration43_completion/case_{i:02d}.json') for i in range(25)];pairs=[]
    for i,j in ((6,7),(8,9),(15,16),(17,18),(19,20),(21,22),(23,24)):
        def align(r,key):
            v=[F(0)]*(1<<r['modes']);p=r['projection']['law'] if key=='law' else r['target']
            for mask,x in enumerate(p):v[sum(1<<r['physical_job']['order'][k] for k in range(r['modes']) if mask>>k&1)]=x
            return v
        assert align(cases[i],'target')==align(cases[j],'target');tv=sum(abs(a-b) for a,b in zip(align(cases[i],'law'),align(cases[j],'law')))/2;upper=rows[i]['maxent_TV_upper']+rows[j]['maxent_TV_upper'];assert tv<=upper;pairs.append(dict(cases=(i,j),aligned_TV=tv,triangle_upper=upper))
    for r in rows:r['exact_maxent_physical_TV_interval']=(max(F(0),r['physical_TV']-r['maxent_TV_upper']),min(F(1),r['physical_TV']+r['maxent_TV_upper']));assert 0<r['exact_maxent_physical_TV_interval'][0]<r['exact_maxent_physical_TV_interval'][1]<r['reference_TV']
    save(out/'certified_comparisons.json',encode(rows));save(out/'order_pairs.json',encode(pairs))
    with (out/'completion.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['case','modes','physical_TV','K2_reference_TV','relative_reduction','TV_to_exact_maxent_upper','exact_maxent_physical_TV_lower','exact_maxent_physical_TV_upper','seconds'])
        for r,t in zip(rows,raw):writer.writerow([r['sequence'],r['modes'],float(r['physical_TV']),float(r['reference_TV']),float(r['relative_reduction']),float(r['maxent_TV_upper']),*[float(x) for x in r['exact_maxent_physical_TV_interval']],t['seconds']])
    with (out/'prefix.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['cutoff','depth10_full_TV_lower','depth12_full_TV_lower','depth12_shifted_full_TV_lower'])
        for b in bounds:writer.writerow([b['cutoff'],float(b['old_ten_mode_lower']),float(b['full144_TV_lower']),float(b['full144_shifted_TV_lower'])])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained');ax=axes[0,0];colors={4:'#a34e00',8:'#147d92',10:'#763cab'}
    for n in (4,8,10):rs=[r for r in rows if r['modes']==n];ax.scatter([float(r['reference_TV']) for r in rs],[float(r['physical_TV']) for r in rs],s=38,color=colors[n],label=f'{n} modes',alpha=.8)
    ax.plot([0,.48],[0,.48],color='#888888',ls='--',lw=1);ax.set(xlabel='K2 real-reference physical TV',ylabel='Feasible completion physical TV',title='A | All 25 fixed cases improve',xlim=(0,.48),ylim=(0,.48));ax.legend(loc='upper left',frameon=False)
    ax=axes[0,1];ids=np.arange(25);ax.semilogy(ids,[float(r['physical_TV']) for r in rows],'o',ms=4,color='#147d92',label='Exact physical-target TV');ax.semilogy(ids,[float(r['maxent_TV_upper']) for r in rows],'s',ms=4,color='#a34e00',label='Upper bound to exact maximum entropy');ax.set(xlabel='Frozen case index',ylabel='Total variation',title='B | Two different accuracy questions',ylim=(1e-8,4));ax.legend(loc='upper right',frameon=False,fontsize=9);ax.grid(axis='y',alpha=.2)
    ax=axes[1,0]
    for k in range(1,5):ds=[d for d in depths if d['cutoff']==k];ax.semilogy([d['depth'] for d in ds],[float(d['TV_lower']) for d in ds],marker='o',ms=3,label=f'K{k}')
    ax.set(xlabel='Enumerated prefix depth (values below 1e-7 outside plot)' ,ylabel='Full144 TV lower bound to exact B,F target',title='C | Actual144-mode program; increasing evidence',ylim=(1e-7,.1),xticks=range(1,13));ax.legend(frameon=False,ncol=4,loc='upper left');ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];x=np.arange(4);ax.bar(x-.18,[float(b['old_ten_mode_lower']) for b in bounds],.34,color='#a3b7c4',label='Depth10');ax.bar(x+.18,[float(b['full144_TV_lower']) for b in bounds],.34,color='#147d92',label='Depth12');ax.set_yscale('log');ax.set(xticks=x,xticklabels=['K1','K2','K3','K4'],ylabel='Full144 TV lower bound',title='D | Stronger lower bounds; no full-law upper bound',ylim=(1e-5,.1));ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Round43: exact low-order completion and twelve-bit full-system witnesses',fontsize=15);fig.savefig(out/'completion_and_prefix.png',dpi=180);plt.close(fig)
    summary=dict(completion_cases=25,improved=25,exact_maxent_also_better=25,relative_reduction_range=[float(min(r['relative_reduction'] for r in rows)),float(max(r['relative_reduction'] for r in rows))],maxent_TV_bound_range=[float(min(r['maxent_TV_upper'] for r in rows)),float(max(r['maxent_TV_upper'] for r in rows))],max_aligned_order_TV=float(max(p['aligned_TV'] for p in pairs)),order_pairs=len(pairs),completed_fit_seconds=sum(r['seconds'] for r in raw),optimizer_success=sum(r['optimizer']['success'] for r in raw),max_final_gradient=max(r['optimizer']['gradient_max'] for r in raw),bounds=[dict(cutoff=b['cutoff'],lower=float(b['full144_TV_lower']),shifted_lower=float(b['full144_shifted_TV_lower']),gain_ratio=float(b['full144_TV_lower']/b['old_ten_mode_lower'])) for b in bounds],source_hashes=hashes,dependencies=deps)
    assert all(digest(ROOT/n)==s for n,s in hashes.items());save(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('source_hashes','dependencies')}))
if __name__=='__main__':main()
