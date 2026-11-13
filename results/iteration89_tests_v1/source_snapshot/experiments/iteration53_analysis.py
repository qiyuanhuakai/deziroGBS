"""Plot certified bounds and separate evidence/operation costs."""
import json,sys,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration22_external import digest,save
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration53_analysis';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('iteration53_analysis','iteration22_external','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    rows=read('results/iteration53_comparisons/rows.json');storage=read('results/iteration53_comparisons/storage.json');v=read('results/iteration53_comparisons/validation.json');assert v['count']==v['passed'];depth=read('results/iteration53_prefix_validation/depth_bounds.json');transform=read('results/iteration53_prefix_validation/transform.json')
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');blue='#19647e';orange='#e08e45';green='#538c55';x=np.arange(4)
    a=axes[0,0]
    for offset,key,label,color in [(-.18,'old_lower','14-mode evidence',orange),(.18,'new_lower','15-mode evidence',blue)]:a.bar(x+offset,[float(r[key]) for r in rows],width=.34,label=label,color=color)
    a.set_yscale('log');a.set_xticks(x,['K1','K2','K3','K4']);a.set_ylim(1e-4,.25);a.set_ylabel('Full 144-mode TV lower bound');a.set_title('A. Same program; stronger certified lower bounds');a.legend(fontsize=9,loc='upper right')
    a=axes[0,1];a.bar(x,[float(r['percent_gain']) for r in rows],color=blue);a.set_xticks(x,['K1','K2','K3','K4']);a.set_ylabel('Lower-bound increase (%)');a.set_title('B. Gain from one more observed mode');a.set_ylim(0,max(float(r['percent_gain']) for r in rows)*1.3)
    for i,r in enumerate(rows):a.text(i,float(r['percent_gain'])*1.03,f"{float(r['percent_gain']):.2f}%",ha='center',fontsize=10)
    a=axes[1,0];r=storage['new_certificates'];keys=['expanded_json_bytes','compact_json_bytes','compressed_bytes'];values=[r[k]/2**20 for k in keys];a.bar(range(3),values,color=[orange,green,blue]);a.set_xticks(range(3),['Full JSON\n(reconstructed)','Compact JSON','Stored gzip']);a.set_ylim(0,max(values)*1.22);a.set_ylabel('MiB, new 16,384 certificates');a.set_title('C. Complete witnesses remain reconstructible')
    for i,v in enumerate(values):a.text(i,v*1.02,f'{v:.1f}',ha='center')
    a=axes[1,1];values=[transform['direct_terms'],transform['producer_interval_subtractions']];a.bar(range(2),values,color=[orange,blue]);a.set_yscale('log');a.set_xticks(range(2),['Direct inclusion\n(independent check)','Integer Mobius\n(production)']);a.set_ylim(1e5,1e8);a.set_ylabel('Term / interval-update count');a.set_title('D. Different exact algorithms agree on all atoms')
    for i,v in enumerate(values):a.text(i,v*1.12,f'{v:,}',ha='center')
    for a in axes.flat:a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    fig.suptitle('Round 53 | 15-mode evidence for the unchanged 144-mode sampler',fontsize=14);fig.savefig(out/'prefix15.png',dpi=160);plt.close(fig)
    assert all(digest(ROOT/n)==s for n,s in hashes.items());assert all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps,scope='No full144 upper bound or matched-quality speed claim; operation counts use distinct units.'))
if __name__=='__main__':main()
