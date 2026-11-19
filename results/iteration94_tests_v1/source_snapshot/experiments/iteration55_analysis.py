"""Visualize directional refinement tradeoffs without conflating references."""
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
    out=ROOT/'results/iteration55_analysis';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('iteration55_analysis','iteration22_external','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    rows=read('results/iteration55_comparisons/rows.json');summary=read('results/iteration55_comparisons/summary.json');v=read('results/iteration55_comparisons/validation.json');assert v['count']==v['passed'];fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');blue='#19647e';orange='#e08e45'
    for a,key,title in ((axes[0,0],'actual_TV','A. Heldout dimension6: full-target TV'),(axes[0,1],'finite_continuous_TV','B. Same four runs: finite-reference TV')):
        values=np.array([[float(next(r for r in rows if r['job']['family']=='complex_holdout' and r['job']['dimension']==6 and r['job']['cells']==[x,y])[key][0]) for x in (4,16)] for y in (4,16)]);im=a.imshow(values,cmap='Blues',aspect='auto',vmin=0,vmax=values.max()*1.25);a.set_xticks([0,1],['4','16']);a.set_yticks([0,1],['4','16']);a.set_xlabel('Cells on drive1');a.set_ylabel('Cells on drive2');a.set_title(title);fig.colorbar(im,ax=a,shrink=.85,label='TV')
        for i in range(2):
            for j in range(2):a.text(j,i,f'{values[i,j]:.7f}',ha='center',va='center',color='white' if values[i,j]>values.max()*.65 else 'black',fontsize=12)
    a=axes[1,0];families=['pair','rotated','complex_holdout'];x=np.arange(3)
    for offset,d,color in [(-.18,4,orange),(.18,6,blue)]:
        r=[next(r for r in rows if r['job']['family']==f and r['job']['dimension']==d and r['job']['cells']==[16,16]) for f in families];a.bar(x+offset,[float(v['actual_TV'][0]) for v in r],width=.34,color=color,label=f'Local dimension {d}')
    a.set_yscale('log');a.set_ylim(1e-4,.3);a.set_xticks(x,['Pair','Passive mixed','Complex heldout']);a.set_ylabel('Full-target TV at (16,16) cells');a.set_title('C. Larger finite input reduces target error');a.legend(fontsize=9,loc='upper left');a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    a=axes[1,1];x=np.arange(2);a.bar(x-.18,[summary['physical_refinements']['improved'],summary['finite_refinements']['improved']],width=.34,label='Improved',color=blue);a.bar(x+.18,[summary['physical_refinements']['worsened'],summary['finite_refinements']['worsened']],width=.34,label='Worsened',color=orange);a.set_xticks(x,['Full Gaussian target','Finite continuous reference']);a.set_ylim(0,26);a.set_ylabel('Count among24 coordinate refinements');a.set_title('D. Even finite-reference TV is not monotone');a.legend(fontsize=9,loc='upper right');a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    fig.suptitle('Round55 | Two Gaussian drives: grid allocation and input bias',fontsize=14);fig.savefig(out/'two_drive_noise.png',dpi=160);plt.close(fig);assert all(digest(ROOT/n)==s for n,s in hashes.items());assert all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,source_hashes=hashes,dependencies=deps,scope='Two-mode rank2 real phase-space noise. Heatmaps have separately labeled color scales.'))
if __name__=='__main__':main()
