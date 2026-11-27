"""Publication figure from raw matched runs and detector covariance data."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

def run(out):
    root=Path(__file__).resolve().parents[1]/'results'
    read=lambda p:json.loads((root/p).read_text())
    fig,ax=plt.subplots(1,2,figsize=(10.2,4.1),layout='constrained')
    fig.suptitle('Dense optical networks and Jiuzhang 3.0 readout',fontsize=14)
    rows=read('iteration114_peers_v1/summary.json')
    for name,label,marker in [('source_click','Direct clicks','o'),('frugal2_cpu','Frugal K=2, CPU','s'),('frugal4_cpu','Frugal K=4, CPU','^')]:
        part=sorted([v for v in rows if v['name']==name],key=lambda v:v['r'])
        ax[0].plot([v['us'] for v in part],[v['click_model_tv']*100 for v in part],marker=marker,label=label)
        for v in part:
            ax[0].annotate(str(v['r']),(v['us'],v['click_model_tv']*100),xytext=(4,5 if v['r']==4 else -10),textcoords='offset points',fontsize=8)
    ax[0].set(xscale='log',yscale='log',xlabel='Microseconds per 13-bit record',ylabel='Complete click TV (%)',
              title='Two and four squeezed inputs')
    ax[0].set_xticks([3,5,10,20,40],labels=['3','5','10','20','40'])
    ax[0].xaxis.set_minor_formatter(NullFormatter())
    ax[0].margins(y=.12)
    ax[0].legend(frameon=False,fontsize=8,loc='lower left');ax[0].grid(alpha=.18,which='both')
    d=read('iteration114_saturation_v1/summary.json')
    names=['Arriving photons\n(optical model)','Fired bins\n(optical model)','Fired bins\n(capture 10)','Fired bins\n(capture 11)']
    values=[np.sqrt(d['var_photons']),np.sqrt(d['var_fired']),*[v['std'] for v in d['hardware']]]
    ax[1].bar(np.arange(4),values,color=['#266d75','#57a49c','#9a7953','#b29573'])
    for i,v in enumerate(values):ax[1].text(i,v+.35,f'{v:.2f}',ha='center',fontsize=9)
    ax[1].set(xticks=np.arange(4),xticklabels=names,ylabel='Standard deviation per record',ylim=(0,30),
              title='High-power total fluctuations')
    ax[1].tick_params(axis='x',labelsize=8)
    for a in ax:
        a.spines[['top','right']].set_visible(False)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=180);plt.close(fig)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('output',type=Path)
    run(a.parse_args().output)
