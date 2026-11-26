"""Render speed and complete 16-detector accuracy from the fixed measurements."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iteration100_spatial_v1 import manifest
from iteration106_sample_v1 import DATA,read


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    words=read(DATA/'iteration106_words_v1/summary.json')['rows']
    frugal=read(DATA/'iteration106_frugal_v1/summary.json')['rows']
    fig,(a,b)=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for p,label,color,marker in [(False,'Direct clicks','#006c69','o'),(True,'Photon counts','#a1471a','s')]:
        x=[r for r in words if r['label']=='optical' and r['pnr']==p and not r['packed']]
        batch=np.array([r['batch'] for r in x])
        a.plot(batch,[r['us'] for r in x],label=label,color=color,marker=marker,lw=1.8)
        a.fill_between(batch,[r['quartiles'][0] for r in x],[r['quartiles'][1] for r in x],color=color,alpha=.14)
    a.axhline(1.27,color='#565966',linestyle='--',label='Jiuzhang 3.0 record interval')
    a.set(xscale='log',yscale='log',xlabel='Records per batch',ylabel='Microseconds per full host record',title='1,152 detector positions')
    a.legend(frameon=False,fontsize=8)
    x=np.arange(len(frugal))
    for name,color,mark in [('Field model','#006c69','o'),('Frugal K2','#656d82','s'),('Frugal K4','#ae5527','^')]:
        y=[100*(r['packed_full_output_projection_tv'] if name=='Field model' else r[name.split()[1]]['tv']) for r in frugal]
        b.scatter(x,y,s=40,color=color,marker=mark,label=name)
    b.set_xticks(x,['First\n16','36 /\n107','Stress\nA','Stress\nB','Random\n1','Random\n2','Random\n3'])
    b.set(yscale='log',ylabel='Total variation distance (%)',title='All 65,536 outcomes on each panel')
    b.legend(frameon=False,fontsize=8,ncol=3,loc='lower left')
    for ax in (a,b):
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    fig.savefig(out/'speed_accuracy.png',dpi=170);plt.close(fig);manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output.resolve())
