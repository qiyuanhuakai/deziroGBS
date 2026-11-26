"""Published full-record accuracy and speed comparisons from independently archived measurements."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iteration100_spatial_v1 import manifest
from iteration106_sample_v1 import DATA,read


def run(out):
    out.mkdir(exist_ok=False,parents=True)
    eq=read(DATA/'iteration108_exact_v1/summary.json')['rows'][1]
    actual=read(DATA/'iteration108_actual_quantum_v1/summary.json')
    pl=read(DATA/'iteration108_residual_v1/summary.json')['spectra']
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8),layout='constrained')
    a=axes[0];t=eq['run'];x=[v['us'][0] for v in t];y=[100*eq['tv_by_cap'][v['cap']] for v in t]
    a.plot(x,y,'o-',color='#17695f',lw=2,label='Coherent pair matching')
    for v,xx,yy in zip(t,x,y):a.annotate('cap '+str(v['cap']),(xx,yy),xytext=(3,8),textcoords='offset points',fontsize=9)
    for i,v in enumerate(eq['frugal']):
        a.scatter(v['us'],100*v['tv'],marker='s',s=48,color='#b75530',label='Frugal' if not i else None)
        a.annotate('K'+str(v['K']),(v['us'],100*v['tv']),xytext=(5,5),textcoords='offset points',fontsize=9)
    a.set(xscale='log',yscale='log',xlabel='Microseconds per complete click record',ylabel='Complete 16-bit TV (%)',title='16 outputs, 6 source pairs')
    a.set_ylim(.1,4.7);a.legend(frameon=False);a.grid(alpha=.2)
    a=axes[1];names=['Main spectrum','Second spectrum'];noise=[v['displacement_photons'] for v in pl];pure=[v['residual_photons'] for v in pl]
    a.barh(names,noise,color='#7193b4',height=.52,label='Gaussian displacement')
    a.barh(names,pure,left=noise,color='#17695f',height=.52,label='Pure-state residual')
    for j,val in enumerate(pure):a.text(noise[j]+val+1,j,f'{val:.2f} residual',va='center',fontsize=9)
    a.set(xlabel='Mean detected photons in the covariance decomposition',title='Jiuzhang 3.0 high-power input',xlim=(0,144))
    a.invert_yaxis();a.legend(frameon=False,loc='lower right');a.spines[['top','right']].set_visible(False)
    fig.savefig(out/'accuracy.png',dpi=175);plt.close(fig);manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);run(p.parse_args().output.resolve())
