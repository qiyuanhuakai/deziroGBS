"""Draw total-count CDF residuals against the two actual references in this study."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iteration103_sample_v1 import DATA
from iteration104_sample_v1 import FIT
from iteration104_optical_v1 import OPT
from iteration100_spatial_v1 import manifest


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    test=np.load(FIT/'witness.npz')['test']
    photon=np.load(OPT/'moments.npz')['prob']
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    colors=['#738090','#126b82','#bb5318']
    for i,(ax,ref,name) in enumerate(zip(axes,(test/test.sum(),photon),('Clicks vs. held-out hardware','Photons vs. optical input'))):
        for j,(model,label) in enumerate(zip(('raw','optical','hardware'),('Round 103','Optical calibration','Hardware calibration'))):
            if model=='raw':
                h=np.load(DATA/'iteration103_transport_gpu_v1/hold_0.npz')[('click','photon')[i]]
            else:
                h=np.load(DATA/f'iteration104_validate_v1/hold_{model}_0.npz')['joint'].sum(i)
            err=(np.cumsum(h)/h.sum())[:220]-np.cumsum(ref)[:220]
            ax.plot(np.arange(220),100*err,color=colors[j],lw=1.9,label=label)
        ax.axhline(0,color='#41474c',lw=.6)
        ax.set_xlim(35,205);ax.set_title(name,loc='left',fontsize=12)
        ax.set_xlabel('Detections per record' if i==0 else 'Photons per record')
        ax.set_ylabel('Simulated CDF minus reference (pp)')
        ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8,loc='upper right')
    fig.savefig(out/'cdf.png',dpi=175)
    plt.close(fig)
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    main(p.parse_args().output.resolve())
