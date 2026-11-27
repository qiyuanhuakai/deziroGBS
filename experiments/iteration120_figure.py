import json
import matplotlib.pyplot as plt
import numpy as np
from iteration120_calibration import OUT,ROOT,DATA

def run():
    bounds=json.loads((OUT/'hardware_panel_bounds.json').read_bytes())['panels']['poisson']
    hold=json.loads((DATA/'iteration119_weights/hardware_holdout.json').read_bytes())['holdouts']
    fig,axs=plt.subplots(1,2,figsize=(8.4,2.85),gridspec_kw={'width_ratios':[1,1.1]})
    ax=axs[0];ax.set_xlim(0,4.75)
    ax.hlines(2,0,.4815,color='#14665c',lw=4)
    ax.plot(.4815,2,'|',ms=12,color='#14665c');ax.text(.57,2,'≤ 0.4815',va='center',fontsize=8)
    for h,y in zip(hold,(1,0)):
        lo=100*h['average_conditional_law_tv_lower99']
        ax.annotate('',xy=(4.6,y),xytext=(lo,y),arrowprops={'arrowstyle':'->','color':'#974721','lw':2.6})
        ax.plot(lo,y,'|',ms=12,color='#974721');ax.text(lo-.1,y,f'≥ {np.floor(lo*1000)/1000:.3f}',va='center',ha='right',fontsize=8)
    ax.set_yticks([2,1,0],['Sampler','File 12','File 13'])
    ax.set_title('A   Complete records vs optical model',loc='left',fontsize=10)
    ax=axs[1];ax.set_xlim(0,1.22)
    for row,y in zip(bounds,(2,1,0)):
        a=row['emitted_to_finite_file_upper']*100;b=row['emitted_to_average_conditional_upper99']*100
        ax.hlines(y,0,b,color='#587391',lw=2)
        ax.plot(b,y,'|',ms=12,color='#3c5c7b')
        ax.plot(a,y,'o',ms=4,color='#3c5c7b')
        ax.text(b+.025,y,f'{np.ceil(b*1000)/1000:.3f}',va='center',fontsize=8)
    ax.set_yticks([2,1,0],['34,35','71,72','102,143'])
    ax.set_title('B   Bank-pair multiplicities vs hardware',loc='left',fontsize=10)
    for ax in axs:
        ax.set_ylim(-.63,2.65);ax.grid(axis='x',color='.88',lw=.7);ax.set_axisbelow(True)
        ax.set_xlabel('TV distance (%)');ax.tick_params(axis='y',length=0)
        ax.spines[['right','top','left']].set_visible(False)
    fig.subplots_adjust(left=.10,right=.985,top=.85,bottom=.22,wspace=.42)
    fig.savefig(ROOT/'papers/number-tilted-gbs/hardware-panels.pdf',metadata={'CreationDate':None});plt.close(fig)
if __name__=='__main__':run()
