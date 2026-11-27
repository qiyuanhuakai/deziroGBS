"""Compare full-record TV bounds against one specified optical calibration."""
import json
import matplotlib.pyplot as plt
from iteration119_hardware import OUT,ROOT

def run():
    held=json.loads((OUT/'hardware_holdout.json').read_bytes())['holdouts']
    model=json.loads((OUT/'clamped_pilot_8.json').read_bytes())
    fig,ax=plt.subplots(figsize=(7.0,2.6))
    ax.set_ylim(-.65,2.55);ax.set_xlim(0,4.6)
    ax.set_yticks([2,1,0],['Sampler, either readout','Capture 12, clicks','Capture 13, clicks'])
    upper=100*model['full_record_tv_upper']
    ax.plot([0,upper],[2]*2,color='#14665c',lw=5,solid_capstyle='butt')
    ax.plot(upper,2,'|',ms=14,color='#14665c',mew=2)
    ax.text(upper+.08,2,'≤ 0.4824%',ha='left',va='center',color='#14665c',weight='bold')
    for h,y in zip(held,[1,0]):
        lower=h['average_conditional_law_tv_lower99']*100
        ax.annotate('',(4.55,y),(lower,y),arrowprops=dict(arrowstyle='->',lw=2.8,color='#904422'))
        ax.plot(lower,y,'|',ms=14,color='#904422',mew=2)
        ax.text(lower-.08,y,f'≥ {int(lower*1000)/1000:.3f}%',ha='right',va='center',color='#904422',weight='bold')
    ax.grid(axis='x',color='.88',lw=.7);ax.set_axisbelow(True)
    ax.set_xlabel('Total variation distance to the high-power optical model (%)')
    ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,pad=8)
    fig.tight_layout()
    fig.savefig(ROOT/'papers/number-tilted-gbs/record-correction.pdf',metadata={'CreationDate':None})
    plt.close(fig)
if __name__=='__main__':run()
