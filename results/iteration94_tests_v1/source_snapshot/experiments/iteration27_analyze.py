"""Plot complete low-click sectors and independent decomposition witnesses."""
import hashlib
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/iteration27_analysis'; out.mkdir(exist_ok=False); source=Path(__file__); data=source.read_bytes(); (out/source.name).write_bytes(data)
    rows=json.loads((ROOT/'results/iteration27_panel/rows.json').read_text()); audit=json.loads((ROOT/'results/iteration27_panel/covariance_audit.json').read_text())
    selected=[r for r in rows if r['panel']=='candidate' and r['noise']==0]+[r for r in rows if r['panel'].startswith('source_')]
    labels=['Candidate, eta=.2','Candidate, eta=.6','Candidate, eta=1','Source mixed','Source optimized x2']
    fig,axes=plt.subplots(1,2,figsize=(12.4,4.5),gridspec_kw={'width_ratios':[1.35,1]})
    x=np.arange(len(selected)); bottom=np.zeros(len(selected))
    for key,label,color in [('vacuum','Vacuum','#315a88'),('one_click_mass','Exactly one click','#60a2ae'),('two_click_mass','Exactly two clicks','#e7aa57')]:
        heights=np.array([r[key] for r in selected]); axes[0].bar(x,heights,bottom=bottom,label=label,color=color); bottom+=heights
    axes[0].set_xticks(x,labels,rotation=25,ha='right'); axes[0].set_ylabel('Selected probability mass'); axes[0].set_ylim(0,1.02); axes[0].legend(fontsize=8); axes[0].set_title('Specified zero-mean Gaussian states')
    for i,r in enumerate(selected): axes[0].text(i,bottom[i]+.025,f"{r['selected_mass']:.3g}",ha='center',fontsize=9)
    values=[float(audit['mixed_minus_'+n]['rayleigh_80_digit']) for n in ('author_pure','candidate_pure')]
    axes[1].barh(['Mixed - source optimized x2','Mixed - chain candidate'],values,color=['#b24b43','#8e658b']); axes[1].axvline(0,color='black',lw=.8); axes[1].set_xlim(min(values)*1.35,.001)
    axes[1].set_xlabel('Rayleigh quotient (80 decimal digits)'); axes[1].set_title('Displacement remainder fails PSD')
    for i,v in enumerate(values): axes[1].text(v-.0002,i,f'{v:.6g}',ha='right',va='center',fontsize=9)
    axes[1].tick_params(axis='y',labelsize=8); fig.suptitle('Iteration 27: mixed low-click events and provenance audit',fontsize=13)
    fig.tight_layout(); fig.savefig(out/'mixed_events_and_psd_witnesses.png',dpi=160); plt.close(fig)
    assert source.read_bytes()==data
    (out/'summary.json').write_text(json.dumps(dict(analyzer_sha256=hashlib.sha256(data).hexdigest(),selected_rows=[r['sequence'] for r in selected],witnesses=values,unchanged_source=True),indent=2)+'\n')


if __name__=='__main__': main()
