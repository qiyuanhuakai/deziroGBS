"""Publication figure: feasible resource and preserved numerical failures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iteration23_support import save,digest
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/iteration28_analysis'; out.mkdir(exist_ok=False); src=Path(__file__); initial=src.read_bytes(); (out/src.name).write_bytes(initial)
    rows=json.loads((ROOT/'results/iteration28_panel/rows.json').read_text()); primary=rows[14]
    pilot=json.loads((ROOT/'results/iteration28_pilot/rows.json').read_text()); previous=json.loads((ROOT/'results/iteration27_panel/covariance_audit.json').read_text())
    accepted=rows[12:15]; fig,axes=plt.subplots(1,3,figsize=(14,4.3))
    x=np.arange(3); lo=np.array([r['full_original_diagnostics']['dual_photon_lower_estimate'] for r in accepted]); hi=np.array([r['full_pure_mean_photons'] for r in accepted])
    axes[0].vlines(x,lo,hi,color='#315a88',lw=3); axes[0].scatter(x,hi,color='#315a88',label='Feasible core',s=35); axes[0].scatter(x,lo,color='#c47c31',marker='_',s=150,label='Dual diagnostic')
    axes[0].set_xticks(x,['1e-4','1e-5','1e-6*']); axes[0].set_xlabel('Active working margin (*primary)'); axes[0].set_ylabel('Pure-core mean photons'); axes[0].set_title('Primal and dual diagnostics'); axes[0].legend(fontsize=8)
    labels=['Author saved','Old chain','No-margin search','Guarded active*']; values=[previous['mixed_minus_author_pure']['minimum_eigenvalue'],previous['mixed_minus_candidate_pure']['minimum_eigenvalue'],pilot[1]['diagnostics']['minimum_remainder_eigenvalue'],primary['minimum_encoded_remainder']]
    axes[1].bar(labels,values,color=['#b24b43']*3+['#4b8b67']); axes[1].set_yscale('symlog',linthresh=1e-8); axes[1].axhline(0,color='black',lw=.7); axes[1].tick_params(axis='x',rotation=25,labelsize=8); axes[1].set_title('Actual saved noise remainder'); axes[1].set_ylabel('Smallest eigenvalue'); axes[1].text(.03,.03,'*50 active modes; full model error 7.22e-14',transform=axes[1].transAxes,fontsize=7.5)
    totals=[300,300,301]; passed=[279,279,301]; names=['Initial codec','Digitwise retry','Environment guard']
    axes[2].bar(names,passed,color='#4b8b67',label='Passed'); axes[2].bar(names,np.array(totals)-passed,bottom=passed,color='#b24b43',label='Failed'); axes[2].set_ylim(0,340); axes[2].set_title('Scalar transport checks'); axes[2].tick_params(axis='x',rotation=25,labelsize=8); axes[2].legend(fontsize=8)
    for i in range(3): axes[2].text(i,totals[i]+8,f'{passed[i]}/{totals[i]}',ha='center',fontsize=9)
    fig.suptitle('Iteration 28: constructive decomposition and host arithmetic precision',fontsize=13); fig.tight_layout(); fig.savefig(out/'resource_and_codec.png',dpi=160); plt.close(fig)
    assert src.read_bytes()==initial; save(out/'summary.json',dict(analyzer_sha256=digest(src),unchanged_source=True,primary_sequence=14))


if __name__=='__main__': main()
