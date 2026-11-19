"""Keep deterministic local bounds separate from rare-event Monte Carlo errors."""
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iteration23_support import save,digest
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/iteration29_analysis'; out.mkdir(exist_ok=False); src=Path(__file__); content=src.read_bytes(); (out/src.name).write_bytes(content)
    rows=json.loads((ROOT/'results/iteration29_panel/rows.json').read_text()); large=rows[16:]; small=rows[:16]
    labels=['New 32','New 64','New 128','New 200','Author PCO','Author haf']; x=np.arange(6)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    values=[r['marginals']['full_threshold_tv_lower_diagnostic'] for r in large]
    axes[0].bar(x,values,color='#315a88'); axes[0].set_ylim(0,.008); axes[0].set_ylabel('Full-output TV lower diagnostic'); axes[0].set_title('Deterministic local threshold witnesses')
    axes[0].set_xticks(x,labels,rotation=30,ha='right',fontsize=8)
    for i,value in enumerate(values): axes[0].text(i,value+.00015,f'{value:.5f}',ha='center',fontsize=8)
    axes[1].errorbar(x,[r['importance']['mean_ratio'] for r in large],yerr=[1.96*r['importance']['ratio_standard_error'] for r in large],fmt='o',capsize=3,color='#a16a2b')
    axes[1].axhline(1,color='#666666',ls='--',label='Gaussian reference'); axes[1].set_ylim(0,1.08); axes[1].set_ylabel('Mixed vacuum probability / Gaussian reference'); axes[1].set_title('Monte Carlo vacuum estimates'); axes[1].set_xticks(x,labels,rotation=30,ha='right',fontsize=8); axes[1].legend(fontsize=8,loc='center right')
    axes[1].text(.02,.95,'Bars: 1.96 empirical SE, not rigorous intervals',transform=axes[1].transAxes,fontsize=7.5,va='top')
    z=[r['monte_carlo_standardized_error'] for r in small]; axes[2].scatter(range(16),z,color=['#4b8b67' if r['state_kind']=='gaussian' else '#8e658b' for r in small]); axes[2].axhline(0,color='#666666',lw=.7); axes[2].axhline(1.96,color='#999999',ls=':'); axes[2].axhline(-1.96,color='#999999',ls=':'); axes[2].set_ylim(-3,3); axes[2].set_xlabel('Fixed small-system case'); axes[2].set_ylabel('(Estimate - exact integral) / empirical SE'); axes[2].set_title('Independent small-system calibration')
    fig.suptitle('Iteration 29: finite MPS and Gaussian core through the same noise channel',fontsize=13); fig.tight_layout(); fig.savefig(out/'same_channel_outputs.png',dpi=160); plt.close(fig)
    baseline=np.load(ROOT/'results/iteration29_panel/case_20.npz',allow_pickle=False)['weights']; pref=large[4]['reference_mixed_vacuum']; paired=[]
    for row in large:
        weights=np.load(ROOT/f"results/iteration29_panel/case_{row['sequence']:02d}.npz",allow_pickle=False)['weights']; difference=weights-baseline
        paired.append(dict(family=row['family'],baseline='author_pco',mean_probability_difference=pref*float(np.mean(difference)),paired_empirical_standard_error=pref*float(np.std(difference,ddof=1))/math.sqrt(len(difference)),scope='Derived comparison on the fixed shared samples; not independent samples or a rigorous confidence certificate'))
    save(out/'paired_differences.json',paired)
    assert src.read_bytes()==content
    save(out/'summary.json',dict(analyzer_sha256=digest(src),unchanged_source=True,large_local_bounds=values,small_max_absolute_standardized_error=max(abs(v) for v in z)))


if __name__=='__main__': main()
