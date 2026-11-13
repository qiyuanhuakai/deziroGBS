"""Summarize direct output sampling without treating frequency agreement as TV certification."""
from pathlib import Path
import json,shutil,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from iteration23_support import digest,save

def main():
    out=ROOT/'results/iteration30_analysis';out.mkdir(exist_ok=False);source=Path(__file__);sha=digest(source);shutil.copy2(source,out/source.name)
    rows=json.loads((ROOT/'results/iteration30_panel/rows.json').read_text());small=rows[:16];large=rows[16:];assert len(rows)==22 and all(r['status']=='completed' for r in rows)
    summary=dict(analyzer_sha256=sha,small_samples=16*65536,large_samples=6*4096,total_samples=16*65536+6*4096,maximum_conditional_law_error=max(r['conditional_max_error'] for r in small),small_empirical_tv_range=[min(r['empirical_full_tv'] for r in small),max(r['empirical_full_tv'] for r in small)],small_maximum_standardized_error=max(r['maximum_standardized_error'] for r in small),large_maximum_marginal_standardized_error=max(r['maximum_marginal_standardized_error'] for r in large),large_rows=large,scope='Actual complete binary output strings of specified normalized finite MPS under the prior Gaussian displacement channel. Empirical distances and errors describe finite samples, not Gaussian-target output TV or rigorous floating-point guarantees.')
    fig,axes=plt.subplots(1,3,figsize=(16,4.7),layout='constrained');names=['New 32','New 64','New 128','New 200','Author PCO','Author haf']
    axes[0].bar(names,[r['samples']/r['sampling_seconds'] for r in large],color='#376887');axes[0].set_ylabel('Complete 144-bit strings / second');axes[0].set_title('Sampling throughput (preparation excluded)');axes[0].tick_params(axis='x',rotation=30)
    for i in (16,19,20,21):
        data=np.load(ROOT/f'results/iteration30_panel/case_{i:02d}.npz');axes[1].plot(np.arange(145),data['click_histogram']/4096,label=names[i-16],alpha=.8)
    axes[1].set_xlim(30,105);axes[1].set_xlabel('Total clicks');axes[1].set_ylabel('Observed frequency');axes[1].set_title('Mixed finite-MPS output samples');axes[1].legend(fontsize=8)
    axes[2].plot(np.arange(16),[r['empirical_full_tv'] for r in small],'o');axes[2].set_xlabel('Fixed small-system case');axes[2].set_ylabel('Empirical TV to integrated finite-state law');axes[2].set_title('65,536 samples per small case');axes[2].set_ylim(bottom=0)
    fig.suptitle('Iteration 30: direct displaced-MPS threshold sampling; no output photon cutoff');fig.savefig(out/'direct_threshold_samples.png',dpi=140);plt.close(fig)
    summary['unchanged_source']=digest(source)==sha;assert summary['unchanged_source'];save(out/'summary.json',summary)
    lines=['| Family | Preparation s | Sampling s | Strings/s | Observed mean clicks | Exact finite-state mean | Max marginal z |','|---|---:|---:|---:|---:|---:|---:|']
    for r in large:lines.append(f"| {r['family']} | {r['preparation_seconds']:.3f} | {r['sampling_seconds']:.3f} | {r['samples']/r['sampling_seconds']:.2f} | {r['mean_clicks']:.6f} | {r['expected_mean_clicks']:.6f} | {r['maximum_marginal_standardized_error']:.3f} |")
    (out/'summary_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in summary.items() if k!='large_rows'}))
if __name__=='__main__':main()
